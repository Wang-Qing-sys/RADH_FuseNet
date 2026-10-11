from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import corrupt_view, modality_dropout, quality_views, span_mask
from .data import WindowDataset, dataset_fingerprint, load_npz, partition_indices, split_subjects, validate_split
from .evaluate import collect_predictions, resolve_device, summarize_predictions
from .preprocessing import RobustNormalizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact numerical identity across devices/PyTorch versions is not guaranteed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not {"model", "training"}.issubset(config):
        raise ValueError("Configuration requires model and training dictionaries.")
    return config


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int):
    total = max(1, epochs*steps_per_epoch)
    warmup = max(0, warmup_epochs*steps_per_epoch)
    def multiplier(step):
        if step < warmup:
            return (step+1)/max(1, warmup)
        progress = (step-warmup)/max(1, total-warmup)
        return .5*(1+math.cos(math.pi*min(1, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _make_loss_batch(batch: dict, config: dict, stage: str):
    x, mask = batch["x"], batch["mask"]
    result = dict(batch)
    result["reconstruction_target"] = x
    result["target_mask"] = mask
    if stage == "pretrain":
        result["x"], result["mask"], result["reconstruction_mask"] = span_mask(
            x, mask, config.get("span_mask_fraction", .15), config.get("span_length", 8))
    else:
        corrupted, observed = corrupt_view(x, mask, config.get("corruption_severity", .35),
                                          humidity_enabled=config.get("humidity_enabled", True))
        result["x"], result["mask"] = modality_dropout(corrupted, observed, config.get("modality_dropout", .2))
        result["reconstruction_mask"] = mask
    return result


def _forward_loss(model, criterion, batch, train_config, stage, epoch):
    clean = model(batch["x"], batch["mask"])
    mild, strong = quality_views(batch["x"], batch["mask"], train_config.get("humidity_enabled", True))
    mild_outputs, strong_outputs = model(*mild), model(*strong)
    loss_batch = _make_loss_batch(batch, train_config, stage)
    outputs = model(loss_batch["x"], loss_batch["mask"])
    losses = criterion(outputs, loss_batch, stage=stage, epoch=epoch,
                       clean_outputs=clean, mild_outputs=mild_outputs, strong_outputs=strong_outputs)
    return losses


@torch.no_grad()
def _pretrain_validation(model, criterion, loader, device, config, epoch) -> float:
    model.eval()
    total, count = 0.0, 0
    # A fixed validation corruption stream gives comparable epochs and does not
    # advance the training RNG. No validation statistics update the normalizer.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(104729)
        for batch in loader:
            batch = _move_batch(batch, device)
            loss = _forward_loss(model, criterion, batch, config, "pretrain", epoch)["total"]
            total += float(loss)*len(batch["x"])
            count += len(batch["x"])
    return total/max(1, count)


def train_model(data_path: str | Path, config_path: str | Path, output_dir: str | Path,
                device: str = "cpu", split_path: str | Path | None = None,
                synthetic_demo: bool = False) -> dict[str, Any]:
    from .losses import RADHLoss
    from .model import ModelConfig, RADHFuseNet

    config = read_config(config_path)
    train_config = dict(config["training"])
    seed = int(train_config.get("seed", 42))
    seed_everything(seed)
    torch.set_num_threads(int(train_config.get("cpu_threads", 2)))
    device_obj = resolve_device(device)
    model_config = ModelConfig(**config["model"])
    model = RADHFuseNet(model_config).to(device_obj)
    data = load_npz(data_path, model_config.horizon, model_config.num_classes)
    if split_path:
        split = validate_split(json.loads(Path(split_path).read_text(encoding="utf-8")), data["subject_ids"])
    else:
        split = split_subjects(data["subject_ids"], seed, train_config.get("val_fraction", .15), train_config.get("test_fraction", .15))
    train_indices = partition_indices(data["subject_ids"], split, "train")
    val_indices = partition_indices(data["subject_ids"], split, "val")
    if train_config.get("finetune_epochs", 1) and not np.any(data["y"][train_indices] >= 0):
        raise ValueError("Supervised training requires some class labels in the training partition.")
    if train_config.get("finetune_epochs", 1) and not np.any(data["y"][val_indices] >= 0):
        raise ValueError("Validation macro-F1 selection requires labelled validation windows.")
    normalizer = RobustNormalizer.fit(data["x"][train_indices], data["mask"][train_indices], clip=train_config.get("normalization_clip", 20.0))
    batch_size = int(train_config.get("batch_size", 16))
    generator = torch.Generator().manual_seed(seed)
    loaders = {"train": DataLoader(WindowDataset(data, train_indices, normalizer), batch_size=batch_size,
                  shuffle=True, num_workers=0, generator=generator),
               "val": DataLoader(WindowDataset(data, val_indices, normalizer), batch_size=batch_size,
                  shuffle=False, num_workers=0)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config["humidity_enabled"] = model_config.humidity_compensation
    criterion = RADHLoss(**config.get("loss", {}))
    fingerprint = dataset_fingerprint(data)
    (output_dir/"split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (output_dir/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history: list[dict[str, Any]] = []
    global_step = 0
    best_score = -float("inf")
    best_nll = float("inf")
    best_pretrain = float("inf")
    pretrain_epochs = int(train_config.get("pretrain_epochs", 1))
    finetune_epochs = int(train_config.get("finetune_epochs", 1))
    if pretrain_epochs < 0 or finetune_epochs < 1:
        raise ValueError("pretrain_epochs must be >=0 and finetune_epochs >=1.")

    def checkpoint(stage, epoch, optimizer, scheduler):
        return {"format_version": 1, "model_state": model.state_dict(),
                "model_config": asdict(model_config), "config": config,
                "normalizer": normalizer.to_dict(), "normalizer_fit_partition": "train",
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "stage": stage, "epoch": epoch, "global_step": global_step,
                "best_val_macro_f1": best_score if math.isfinite(best_score) else None,
                "best_pretrain_val_loss": best_pretrain if math.isfinite(best_pretrain) else None,
                "split": split, "seed": seed, "dataset_fingerprint": fingerprint,
                "synthetic_demo": synthetic_demo, "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "loader_rng_state": generator.get_state(),
                "torch_version": str(torch.__version__)}

    for stage, epochs in (("pretrain", pretrain_epochs), ("supervised", finetune_epochs)):
        if not epochs:
            continue
        if stage == "supervised" and pretrain_epochs:
            selected = torch.load(output_dir/"best_pretrain.pt", map_location=device_obj, weights_only=True)
            model.load_state_dict(selected["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config.get("learning_rate", 2e-4)),
                                      weight_decay=float(train_config.get("weight_decay", .01)))
        scheduler = _scheduler(optimizer, len(loaders["train"]), epochs, int(train_config.get("warmup_epochs", 5)))
        for epoch in range(epochs):
            model.train()
            totals: dict[str, float] = {}
            samples, max_grad_norm = 0, 0.0
            for batch in loaders["train"]:
                batch = _move_batch(batch, device_obj)
                optimizer.zero_grad(set_to_none=True)
                losses = _forward_loss(model, criterion, batch, train_config, stage, epoch)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(f"Nonfinite {stage} training loss at step {global_step}.")
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("gradient_clip", 1.0)), error_if_nonfinite=True)
                max_grad_norm = max(max_grad_norm, float(norm))
                optimizer.step()
                scheduler.step()
                global_step += 1
                count = len(batch["x"])
                samples += count
                for name, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        totals[name] = totals.get(name, 0.0) + float(value.detach())*count
            row: dict[str, Any] = {"stage": stage, "epoch": epoch+1, "global_step": global_step,
                "optimizer_steps": len(loaders["train"]), "learning_rate": optimizer.param_groups[0]["lr"],
                "max_unclipped_gradient_norm": max_grad_norm,
                "train": {name: total/samples for name, total in totals.items()},
                "loss_status": getattr(criterion, "last_status", {})}
            if stage == "pretrain":
                val_loss = _pretrain_validation(model, criterion, loaders["val"], device_obj, train_config, epoch)
                row["validation"] = {"pretrain_total_loss": val_loss}
                improved = val_loss < best_pretrain
                best_pretrain = min(best_pretrain, val_loss)
                history.append(row)
                if improved:
                    torch.save(checkpoint(stage, epoch, optimizer, scheduler), output_dir/"best_pretrain.pt")
            else:
                validation = summarize_predictions(collect_predictions(model, loaders["val"], device_obj))
                row["validation"] = validation
                score, nll = float(validation["macro_f1"]), float(validation["nll"])
                improved = score > best_score or (score == best_score and nll < best_nll)
                if improved:
                    best_score, best_nll = score, nll
                history.append(row)
                if improved:
                    torch.save(checkpoint(stage, epoch, optimizer, scheduler), output_dir/"best.pt")
            torch.save(checkpoint(stage, epoch, optimizer, scheduler), output_dir/"last.pt")
            (output_dir/"history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"stage": stage, "epoch": epoch+1, "train_loss": row["train"]["total"], "validation": row["validation"]}, ensure_ascii=False), flush=True)
    return {"checkpoint": str(output_dir/"best.pt"), "pretrain_checkpoint": str(output_dir/"best_pretrain.pt") if pretrain_epochs else None,
            "best_val_macro_f1": best_score, "global_step": global_step, "split": split,
            "synthetic_demo": synthetic_demo}
