from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import corrupt_view
from .data import WindowDataset, dataset_fingerprint, load_npz, partition_indices, validate_split
from .preprocessing import RobustNormalizer


def classification_metrics(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> dict[str, Any]:
    """Macro metrics include all configured classes; missing labels are excluded."""
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    valid = (y >= 0) & (y < probs.shape[1])
    y, probs = y[valid].astype(int), probs[valid]
    if not len(y):
        return {"n_labelled": 0, "accuracy": None, "macro_precision": None, "macro_recall": None,
                "macro_f1": None, "balanced_accuracy": None, "ece": None, "nll": None,
                "brier": None, "risk_coverage_auc": None}
    probs = np.clip(probs, 1e-12, 1)
    probs /= probs.sum(axis=1, keepdims=True)
    classes = probs.shape[1]
    prediction = probs.argmax(axis=1)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    tp = matrix.diagonal()
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(classes, float), where=matrix.sum(axis=0)>0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros(classes, float), where=matrix.sum(axis=1)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(classes, float), where=precision+recall>0)
    confidence = probs.max(axis=1)
    correct = (prediction == y).astype(float)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, bins+1)[:-1], np.linspace(0, 1, bins+1)[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            ece += inside.mean()*abs(correct[inside].mean()-confidence[inside].mean())
    order = np.argsort(-confidence, kind="stable")
    risk = np.cumsum(1-correct[order])/np.arange(1, len(y)+1)
    return {"n_labelled": int(len(y)), "accuracy": float(correct.mean()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "ece": float(ece), "ece_bins": bins,
            "nll": float(-np.log(probs[np.arange(len(y)), y]).mean()),
            "brier": float(np.square(probs-np.eye(classes)[y]).sum(axis=1).mean()),
            "risk_coverage_auc": float(risk.mean()), "confusion_matrix": matrix.tolist(),
            "class_support": matrix.sum(axis=1).tolist(),
            "absent_classes": np.flatnonzero(matrix.sum(axis=1)==0).tolist(),
            "macro_convention": "all configured classes; zero division returns 0",
            "risk_coverage_convention": "mean cumulative error, descending maximum class probability"}


def regression_metrics(target: np.ndarray, mean: np.ndarray, variance: np.ndarray | None = None) -> dict[str, Any]:
    valid = np.isfinite(target)
    if not valid.any():
        return {"n_regression_targets": 0, "regression_mae": None, "regression_rmse": None}
    error = mean[valid] - target[valid]
    result = {"n_regression_targets": int(valid.sum()), "regression_mae": float(np.abs(error).mean()),
              "regression_rmse": float(np.sqrt(np.square(error).mean()))}
    if variance is not None:
        result["mean_predictive_variance"] = float(variance[valid].mean())
    return result


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    return torch.device(device)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    from .model import ModelConfig, RADHFuseNet
    # Only tensor/primitive state is written; no arbitrary object loading fallback.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = RADHFuseNet(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint, RobustNormalizer.from_dict(checkpoint["normalizer"])


def _nig_outputs(output: dict) -> tuple[torch.Tensor, torch.Tensor]:
    nig = output["nig"]
    mu = nig.get("mu", nig.get("gamma"))
    nu = nig.get("nu", nig.get("v"))
    alpha, beta = nig["alpha"], nig["beta"]
    variance = beta*(1+nu)/(nu*(alpha-1).clamp_min(1e-6))
    return mu, variance


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: torch.device,
                        corruption: str = "clean", severity: float = 1.0,
                        missing_channel: int | None = None, seed: int = 42) -> dict[str, np.ndarray]:
    model.eval()
    arrays: dict[str, list] = {key: [] for key in ("probs", "uncertainty", "reliability", "mean", "variance", "y", "target", "index", "subject_ids", "trajectory_valid")}
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        x, mask = batch["x"].to(device), batch["mask"].to(device)
        if corruption != "clean":
            x, mask = corrupt_view(x, mask, severity, corruption,
                humidity_enabled=getattr(model.config, "humidity_compensation", True), generator=generator)
        if missing_channel is not None:
            x, mask = x.clone(), mask.clone()
            x[:, missing_channel], mask[:, missing_channel] = 0, False
        output = model(x, mask)
        mean, variance = _nig_outputs(output)
        for key, value in (("probs", output["probs"]), ("uncertainty", output["uncertainty"]),
                           ("reliability", output["reliability"]), ("mean", mean), ("variance", variance),
                           ("trajectory_valid", output.get("trajectory_valid", mask[:, :3].any(dim=(1, 2))))):
            arrays[key].append(value.detach().cpu().numpy())
        for key in ("y", "target", "index"):
            arrays[key].append(batch[key].cpu().numpy())
        arrays["subject_ids"].append(np.asarray(batch["subject_id"]))
    if not arrays["y"]:
        raise ValueError("No windows in requested partition.")
    return {key: np.concatenate(values, axis=0) for key, values in arrays.items()}


def summarize_predictions(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = classification_metrics(predictions["y"], predictions["probs"])
    valid_trajectory = predictions.get("trajectory_valid", np.ones(len(predictions["y"]), dtype=bool))
    target = np.where(valid_trajectory[:, None], predictions["target"], np.nan)
    metrics.update(regression_metrics(target, predictions["mean"], predictions["variance"]))
    metrics["n_invalid_trajectories"] = int((~valid_trajectory).sum())
    metrics.update(n_windows=len(predictions["y"]), n_subjects=len(np.unique(predictions["subject_ids"])),
                   mean_uncertainty=float(predictions["uncertainty"].mean()),
                   mean_reliability=predictions["reliability"].mean(axis=0).tolist())
    return metrics


def evaluate_checkpoint(data_path: str | Path, checkpoint_path: str | Path,
                        output_path: str | Path, device: str = "cpu", batch_size: int = 16,
                        partition: str = "test", robustness: bool = False) -> dict[str, Any]:
    device_obj = resolve_device(device)
    model, checkpoint, normalizer = load_checkpoint(checkpoint_path, device_obj)
    config = checkpoint["model_config"]
    data = load_npz(data_path, horizon=config["horizon"], num_classes=config["num_classes"])
    if dataset_fingerprint(data) != checkpoint["dataset_fingerprint"]:
        raise ValueError("Evaluation data differs from the training NPZ. Use predict for new data; held-out evaluation requires the original subject split.")
    split = validate_split(checkpoint["split"], data["subject_ids"])
    if partition == "all":
        raise ValueError("Evaluation requires an explicit train/val/test partition; default test.")
    indices = partition_indices(data["subject_ids"], split, partition)
    loader = DataLoader(WindowDataset(data, indices, normalizer), batch_size=batch_size, shuffle=False)
    predictions = collect_predictions(model, loader, device_obj)
    metrics = {"partition": partition, "checkpoint": str(checkpoint_path),
               "synthetic_demo": bool(checkpoint.get("synthetic_demo", False)),
               "subject_ids": split[partition], "clean": summarize_predictions(predictions)}
    if robustness:
        metrics["robustness"] = {}
        conditions = ["gaussian", "spikes", "dropout", "drift", "shift"]
        if config.get("humidity_compensation", True):
            conditions.append("humidity")
        for kind in conditions:
            result = collect_predictions(model, loader, device_obj, kind, seed=checkpoint["seed"])
            metrics["robustness"][kind] = summarize_predictions(result)
        for channel in range(5):
            result = collect_predictions(model, loader, device_obj, missing_channel=channel)
            metrics["robustness"][f"missing_channel_{channel}"] = summarize_predictions(result)
        metrics["robustness_policy"] = "Fixed normalized-unit engineering severities; no retraining or test-set tuning. See radh/augment.py."
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return metrics
