from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .data import WindowDataset, load_npz, partition_indices
from .evaluate import collect_predictions, load_checkpoint, resolve_device


def predict_file(data_path: str | Path, checkpoint_path: str | Path, output_path: str | Path,
                 device: str = "cpu", batch_size: int = 16, partition: str = "all") -> Path:
    device_obj = resolve_device(device)
    model, checkpoint, normalizer = load_checkpoint(checkpoint_path, device_obj)
    config = checkpoint["model_config"]
    data = load_npz(data_path, horizon=config["horizon"], num_classes=config["num_classes"])
    indices = np.arange(len(data["x"])) if partition == "all" else partition_indices(data["subject_ids"], checkpoint["split"], partition)
    loader = DataLoader(WindowDataset(data, indices, normalizer), batch_size=batch_size, shuffle=False)
    predictions = collect_predictions(model, loader, device_obj)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["window_index", "subject_id", "true_class", "predicted_class", "uncertainty", "trajectory_valid"]
    fields += [f"probability_{c}" for c in range(config["num_classes"])]
    fields += [f"reliability_{m}" for m in range(5)]
    fields += [name for h in range(config["horizon"]) for name in (f"mean_{h}", f"predictive_variance_{h}")]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(indices)):
            row = {"window_index": int(predictions["index"][i]), "subject_id": predictions["subject_ids"][i],
                   "true_class": int(predictions["y"][i]), "predicted_class": int(predictions["probs"][i].argmax()),
                   "uncertainty": float(predictions["uncertainty"][i].reshape(-1)[0]),
                   "trajectory_valid": bool(predictions["trajectory_valid"][i])}
            row.update({f"probability_{c}": float(predictions["probs"][i, c]) for c in range(config["num_classes"])})
            row.update({f"reliability_{m}": float(predictions["reliability"][i, m]) for m in range(5)})
            for h in range(config["horizon"]):
                row[f"mean_{h}"] = float(predictions["mean"][i, h]) if row["trajectory_valid"] else ""
                row[f"predictive_variance_{h}"] = float(predictions["variance"][i, h]) if row["trajectory_valid"] else ""
            writer.writerow(row)
    return output_path
