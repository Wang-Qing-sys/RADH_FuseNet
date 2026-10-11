from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import RobustNormalizer

CHANNEL_NAMES = ("PPG", "PPS", "RSP", "EHS", "RHS")
CLASS_NAMES = ("activated", "deactivated", "unpleasant", "pleasant", "neutral")


def load_npz(path: str | Path, horizon: int | None = None, num_classes: int | None = None) -> dict[str, np.ndarray]:
    """Load an NPZ without permitting pickle/object deserialization.

    Required: x [N,5,T], subject_ids [N]. Optional: mask, y [N] (-1
    means unlabelled), target [N,H] (NaN means unavailable), timestamp [N].
    Nonfinite input samples are always marked missing and replaced with zero.
    """
    with np.load(path, allow_pickle=False) as archive:
        if "x" not in archive or "subject_ids" not in archive:
            raise ValueError("NPZ requires x and subject_ids; see the documented data contract.")
        data = {key: archive[key] for key in archive.files}
    x = np.asarray(data["x"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 5 or x.shape[0] == 0 or x.shape[2] < 8:
        raise ValueError(f"x must have nonempty shape [N,5,T] with T>=8; got {x.shape}.")
    n = len(x)
    subject_ids = np.asarray(data["subject_ids"])
    if subject_ids.shape != (n,) or subject_ids.dtype.kind not in "USiu":
        raise ValueError("subject_ids must be a length-N integer or Unicode/string array, never object dtype.")
    subject_ids = subject_ids.astype(str)
    if np.any(np.char.str_len(subject_ids) == 0):
        raise ValueError("Empty subject identifiers are not allowed.")
    mask = np.asarray(data.get("mask", np.isfinite(x)), dtype=bool)
    if mask.shape != x.shape:
        raise ValueError("mask must have the same shape as x.")
    mask = mask & np.isfinite(x)
    raw_y = np.asarray(data.get("y", np.full(n, -1)))
    if raw_y.shape != (n,) or not np.all(np.isfinite(raw_y)) or not np.all(raw_y == np.floor(raw_y)):
        raise ValueError("y must be a length-N integer array; use -1 for missing labels.")
    y = raw_y.astype(np.int64)
    if np.any(y < -1) or (num_classes is not None and np.any(y >= num_classes)):
        raise ValueError("Class labels must be -1 or in [0,num_classes).")
    target = np.asarray(data.get("target", np.full((n, horizon or 1), np.nan)), dtype=np.float32)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] != n or target.shape[1] < 1:
        raise ValueError("target must be [N,H]; use NaN for missing continuous labels.")
    if horizon is not None and target.shape[1] != horizon:
        raise ValueError(f"target horizon {target.shape[1]} does not match model horizon {horizon}.")
    target = np.where(np.isfinite(target), target, np.nan)
    result = {"x": np.where(mask, x, 0).astype(np.float32), "mask": mask, "y": y,
              "target": target, "subject_ids": subject_ids}
    if "timestamp" in data:
        if data["timestamp"].shape != (n,):
            raise ValueError("timestamp must be a length-N array.")
        result["timestamp"] = np.asarray(data["timestamp"])
    return result


def validate_split(split: dict[str, list[str]], subject_ids: np.ndarray) -> dict[str, list[str]]:
    """Require all participants exactly once across nonempty train/val/test."""
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Split JSON must contain exactly train, val, and test subject-ID lists.")
    normalized = {part: [str(s) for s in values] for part, values in split.items()}
    for part, values in normalized.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Partition {part} must be nonempty and have no duplicate IDs.")
    parts = [set(normalized[p]) for p in ("train", "val", "test")]
    if any(parts[a] & parts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Subject leakage: train, val, and test IDs overlap.")
    known = set(np.asarray(subject_ids).astype(str).tolist())
    if set.union(*parts) != known:
        raise ValueError("Split must assign every data subject exactly once, with no unknown IDs.")
    return normalized


def split_subjects(subject_ids: np.ndarray, seed: int = 42, val_fraction: float = 0.15,
                   test_fraction: float = 0.15) -> dict[str, list[str]]:
    """Deterministic subject shuffle; no windows from one person cross a split."""
    subjects = np.unique(np.asarray(subject_ids).astype(str))
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for separate train/val/test partitions.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    subjects = np.random.default_rng(seed).permutation(subjects)
    n_val = max(1, round(len(subjects) * val_fraction))
    n_test = max(1, round(len(subjects) * test_fraction))
    if n_val + n_test >= len(subjects):
        raise ValueError("Requested split leaves no training participants; use smaller fractions.")
    split = {"val": subjects[:n_val].tolist(), "test": subjects[n_val:n_val+n_test].tolist(),
             "train": subjects[n_val+n_test:].tolist()}
    return validate_split(split, subject_ids)


def partition_indices(subject_ids: np.ndarray, split: dict[str, list[str]], partition: str) -> np.ndarray:
    if partition not in split:
        raise ValueError(f"Unknown partition {partition}.")
    return np.flatnonzero(np.isin(np.asarray(subject_ids).astype(str), split[partition]))


def dataset_fingerprint(data: dict[str, np.ndarray]) -> str:
    """Order-sensitive checksum protects checkpoint split reuse on another file."""
    digest = hashlib.sha256()
    for key in ("x", "mask", "y", "target", "subject_ids"):
        values = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


class WindowDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], indices: np.ndarray,
                 normalizer: RobustNormalizer):
        self.data = data
        self.indices = np.asarray(indices, np.int64)
        self.x = normalizer.transform(data["x"][self.indices], data["mask"][self.indices])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.indices[index]
        return {"x": torch.from_numpy(self.x[index]),
                "mask": torch.from_numpy(self.data["mask"][original]),
                "y": torch.tensor(int(self.data["y"][original]), dtype=torch.long),
                "target": torch.from_numpy(self.data["target"][original]),
                "subject_id": str(self.data["subject_ids"][original]),
                "index": int(original)}


def make_demo_data(path: str | Path, subjects: int = 6, windows_per_subject: int = 5,
                   samples: int = 128, horizon: int = 4, num_classes: int = 5,
                   seed: int = 42, sample_hz: float = 32.0) -> Path:
    """Create SIMULATED signals for an executable smoke test, never study results.

    Continuous targets come from the simulated latent trajectory, not a numeric
    remapping of class IDs. Only this demonstration has known synthetic truth.
    """
    if subjects < 3 or windows_per_subject < 1 or samples < 8 or horizon < 1:
        raise ValueError("Demo needs >=3 subjects, >=1 window/subject, T>=8, and H>=1.")
    rng = np.random.default_rng(seed)
    x, masks, labels, targets, ids = [], [], [], [], []
    t = np.arange(samples, dtype=np.float32) / sample_hz
    for subject in range(subjects):
        individual_offset = rng.normal(0, 0.12, 5)
        for window in range(windows_per_subject):
            label = window % num_classes
            latent = float(rng.uniform(0.1, 0.9))
            slope = float(rng.uniform(-0.035, 0.035))
            phase = float(rng.uniform(0, 2 * np.pi))
            heart = 0.9 + 0.13 * label + 0.15 * latent
            resp = 0.18 + 0.10 * latent
            ppg = np.sin(2*np.pi*heart*t+phase) + .22*np.sin(4*np.pi*heart*t+phase)
            pps = .8*np.sin(2*np.pi*heart*t+phase-.2) + .15*np.sin(6*np.pi*heart*t+phase)
            ehs = .3*np.sin(2*np.pi*.025*t+phase) + .02*t
            clean_resp = (.8 + .2*latent)*np.sin(2*np.pi*resp*t+phase)
            rhs = clean_resp + .45*ehs
            rsp = clean_resp + .1*rhs
            values = np.stack([ppg, pps, rsp, ehs, rhs]) + individual_offset[:, None]
            values += rng.normal(0, .035, values.shape)
            observed = rng.random(values.shape) > .015
            if window == 0 and subject % 3 == 0:
                observed[1, samples//3:samples//3+max(2, samples//12)] = False
            future = np.clip(latent + slope*np.arange(1, horizon+1), 0, 1).astype(np.float32)
            if window == 1:
                future[-1] = np.nan  # exercise partially missing regression labels
            x.append(np.where(observed, values, 0).astype(np.float32))
            masks.append(observed)
            labels.append(label)
            targets.append(future)
            ids.append(f"demo_subject_{subject:02d}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.stack(x), mask=np.stack(masks), y=np.asarray(labels, np.int64),
                        target=np.stack(targets), subject_ids=np.asarray(ids, dtype="U32"))
    path.with_suffix(".json").write_text(json.dumps({"kind": "synthetic_smoke_test_only", "seed": seed,
        "subjects": subjects, "windows_per_subject": windows_per_subject, "sample_hz": sample_hz,
        "channels": CHANNEL_NAMES, "warning": "Simulated data; not the manuscript dataset and not scientific replication."}, indent=2), encoding="utf-8")
    return path
