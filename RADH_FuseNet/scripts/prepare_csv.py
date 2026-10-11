#!/usr/bin/env python3
"""Prepare one-subject, one-session CSV recordings into masked window NPZ.

Every file is independent: no window, interpolation, filtering, or alignment
crosses a file boundary. Concrete window/filter choices are user policies.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from radh.preprocessing import align_pair_by_correlation, filter_observed_runs, resample_masked


def number(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except ValueError as error:
        raise ValueError(f"Non-numeric sensor/target value: {value!r}") from error


def prepare(args):
    windows, masks, labels, targets, subjects, timestamps = [], [], [], [], [], []
    summary = []
    for source in args.input:
        path = Path(source)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if not set(args.channels + [args.subject_column]).issubset(columns):
                raise ValueError(f"{path}: missing channels or subject identifier column.")
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError(f"{path}: need at least two rows.")
        subject_ids = {row[args.subject_column].strip() for row in rows}
        if len(subject_ids) != 1 or not next(iter(subject_ids)):
            raise ValueError(f"{path}: each CSV must contain exactly one nonempty subject ID.")
        subject = next(iter(subject_ids))
        if args.timestamp_column in columns:
            times = np.array([number(row[args.timestamp_column]) for row in rows])
            if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1/args.source_hz, rtol=.02, atol=1e-6):
                raise ValueError(f"{path}: timestamps must be uniformly spaced seconds at --source-hz. Insert explicit missing rows for gaps; do not silently compress time.")
            start_time = float(times[0])
        else:
            start_time = 0.0
        x = np.array([[number(row[channel]) for row in rows] for channel in args.channels], dtype=np.float32)
        mask = np.isfinite(x)
        for channel_index, channel in enumerate(args.channels):
            mask_column = f"{channel}_mask"
            if mask_column in columns:
                mask[channel_index] &= np.array([row[mask_column].strip().lower() in {"1", "true", "yes"} for row in rows])
        x = filter_observed_runs(np.where(mask, x, 0), mask, args.source_hz,
                                 tuple(args.pulse_band) if args.pulse_band else None,
                                 args.humidity_detrend_seconds, args.wavelet)
        x, mask = resample_masked(x, mask, args.source_hz, args.target_hz)
        original_y = np.array([number(row.get(args.label_column, "-1")) for row in rows])
        original_y = np.where(np.isfinite(original_y), original_y, -1)
        if not np.all(original_y == np.floor(original_y)) or (original_y < -1).any():
            raise ValueError(f"{path}: labels must be nonnegative integer class IDs or -1.")
        sample_map = np.minimum(np.round(np.arange(x.shape[-1])*args.source_hz/args.target_hz).astype(int), len(rows)-1)
        y = original_y[sample_map].astype(np.int64)
        intensity = np.full(x.shape[-1], np.nan, dtype=np.float32)
        if args.intensity_column:
            if args.intensity_column not in columns:
                raise ValueError(f"{path}: requested intensity column does not exist.")
            values = np.array([number(row[args.intensity_column]) for row in rows], dtype=np.float32)[None, :]
            intensity_x, intensity_mask = resample_masked(values, np.isfinite(values), args.source_hz, args.target_hz)
            intensity = np.where(intensity_mask[0], intensity_x[0], np.nan)
        count = 0
        for start in range(0, x.shape[-1]-args.window+1, args.stride):
            end = start+args.window
            window, observed = x[:, start:end].copy(), mask[:, start:end].copy()
            if observed.mean() < args.minimum_observed:
                continue
            if args.align_max_lag:
                window[1], observed[1], _ = align_pair_by_correlation(window[0], window[1], observed[0], observed[1], args.align_max_lag)
            known = y[start:end][y[start:end] >= 0]
            label = -1
            if len(known):
                unique, counts = np.unique(known, return_counts=True)
                winner = counts.argmax()
                if counts[winner]/args.window >= args.label_purity:
                    label = int(unique[winner])
            future = np.full(args.horizon, np.nan, dtype=np.float32)
            for h in range(args.horizon):
                index = end+h*args.horizon_step
                if index < len(intensity):
                    future[h] = intensity[index]
            windows.append(window)
            masks.append(observed)
            labels.append(label)
            targets.append(future)
            subjects.append(subject)
            timestamps.append(start_time+start/args.target_hz)
            count += 1
        summary.append({"file": str(path), "subject_id": subject, "input_rows": len(rows), "windows": count})
    if not windows:
        raise ValueError("No valid full windows. Check recording lengths, window size, and minimum observed fraction.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.stack(windows).astype(np.float32), mask=np.stack(masks),
        y=np.asarray(labels, dtype=np.int64), target=np.stack(targets), subject_ids=np.asarray(subjects),
        timestamp=np.asarray(timestamps, dtype=np.float64))
    metadata = {"kind": "user_provided_csv", "sources": summary, "policy": vars(args),
        "normalization": "Not applied here; training fits its own training-subject-only robust statistics.",
        "missingness": "No gaps are interpolated. Polyphase anti-aliasing is applied separately to observed runs.",
        "labels": "Integer classes provided by the user; continuous future targets only from the explicit intensity column.",
        "caution": "Filtering and optional alignment are offline preprocessing assumptions, not recovered paper acquisition parameters."}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"保存 {len(windows)} 个窗口、{len(set(subjects))} 位受试者：{output}")


def main():
    p = argparse.ArgumentParser(description="CSV → NPZ；每个文件只能包含一位受试者的一段记录")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--channels", nargs=5, default=["PPG", "PPS", "RSP", "EHS", "RHS"])
    p.add_argument("--subject-column", default="subject_id")
    p.add_argument("--timestamp-column", default="timestamp")
    p.add_argument("--label-column", default="y")
    p.add_argument("--intensity-column", help="真实连续强度列；不提供则 target 全为 NaN")
    p.add_argument("--source-hz", type=float, required=True)
    p.add_argument("--target-hz", type=float, required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--horizon-step", type=int, default=1, help="连续目标之间的重采样样本间隔")
    p.add_argument("--label-purity", type=float, default=.8)
    p.add_argument("--minimum-observed", type=float, default=.25)
    p.add_argument("--pulse-band", nargs=2, type=float, help="可选脉搏带通下限/上限 Hz；论文未给出固定值")
    p.add_argument("--humidity-detrend-seconds", type=float)
    p.add_argument("--wavelet", action="store_true", help="可选，需要 PyWavelets")
    p.add_argument("--align-max-lag", type=int, default=0, help="可选，PPG/PPS 有限互相关对齐的最大样本偏移")
    args = p.parse_args()
    if min(args.source_hz, args.target_hz, args.window, args.stride, args.horizon, args.horizon_step) <= 0:
        p.error("采样率、窗口、步长和预测长度必须大于零")
    prepare(args)


if __name__ == "__main__":
    main()
