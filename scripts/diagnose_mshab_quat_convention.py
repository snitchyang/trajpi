#!/usr/bin/env python3
"""Diagnose quaternion convention mismatch in websocket Pi trajectory traces.

This script compares predicted and actual trajectories under multiple quaternion
reordering hypotheses to infer whether one side is stored as ``wxyz`` while the
other is interpreted as ``xyzw``.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def _apply_reorder(traj: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(traj, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] < 7:
        raise ValueError(f"Trajectory must be (T, D>=7), got {out.shape}")
    quat = out[:, 3:7]
    if mode == "identity":
        pass
    elif mode == "wxyz_to_xyzw":
        quat = _wxyz_to_xyzw(quat)
    elif mode == "xyzw_to_wxyz":
        quat = _xyzw_to_wxyz(quat)
    else:
        raise ValueError(f"Unknown reorder mode: {mode}")
    out[:, 3:7] = quat
    return out


def _normalize_quat(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, eps)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion angular distance in degrees, invariant to sign."""
    q1n = _normalize_quat(q1)
    q2n = _normalize_quat(q2)
    dots = np.sum(q1n * q2n, axis=-1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


class HypothesisStats:
    def __init__(self, name):
        self.name = name
        self.pos_l1_sum = 0.0
        self.quat_l1_sum = 0.0
        self.quat_ang_deg_sum = 0.0
        self.rows = 0
        self.chunks = 0

    def add(self, pos_l1: np.ndarray, quat_l1: np.ndarray, quat_ang_deg: np.ndarray) -> None:
        self.pos_l1_sum += float(np.sum(pos_l1))
        self.quat_l1_sum += float(np.sum(quat_l1))
        self.quat_ang_deg_sum += float(np.sum(quat_ang_deg))
        self.rows += int(len(pos_l1))
        self.chunks += 1

    def summary(self) -> tuple[float, float, float]:
        if self.rows == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            self.pos_l1_sum / self.rows,
            self.quat_l1_sum / self.rows,
            self.quat_ang_deg_sum / self.rows,
        )


def _iter_pairs(records: Iterable[dict], pred_key: str, actual_key: str):
    for rec in records:
        pred = rec.get(pred_key)
        actual = rec.get(actual_key)
        if not isinstance(pred, np.ndarray) or not isinstance(actual, np.ndarray):
            continue
        if pred.ndim != 2 or actual.ndim != 2:
            continue
        if pred.shape[1] < 7 or actual.shape[1] < 7:
            continue
        h = min(len(pred), len(actual))
        if h <= 0:
            continue
        yield pred[:h], actual[:h]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True, help="Path to websocket_pi_inference_trace.pt")
    p.add_argument("--pred-key", type=str, default="pred_traj", help="Predicted trajectory key in each record.")
    p.add_argument("--actual-key", type=str, default="actual_traj", help="Actual trajectory key in each record.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trace_obj = torch.load(args.trace.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(trace_obj, dict) or "records" not in trace_obj or not isinstance(trace_obj["records"], list):
        raise ValueError(f"Invalid trace format: {args.trace}")

    hypotheses = {
        "none": ("identity", "identity"),
        "pred_wxyz_to_xyzw": ("wxyz_to_xyzw", "identity"),
        "actual_wxyz_to_xyzw": ("identity", "wxyz_to_xyzw"),
        "both_wxyz_to_xyzw": ("wxyz_to_xyzw", "wxyz_to_xyzw"),
    }
    stats = {name: HypothesisStats(name=name) for name in hypotheses}

    for pred, actual in _iter_pairs(trace_obj["records"], args.pred_key, args.actual_key):
        for name, (pred_mode, actual_mode) in hypotheses.items():
            p = _apply_reorder(pred, pred_mode)
            a = _apply_reorder(actual, actual_mode)
            pos_l1 = np.mean(np.abs(p[:, :3] - a[:, :3]), axis=-1)
            quat_l1 = np.mean(np.abs(p[:, 3:7] - a[:, 3:7]), axis=-1)
            quat_ang_deg = _quat_angle_deg(p[:, 3:7], a[:, 3:7])
            stats[name].add(pos_l1=pos_l1, quat_l1=quat_l1, quat_ang_deg=quat_ang_deg)

    valid = [s for s in stats.values() if s.rows > 0]
    if not valid:
        raise ValueError(
            f"No valid trajectory pairs found for pred_key={args.pred_key!r}, actual_key={args.actual_key!r}."
        )

    print(f"Trace: {args.trace}")
    print(f"Pairs: {sum(1 for _ in _iter_pairs(trace_obj['records'], args.pred_key, args.actual_key))}")
    print("Metric units: pos_l1 (meters), quat_l1 (raw), quat_ang_deg (degrees)")
    print("")
    print(
        f"{'hypothesis':28s} {'rows':>10s} {'chunks':>8s} {'pos_l1':>12s} {'quat_l1':>12s} {'quat_ang_deg':>14s}"
    )
    print("-" * 90)
    for name in hypotheses:
        s = stats[name]
        pos_l1, quat_l1, quat_ang = s.summary()
        print(f"{name:28s} {s.rows:10d} {s.chunks:8d} {pos_l1:12.6g} {quat_l1:12.6g} {quat_ang:14.6g}")

    best_by_quat_ang = min(valid, key=lambda s: s.summary()[2])
    best_by_quat_l1 = min(valid, key=lambda s: s.summary()[1])
    print("")
    print(f"Best by quat angle: {best_by_quat_ang.name}")
    print(f"Best by quat L1:    {best_by_quat_l1.name}")

    if best_by_quat_ang.name == "pred_wxyz_to_xyzw":
        print("Inference: pred trajectories likely stored as wxyz while actual is treated as xyzw.")
    elif best_by_quat_ang.name == "actual_wxyz_to_xyzw":
        print("Inference: actual trajectories likely stored as wxyz while pred is treated as xyzw.")
    elif best_by_quat_ang.name == "both_wxyz_to_xyzw":
        print("Inference: both sides likely use wxyz but are currently interpreted as xyzw.")
    else:
        print("Inference: no reorder looks best; mismatch may come from frame alignment or dynamics, not quat order.")


if __name__ == "__main__":
    main()
