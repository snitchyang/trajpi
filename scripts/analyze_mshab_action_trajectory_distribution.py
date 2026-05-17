#!/usr/bin/env python3
"""
Analyze action and trajectory distributions for an MSHAB LeRobot dataset.

The script mirrors the config-defined dataset loading path:
  - config -> DataConfig -> LeRobotDataset(delta_timestamps=action_sequence_keys)
  - optional PromptFromLeRobotTask
  - repack_transforms.inputs, including MshabTcpBaseToWorldActions when traj_action=True

It aggregates chunks from N samples and writes plots + stats under --out-dir:
  - raw action distributions and temporal mean/std over horizon
  - tcp_pose_wrt_base distributions
  - transformed action / world-TCP trajectory distributions
  - relative trajectory XY paths and endpoint scatter
  - JSON summary statistics

Example:
  python scripts/analyze_mshab_action_trajectory_distribution.py pi05_mstraj \\
    --num-samples 5000 --sample-stride 5 --out-dir analysis/pi05_mstraj_dist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPT = Path(__file__).resolve()
_OPI_ROOT = _SCRIPT.parent.parent
_OPI_SRC = _OPI_ROOT / "src"
if _OPI_SRC.is_dir() and str(_OPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPI_SRC))

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

from openpi import transforms as _transforms  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402


def _to_numpy_tree(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _to_numpy_tree(v) for k, v in data.items()}
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)


def _apply_with_prompt(fn: _transforms.DataTransformFn, raw: dict) -> dict:
    data = _to_numpy_tree(raw)
    if "prompt" not in data:
        data["prompt"] = ""
    return _to_numpy_tree(fn(data))


def _stack_chunks(chunks: list[np.ndarray]) -> np.ndarray | None:
    if not chunks:
        return None
    return np.stack(chunks, axis=0)


def _flatten_time(chunks: np.ndarray | None) -> np.ndarray | None:
    if chunks is None:
        return None
    if chunks.ndim != 3:
        return None
    return chunks.reshape(-1, chunks.shape[-1])


def _dim_names(prefix: str, dim: int) -> list[str]:
    names = [f"{prefix}_{i}" for i in range(dim)]
    if prefix == "raw_action" and dim >= 13:
        names[7] = "raw_gripper"
        names[11] = "raw_base_v"
        names[12] = "raw_base_w"
    if prefix in {"tcp_base", "traj_action"} and dim >= 7:
        names[:7] = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw"]
    if prefix == "traj_action" and dim >= 8:
        names[7] = "traj_gripper"
    return names


def _safe_stats(arr: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if arr.size == 0:
        return stats
    for i, name in enumerate(names):
        x = arr[:, i].astype(np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        stats[name] = {
            "count": float(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "q01": float(np.quantile(x, 0.01)),
            "q05": float(np.quantile(x, 0.05)),
            "q50": float(np.quantile(x, 0.50)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
        }
    return stats


def _plot_hist_grid(arr: np.ndarray | None, names: list[str], title: str, path: Path, bins: int) -> None:
    if arr is None or arr.size == 0:
        return
    dim = arr.shape[-1]
    cols = min(4, dim)
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), squeeze=False, layout="tight")
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= dim:
            ax.axis("off")
            continue
        x = arr[:, i]
        x = x[np.isfinite(x)]
        ax.hist(x, bins=bins, color="C0", alpha=0.8)
        ax.set_title(names[i], fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_temporal_mean_std(chunks: np.ndarray | None, names: list[str], title: str, path: Path) -> None:
    if chunks is None or chunks.size == 0 or chunks.ndim != 3:
        return
    dim = chunks.shape[-1]
    cols = min(4, dim)
    rows = int(np.ceil(dim / cols))
    t = np.arange(chunks.shape[1])
    mean = np.nanmean(chunks, axis=0)
    std = np.nanstd(chunks, axis=0)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), squeeze=False, layout="tight")
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= dim:
            ax.axis("off")
            continue
        ax.plot(t, mean[:, i], color="C1", label="mean")
        ax.fill_between(t, mean[:, i] - std[:, i], mean[:, i] + std[:, i], color="C1", alpha=0.25, label="+/- std")
        ax.set_title(names[i], fontsize=9)
        ax.set_xlabel("horizon step")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_corr(arr: np.ndarray | None, names: list[str], title: str, path: Path) -> None:
    if arr is None or arr.shape[0] < 2 or arr.shape[1] < 2:
        return
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if arr.shape[0] < 2:
        return
    corr = np.corrcoef(arr.T)
    fig, ax = plt.subplots(figsize=(0.65 * len(names) + 3, 0.65 * len(names) + 2), layout="tight")
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(names)), names, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_views(traj_chunks: np.ndarray | None, out_dir: Path, max_paths: int, bins: int) -> dict[str, Any]:
    if traj_chunks is None or traj_chunks.size == 0 or traj_chunks.shape[-1] < 3:
        return {}

    xyz = traj_chunks[:, :, :3]
    rel = xyz - xyz[:, :1, :]
    endpoints = rel[:, -1, :]
    path_len = np.linalg.norm(np.diff(rel, axis=1), axis=-1).sum(axis=1)
    endpoint_norm = np.linalg.norm(endpoints, axis=-1)

    n = rel.shape[0]
    keep = np.linspace(0, n - 1, min(max_paths, n), dtype=int)

    fig, ax = plt.subplots(figsize=(6, 6), layout="tight")
    for i in keep:
        ax.plot(rel[i, :, 0], rel[i, :, 1], color="C0", alpha=0.12, linewidth=0.8)
    ax.scatter(rel[keep, 0, 0], rel[keep, 0, 1], s=10, color="green", label="start")
    ax.scatter(rel[keep, -1, 0], rel[keep, -1, 1], s=10, color="red", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("relative x (m)")
    ax.set_ylabel("relative y (m)")
    ax.set_title("Relative TCP XY trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "trajectory_relative_xy_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), layout="tight")
    sc = ax.scatter(endpoints[:, 0], endpoints[:, 1], c=endpoint_norm, s=8, cmap="viridis", alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("endpoint delta x (m)")
    ax.set_ylabel("endpoint delta y (m)")
    ax.set_title("Relative trajectory endpoints")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="endpoint |delta xyz| (m)")
    fig.savefig(out_dir / "trajectory_endpoint_xy_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="tight")
    axes[0].hist(path_len, bins=bins, color="C2", alpha=0.85)
    axes[0].set_title("trajectory path length")
    axes[0].set_xlabel("sum ||delta step|| (m)")
    axes[0].grid(True, alpha=0.25)
    axes[1].hist(endpoint_norm, bins=bins, color="C3", alpha=0.85)
    axes[1].set_title("endpoint displacement")
    axes[1].set_xlabel("||endpoint - start|| (m)")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(out_dir / "trajectory_length_endpoint_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rel_flat = rel.reshape(-1, 3)
    _plot_hist_grid(
        rel_flat,
        ["rel_x", "rel_y", "rel_z"],
        "Relative TCP xyz distribution over all horizon steps",
        out_dir / "trajectory_relative_xyz_hist.png",
        bins,
    )

    return {
        "path_length": {
            "mean": float(np.mean(path_len)),
            "std": float(np.std(path_len)),
            "q05": float(np.quantile(path_len, 0.05)),
            "q50": float(np.quantile(path_len, 0.50)),
            "q95": float(np.quantile(path_len, 0.95)),
        },
        "endpoint_norm": {
            "mean": float(np.mean(endpoint_norm)),
            "std": float(np.std(endpoint_norm)),
            "q05": float(np.quantile(endpoint_norm, 0.05)),
            "q50": float(np.quantile(endpoint_norm, 0.50)),
            "q95": float(np.quantile(endpoint_norm, 0.95)),
        },
    }


def _choose_indices(length: int, start: int, num: int, stride: int, random_samples: bool, seed: int) -> list[int]:
    if random_samples:
        rng = np.random.default_rng(seed)
        count = min(num, max(0, length))
        return sorted(int(i) for i in rng.choice(length, size=count, replace=False))
    return [i for i in (start + stride * j for j in range(num)) if 0 <= i < length]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_name", help="Train config name, e.g. pi05_mstraj")
    parser.add_argument("--repo-id", type=str, default=None, help="Override DataConfig.repo_id.")
    parser.add_argument("--out-dir", type=str, default="mshab_action_traj_analysis", help="Directory for plots/stats.")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of chunk samples to try.")
    parser.add_argument("--start-index", type=int, default=0, help="First global dataset index in sequential mode.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Stride in sequential mode.")
    parser.add_argument("--random-samples", action="store_true", help="Sample global indices uniformly instead of sequentially.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --random-samples.")
    parser.add_argument("--action-horizon", type=int, default=None, help="Override config.model.action_horizon.")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bin count.")
    parser.add_argument("--max-plot-trajectories", type=int, default=300, help="Max XY trajectories drawn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = args.repo_id or data_config.repo_id
    if repo_id is None:
        raise SystemExit("No repo_id in config; pass --repo-id.")

    action_horizon = int(args.action_horizon) if args.action_horizon is not None else int(train_config.model.action_horizon)
    meta = lerobot_dataset.LeRobotDatasetMetadata(str(repo_id))
    fps = float(meta.fps) if meta.fps else 20.0
    delta_timestamps = {
        key: [t / fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }

    dataset = lerobot_dataset.LeRobotDataset(str(repo_id), delta_timestamps=delta_timestamps)
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])

    repack_pipeline = _transforms.compose(list(data_config.repack_transforms.inputs))
    indices = _choose_indices(
        len(dataset),
        int(args.start_index),
        int(args.num_samples),
        max(1, int(args.sample_stride)),
        bool(args.random_samples),
        int(args.seed),
    )

    raw_action_chunks: list[np.ndarray] = []
    tcp_base_chunks: list[np.ndarray] = []
    traj_action_chunks: list[np.ndarray] = []
    task_counts: dict[str, int] = {}
    skipped: list[dict[str, Any]] = []

    for n, idx in enumerate(indices):
        try:
            raw = dataset[idx]
            raw_np = _to_numpy_tree(raw)
            task = str(raw_np.get("prompt", raw_np.get("task_index", "unknown")))
            task_counts[task] = task_counts.get(task, 0) + 1

            if "action" in raw_np:
                raw_action_chunks.append(np.asarray(raw_np["action"], dtype=np.float64))
            if "tcp_pose_wrt_base" in raw_np:
                tcp_base_chunks.append(np.asarray(raw_np["tcp_pose_wrt_base"], dtype=np.float64))

            transformed = _apply_with_prompt(repack_pipeline, raw)
            if "actions" in transformed:
                traj_action_chunks.append(np.asarray(transformed["actions"], dtype=np.float64))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"index": int(idx), "error": str(exc)})

        if (n + 1) % 500 == 0:
            print(f"processed {n + 1}/{len(indices)} samples; kept traj={len(traj_action_chunks)} skipped={len(skipped)}", file=sys.stderr)

    raw_chunks = _stack_chunks(raw_action_chunks)
    tcp_chunks = _stack_chunks(tcp_base_chunks)
    traj_chunks = _stack_chunks(traj_action_chunks)

    raw_flat = _flatten_time(raw_chunks)
    tcp_flat = _flatten_time(tcp_chunks)
    traj_flat = _flatten_time(traj_chunks)

    summary: dict[str, Any] = {
        "config_name": args.config_name,
        "repo_id": str(repo_id),
        "fps": fps,
        "action_horizon": action_horizon,
        "requested_indices": len(indices),
        "kept_raw_action_chunks": len(raw_action_chunks),
        "kept_tcp_pose_chunks": len(tcp_base_chunks),
        "kept_traj_action_chunks": len(traj_action_chunks),
        "skipped_count": len(skipped),
        "skipped_first_20": skipped[:20],
        "task_counts": task_counts,
    }

    if raw_flat is not None:
        raw_names = _dim_names("raw_action", raw_flat.shape[-1])
        summary["raw_action_stats"] = _safe_stats(raw_flat, raw_names)
        _plot_hist_grid(raw_flat, raw_names, "Raw dataset action distribution", out_dir / "raw_action_hist.png", int(args.bins))
        _plot_temporal_mean_std(raw_chunks, raw_names, "Raw dataset action mean/std over horizon", out_dir / "raw_action_temporal_mean_std.png")
        _plot_corr(raw_flat, raw_names, "Raw action dimension correlation", out_dir / "raw_action_corr.png")

    if tcp_flat is not None:
        tcp_names = _dim_names("tcp_base", tcp_flat.shape[-1])
        summary["tcp_pose_wrt_base_stats"] = _safe_stats(tcp_flat, tcp_names)
        _plot_hist_grid(tcp_flat, tcp_names, "TCP pose wrt base distribution", out_dir / "tcp_pose_wrt_base_hist.png", int(args.bins))
        _plot_temporal_mean_std(tcp_chunks, tcp_names, "TCP pose wrt base mean/std over horizon", out_dir / "tcp_pose_wrt_base_temporal_mean_std.png")

    if traj_flat is not None:
        traj_names = _dim_names("traj_action", traj_flat.shape[-1])
        summary["traj_action_stats"] = _safe_stats(traj_flat, traj_names)
        _plot_hist_grid(traj_flat, traj_names, "Transformed trajectory action distribution", out_dir / "traj_action_hist.png", int(args.bins))
        _plot_temporal_mean_std(traj_chunks, traj_names, "Transformed trajectory action mean/std over horizon", out_dir / "traj_action_temporal_mean_std.png")
        _plot_corr(traj_flat, traj_names, "Transformed trajectory action correlation", out_dir / "traj_action_corr.png")
        summary["trajectory_relative_stats"] = _plot_trajectory_views(
            traj_chunks,
            out_dir,
            int(args.max_plot_trajectories),
            int(args.bins),
        )

    with open(out_dir / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote analysis to {out_dir}")
    print(f"kept traj chunks: {len(traj_action_chunks)} / requested {len(indices)}; skipped {len(skipped)}")
    if task_counts:
        top_tasks = sorted(task_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("top tasks:")
        for task, count in top_tasks:
            print(f"  {count:6d}  {task}")


if __name__ == "__main__":
    main()
