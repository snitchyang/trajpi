#!/usr/bin/env python3
"""Visualize ``PolicyRecorder`` dumps (``policy_records/step_*.npy``).

Each file is a 0-d ``object`` array whose ``.item()`` is a flat dict from
``flax.traverse_util.flatten_dict(..., sep='/')`` with ``inputs/...`` and
``outputs/...`` keys. This script shows input images and predicted actions.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


def load_flat_record(path: pathlib.Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if raw.shape != () or raw.dtype != object:
        raise ValueError(
            f"{path}: expected 0-d object array (PolicyRecorder format), got shape={raw.shape}, dtype={raw.dtype}"
        )
    obj = raw.item()
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected dict inside object array, got {type(obj)}")
    out: dict[str, np.ndarray] = {}
    for k, v in obj.items():
        out[str(k)] = np.asarray(v)
    return out


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """(H,W,C) or (C,H,W); float [0,1] or uint8 -> uint8 (H,W,C)."""
    x = np.asarray(img)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 5:
        x = x[0, -1]
    if x.ndim != 3:
        raise ValueError(f"expected 3-d image after squeeze, got shape {x.shape}")
    # CHW
    if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if np.issubdtype(x.dtype, np.floating):
        if x.max() <= 1.0 + 1e-3:
            x = (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x


def _is_probably_image(key: str, arr: np.ndarray) -> bool:
    if not key.startswith("inputs/"):
        return False
    kl = key.lower()
    if "image_mask" in kl:
        return False
    if "image" not in kl:
        return False
    if arr.dtype == object:
        return False
    if arr.ndim < 3:
        return False
    return True


def _collect_images(flat: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    items: list[tuple[str, np.ndarray]] = []
    for k in sorted(flat.keys()):
        v = flat[k]
        if not _is_probably_image(k, v):
            continue
        try:
            hwc = _to_hwc_uint8(v)
        except ValueError:
            continue
        short = k.removeprefix("inputs/")
        items.append((short, hwc))
    return items


def _find_actions(flat: dict[str, np.ndarray]) -> np.ndarray | None:
    if "outputs/actions" in flat:
        return np.asarray(flat["outputs/actions"], dtype=np.float64)
    for k, v in flat.items():
        if k.startswith("outputs/") and k.endswith("/actions"):
            return np.asarray(v, dtype=np.float64)
    return None


def _natural_step_paths(record_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(record_dir.glob("step_*.npy"), key=lambda p: _step_index(p.name))
    return paths


def _step_index(name: str) -> int:
    m = re.search(r"step_(\d+)", name)
    return int(m.group(1)) if m else -1


def visualize_step(
    flat: dict[str, np.ndarray],
    *,
    title: str,
    out_path: pathlib.Path | "./server_visualization",
    show: bool,
) -> None:
    images = _collect_images(flat)
    actions = _find_actions(flat)

    n_img = max(1, len(images))
    fig = plt.figure(figsize=(4 * n_img, 7), layout="constrained")
    gs = fig.add_gridspec(2, n_img, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.2)

    if not images:
        ax0 = fig.add_subplot(gs[0, :])
        ax0.text(0.5, 0.5, "No input image keys found\n(under inputs/... with 'image' in path)", ha="center", va="center")
        ax0.axis("off")
    else:
        for i, (name, im) in enumerate(images):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(im)
            ax.set_title(name[:48] + ("..." if len(name) > 48 else ""), fontsize=9)
            ax.axis("off")

    ax_act = fig.add_subplot(gs[1, :])
    if actions is None:
        ax_act.text(0.5, 0.5, "No outputs/actions found", ha="center", va="center", transform=ax_act.transAxes)
        ax_act.axis("off")
    else:
        a = np.asarray(actions)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        im = ax_act.imshow(a.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
        ax_act.set_xlabel("time index (action horizon)")
        ax_act.set_ylabel("action dim")
        ax_act.set_title(f"actions  shape={tuple(a.shape)}")
        plt.colorbar(im, ax=ax_act, fraction=0.02, pad=0.02)

    fig.suptitle(title)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=pathlib.Path,
        default=pathlib.Path("policy_records"),
        help="Directory containing step_*.npy (default: policy_records)",
    )
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help='Optional step range as "start:end" (end exclusive), e.g. "0:50"',
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="If set, save PNGs here instead of only showing windows",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Never call plt.show() (for headless; use with --out-dir)",
    )
    p.add_argument(
        "--also-show",
        action="store_true",
        help="If --out-dir is set, still open a window for each step (default: only save when --out-dir is set)",
    )
    args = p.parse_args()

    record_dir = args.dir.expanduser().resolve()
    if not record_dir.is_dir():
        raise SystemExit(f"Not a directory: {record_dir}")

    paths = _natural_step_paths(record_dir)
    if not paths:
        raise SystemExit(f"No step_*.npy files under {record_dir}")

    if args.steps:
        parts = args.steps.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        paths = [x for x in paths if start <= _step_index(x.name) < end] if end is not None else [x for x in paths if _step_index(x.name) >= start]

    out_root = args.out_dir.expanduser().resolve() if args.out_dir is not None else None
    want_show = not args.no_show and (out_root is None or args.also_show)

    for path in paths:
        flat = load_flat_record(path)
        out_png = out_root / f"{path.stem}.png" if out_root is not None else None
        visualize_step(
            flat,
            title=f"{path.name}  ({record_dir.name})",
            out_path=out_png,
            show=want_show,
        )


if __name__ == "__main__":
    main()