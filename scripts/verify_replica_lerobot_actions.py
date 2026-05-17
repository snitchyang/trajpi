#!/usr/bin/env python3
"""
Verify that LeRobot-format `action` columns match the original rearrange-dataset HDF5 files.

Conversion layout (as in `meta/info.json`):
- Episodes are the concatenation of `traj_*` groups in each file listed in `source_h5_files`,
  in file order; within a file, trajectories are ordered by numeric suffix (traj_0, traj_1, ...).
- Each H5 group has `actions` with shape (T, 13). The LeRobot episode has T+1 rows: the last
  frame repeats the final action (so frame_index T matches H5 action index T-1).

Usage:
  cd /path/to/openpi
  conda run -n openpi python scripts/verify_replica_lerobot_actions.py \\
    --lerobot-root=datasets/mshab/lerobot_replica_all \\
    --rearrange-dataset-root=/path/to/.../rearrange-dataset

  # Random spot-checks (default 32 episodes x 3 frames each)
  python scripts/verify_replica_lerobot_actions.py --rearrange-dataset-root=... --seed=0

  # One explicit frame
  python scripts/verify_replica_lerobot_actions.py --rearrange-dataset-root=... \\
    --episodes=100 --frames=5,0

  # If a source .h5 is missing locally, pass one integer per file in `source_h5_files` (same order as
  # in meta/info.json) so global episode indices still line up. Each value is the number of traj_*
  # groups in that file. On a machine with all H5s: open each file, count traj_*, and take
  # sum == info.json "total_episodes" as a consistency check.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError as e:  # pragma: no cover
    print("This script requires h5py. Install it or use the conda env that has the conversion deps.", file=sys.stderr)
    raise e

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as e:  # pragma: no cover
    print("This script requires pyarrow (for parquet).", file=sys.stderr)
    raise e


REARRANGE_MARKER = "rearrange-dataset"


def _resolve_h5_path(stored_path: str, rearrange_root: Path | None) -> Path:
    """If the path from info.json is missing, try the same relative path under --rearrange-dataset-root."""
    p = Path(stored_path)
    if p.is_file():
        return p
    if rearrange_root is None:
        raise FileNotFoundError(
            f"H5 not found: {stored_path!s}\n"
            f"Pass --rearrange-dataset-root=.../rearrange-dataset (parent of prepare_groceries/, set_table/, ...)."
        )
    if REARRANGE_MARKER not in stored_path:
        raise FileNotFoundError(
            f"Cannot map {stored_path!s} under rearrange-dataset: marker {REARRANGE_MARKER!r} not in path."
        )
    rel = stored_path.split(REARRANGE_MARKER, 1)[1].lstrip("/")
    candidate = rearrange_root / rel
    if not candidate.is_file():
        raise FileNotFoundError(f"Resolved H5 not found: {candidate}")
    return candidate


def _traj_keys_sorted(grp) -> list[str]:
    keys = [k for k in grp.keys() if k.startswith("traj_")]

    def sort_key(name: str) -> int:
        return int(name.split("_", 1)[1])

    return sorted(keys, key=sort_key)


def _build_h5_index(
    stored_h5_paths: list[str],
    rearrange_root: Path | None,
    traj_counts_override: list[int] | None = None,
    total_episodes_hint: int | None = None,
) -> tuple[list[Path | None], list[int], int]:
    """
    Returns (resolved_paths, traj_counts_per_file, total_trajs).
    Entry `resolved_paths[i]` is None if the file is missing on disk. Counts for missing files come
    from `traj_counts_override` or, if exactly one file is missing, from
    `total_episodes_hint - sum(trajs in present files)`.
    """
    resolved: list[Path | None] = []
    counts: list[int] = []
    n_files = len(stored_h5_paths)
    for stored in stored_h5_paths:
        p: Path | None
        try:
            p = _resolve_h5_path(stored, rearrange_root)
        except FileNotFoundError:
            p = None
        if p is not None:
            with h5py.File(p, "r") as f:
                n = len(_traj_keys_sorted(f))
            if n == 0:
                raise ValueError(f"No traj_* groups in {p}")
        else:
            n = -1
        resolved.append(p)
        counts.append(n)

    missing_ix = [i for i, p in enumerate(resolved) if p is None]
    for i in missing_ix:
        if traj_counts_override is not None and i < len(traj_counts_override):
            counts[i] = int(traj_counts_override[i])
        elif len(missing_ix) == 1 and total_episodes_hint is not None:
            on_disk = sum(c for j, c in enumerate(counts) if resolved[j] is not None)
            inferred = int(total_episodes_hint) - on_disk
            if inferred < 0:
                raise ValueError(
                    f"Inferred trajectory count {inferred} for missing file; check total_episodes={total_episodes_hint}."
                )
            counts[i] = inferred
            print(
                f"Note: missing {stored_h5_paths[i]!s} — using {inferred} trajs (total_episodes {total_episodes_hint} "
                f"minus {on_disk} on-disk).",
                file=sys.stderr,
            )
        else:
            raise FileNotFoundError(
                f"Missing {stored_h5_paths[i]!s}. Use --rearrange-dataset-root, pass --traj-counts with "
                f"{n_files} integers, or keep only one missing file so count can be inferred from info.json total_episodes."
            )

    if traj_counts_override is not None and len(traj_counts_override) != n_files:
        print(
            f"Warning: --traj-counts has length {len(traj_counts_override)} but source_h5_files has {n_files}.",
            file=sys.stderr,
        )
    total = sum(counts)
    return resolved, counts, total


def _episode_to_traj(
    episode_index: int, resolved_paths: list[Path | None], traj_counts: list[int]
) -> tuple[Path | None, str, int, int]:
    """
    Returns (h5_path_or_none, traj_name, local_traj_index, file_index).
    h5_path is None if the source file is missing on disk (counts from --traj-counts only).
    """
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    e = episode_index
    for fi, (path, n) in enumerate(zip(resolved_paths, traj_counts, strict=True)):
        if e < n:
            return path, f"traj_{e}", e, fi
        e -= n
    raise IndexError(
        f"episode_index {episode_index} is out of range; "
        f"dataset has {sum(traj_counts)} trajectories (from H5 index)."
    )


def _h5_row_for_frame(frame_index: int, n_actions: int) -> int:
    """Map LeRobot frame_index to index into H5 `actions` (last frame duplicates last action)."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if n_actions == 0:
        raise ValueError("H5 has zero actions")
    return min(int(frame_index), n_actions - 1)


def _lerobot_parquet_path(lerobot_root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    # episode_index = episode_index % chunks_size
    return lerobot_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _load_episode_tasks(lerobot_root: Path) -> dict[int, list[str]]:
    """Map episode_index -> tasks (from meta/episodes.jsonl)."""
    out: dict[int, list[str]] = {}
    path = lerobot_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return out
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            epi = o.get("episode_index")
            if epi is None or "tasks" not in o:
                continue
            t = o["tasks"]
            if isinstance(t, list):
                out[int(epi)] = [str(x) for x in t]
    return out


def _read_lerobot_action(lerobot_root: Path, episode_index: int, frame_index: int, chunks_size: int) -> np.ndarray:
    p = _lerobot_parquet_path(lerobot_root, episode_index, chunks_size)
    if not p.is_file():
        raise FileNotFoundError(f"LeRobot parquet not found: {p}")
    t = pq.read_table(p, columns=["action", "frame_index"])
    n = t.num_rows
    if n == 0:
        raise ValueError(f"Empty parquet: {p}")
    if frame_index >= n:
        raise IndexError(
            f"frame_index {frame_index} >= episode length {n} (episode {episode_index}, file {p.name})"
        )
    if int(t.column("frame_index")[frame_index].as_py()) != frame_index:
        # Defensive: rows should be sorted by frame_index
        for i in range(n):
            if int(t.column("frame_index")[i].as_py()) == frame_index:
                frame_index = i
                break
        else:
            raise RuntimeError("frame_index not found in column")
    row = t.column("action")[frame_index].as_py()
    return np.asarray(row, dtype=np.float32)


def _read_h5_action(
    h5_path: Path, traj_name: str, h5_row: int, action_shape: int
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        actions = f[traj_name]["actions"][:]
    if actions.shape[-1] != action_shape:
        raise ValueError(f"Expected action dim {action_shape}, got {actions.shape} in {h5_path} {traj_name}")
    if h5_row >= len(actions):
        raise IndexError(f"H5 action row {h5_row} out of range (len {len(actions)}) in {h5_path} {traj_name}")
    return np.asarray(actions[h5_row], dtype=np.float32)


def _parse_int_pairs(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":", 1)
        out.append((int(a), int(b)))
    return out


def _fail_context(
    ep: int,
    file_index: int,
    stored_h5: list[str],
    h5_path: Path,
    tasks_by_ep: dict[int, list[str]],
) -> str:
    if 0 <= file_index < len(stored_h5):
        stored = stored_h5[file_index]
    else:
        stored = f"(invalid source_h5_files index {file_index})"
    try:
        resolved = str(h5_path.resolve())
    except OSError:
        resolved = str(h5_path)
    if ep in tasks_by_ep:
        tasks_s = json.dumps(tasks_by_ep[ep], ensure_ascii=False)
    else:
        tasks_s = "(not found in meta/episodes.jsonl)"
    return (
        f"  leRobot tasks: {tasks_s}\n"
        f"  h5 path (info.json source_h5_files): {stored}\n"
        f"  h5 path (resolved on disk): {resolved}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("datasets/mshab/lerobot_replica_all"),
        help="LeRobot dataset root (contains meta/ and data/).",
    )
    p.add_argument(
        "--rearrange-dataset-root",
        type=Path,
        default=None,
        help="Directory that ends with .../rearrange-dataset (used if absolute paths in info.json are wrong).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random episode/frame selection.")
    p.add_argument(
        "--num-episodes",
        type=int,
        default=32,
        help="Number of random episodes to check (ignored if --episodes is set).",
    )
    p.add_argument(
        "--frames-per-episode",
        type=int,
        default=3,
        help="How many random frames to check per random episode.",
    )
    p.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Explicit episodes, comma-separated, e.g. 0,100,43200 (disables random episode set).",
    )
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Explicit (episode,frame) pairs: ep0:fr0,ep1:fr1,...; if set, --episodes is ignored for sampling.",
    )
    p.add_argument(
        "--atol", type=float, default=0.0, help="Absolute tolerance for np.allclose (default: exact match)."
    )
    p.add_argument(
        "--rtol", type=float, default=0.0, help="Relative tolerance for np.allclose (default: exact match)."
    )
    p.add_argument(
        "--traj-counts",
        type=Path,
        default=None,
        help="JSON file: list of ints, one per source_h5_files, used when an H5 path is missing on disk.",
    )
    args = p.parse_args()

    meta = args.lerobot_root / "meta" / "info.json"
    if not meta.is_file():
        print(f"Missing {meta}", file=sys.stderr)
        return 1
    info = json.loads(meta.read_text())
    stored_h5: list[str] = info["source_h5_files"]
    chunks_size = int(info.get("chunks_size", 1000))
    exp_action_dim = 13
    if "features" in info and "action" in info["features"]:
        sh = info["features"]["action"].get("shape", [exp_action_dim])
        if sh:
            exp_action_dim = int(sh[0])

    traj_override: list[int] | None = None
    auto_counts = args.lerobot_root / "meta" / "h5_traj_counts.json"
    if args.traj_counts is not None:
        traj_override = json.loads(args.traj_counts.read_text())
        if not isinstance(traj_override, list) or not traj_override:
            print("--traj-counts must be a non-empty JSON array of integers.", file=sys.stderr)
            return 1
        traj_override = [int(x) for x in traj_override]
    elif auto_counts.is_file():
        raw = json.loads(auto_counts.read_text())
        if isinstance(raw, list):
            traj_override = [int(x) for x in raw]
        elif isinstance(raw, dict) and "counts" in raw:
            traj_override = [int(x) for x in raw["counts"]]
    print("Building H5 index (one short open per file)...", flush=True)
    resolved, traj_counts, total_h5_trajs = _build_h5_index(
        stored_h5,
        args.rearrange_dataset_root,
        traj_counts_override=traj_override,
        total_episodes_hint=int(info["total_episodes"]),
    )
    total_episodes_meta = int(info["total_episodes"])
    if total_h5_trajs != total_episodes_meta:
        print(
            f"Warning: H5 trajectory count {total_h5_trajs} != info.json total_episodes {total_episodes_meta}.",
            file=sys.stderr,
        )

    tasks_by_ep = _load_episode_tasks(args.lerobot_root)

    if args.frames:
        checks = _parse_int_pairs(args.frames)
    elif args.episodes:
        eps = [int(x) for x in args.episodes.split(",") if x.strip()]
        rng = random.Random(args.seed)
        checks2: list[tuple[int, int]] = []
        for ep in eps:
            plen = args.lerobot_root / "meta" / "episodes.jsonl"
            length: int | None = None
            if plen.is_file():
                with plen.open() as f:
                    for line in f:
                        o = json.loads(line)
                        if o.get("episode_index") == ep:
                            length = int(o["length"])
                            break
            if length is None:
                # read parquet row count
                pt = _lerobot_parquet_path(args.lerobot_root, ep, chunks_size)
                t = pq.read_table(pt, columns=["action"])
                length = t.num_rows
            for _ in range(args.frames_per_episode):
                checks2.append((ep, rng.randrange(0, length)))
        checks = checks2
    else:
        rng = random.Random(args.seed)
        max_ep = min(total_episodes_meta, total_h5_trajs) - 1
        if max_ep < 0:
            print("No episodes to sample.", file=sys.stderr)
            return 1
        plen = args.lerobot_root / "meta" / "episodes.jsonl"
        length_by_ep: dict[int, int] = {}
        if plen.is_file():
            with plen.open() as f:
                for line in f:
                    o = json.loads(line)
                    epi = o.get("episode_index")
                    if epi is not None:
                        length_by_ep[int(epi)] = int(o["length"])

        def episode_length(ep: int) -> int:
            if ep in length_by_ep:
                return length_by_ep[ep]
            pt = _lerobot_parquet_path(args.lerobot_root, ep, chunks_size)
            if not pt.is_file():
                raise FileNotFoundError(str(pt))
            t = pq.read_table(pt, columns=["action"])
            return t.num_rows

        def sample_ep_with_parquet() -> int:
            for _ in range(10_000):
                cand = rng.randint(0, max_ep)
                if _lerobot_parquet_path(args.lerobot_root, cand, chunks_size).is_file():
                    return cand
            raise RuntimeError(
                f"No LeRobot parquets found under {args.lerobot_root!s} after 10000 random tries; check --lerobot-root."
            )

        checks = []
        for _ in range(args.num_episodes):
            ep = sample_ep_with_parquet()
            try:
                length = episode_length(ep)
            except FileNotFoundError:
                continue
            for _ in range(args.frames_per_episode):
                fr = rng.randrange(0, length)
                checks.append((ep, fr))
        if not checks:
            print(
                f"No random checks could be scheduled (no LeRobot parquets under {args.lerobot_root / 'data'!s}?).",
                file=sys.stderr,
            )
            return 1

    failed = 0
    skipped = 0
    for ep, fr in checks:
        pq_path = _lerobot_parquet_path(args.lerobot_root, ep, chunks_size)
        if not pq_path.is_file():
            print(
                f"skip ep={ep} fr={fr} (LeRobot parquet not on disk: {pq_path.name}).",
                file=sys.stderr,
            )
            skipped += 1
            continue
        h5_path, traj_name, _, h5_file_index = _episode_to_traj(ep, resolved, traj_counts)
        if h5_path is None:
            print(
                f"skip ep={ep} fr={fr} (source H5 for this block is not on disk; set --traj-counts to keep global indices).",
                file=sys.stderr,
            )
            skipped += 1
            continue
        with h5py.File(h5_path, "r") as f:
            n_act = len(f[traj_name]["actions"])
        h5_i = _h5_row_for_frame(fr, n_act)
        a_h5 = _read_h5_action(h5_path, traj_name, h5_i, exp_action_dim)
        a_lr = _read_lerobot_action(args.lerobot_root, ep, fr, chunks_size)
        if a_h5.shape != a_lr.shape:
            ctx = _fail_context(ep, h5_file_index, stored_h5, h5_path, tasks_by_ep)
            print(
                f"FAIL ep={ep} fr={fr}: shape {a_h5.shape} vs {a_lr.shape}\n{ctx}\n"
                f"  traj {traj_name} h5_row={h5_i} (n_actions={n_act})",
                file=sys.stderr,
            )
            failed += 1
            continue
        if not np.allclose(a_h5, a_lr, rtol=args.rtol, atol=args.atol):
            d = float(np.max(np.abs(a_h5 - a_lr)))
            ctx = _fail_context(ep, h5_file_index, stored_h5, h5_path, tasks_by_ep)
            print(
                f"FAIL ep={ep} fr={fr} max_abs_diff={d}\n{ctx}\n"
                f"  h5 {traj_name} row={h5_i} (n_actions={n_act})\n"
                f"  h5 {a_h5}\n  lr {a_lr}",
                file=sys.stderr,
            )
            failed += 1
        else:
            print(f"ok ep={ep} fr={fr}  {h5_path.name} {traj_name} h5_row={h5_i}/{n_act}")

    ran = len(checks) - skipped
    if ran == 0:
        print(
            f"No checks ran (scheduled {len(checks)} samples, all skipped). "
            f"Use --episodes with parquets that exist under {args.lerobot_root / 'data'!s}, "
            f"or copy the full LeRobot split.",
            file=sys.stderr,
        )
        return 1
    if failed:
        print(
            f"Done: {failed}/{ran} run checks failed (total scheduled {len(checks)}, skipped {skipped}).",
            file=sys.stderr,
        )
        return 1
    print(f"All {ran} run checks passed (scheduled {len(checks)}, skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
