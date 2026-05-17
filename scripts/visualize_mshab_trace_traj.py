#!/usr/bin/env python3
"""Overlay a saved websocket Pi inference trace on a MSHAB LeRobot episode video.

This is the trace-only companion to ``scripts/infer_and_visualize_traj.py``:
it does not run the model.  Instead it loads the ``*.pt`` trace written by the
websocket inference path and projects each saved trajectory chunk onto the top
camera video using the same MSHAB camera helpers and torso compensation logic.

Default example for the trace/video pair discussed in the workspace:

  python scripts/visualize_mshab_trace_traj.py \
    --video-out mshab_trace_episode_000031.mp4 \
    --camera-json mshab_fetch_head_base.json

The trace only contains predictions at chunk starts (usually every 32 frames).
By default, only frames whose index matches ``record["step"]`` are annotated.
Use ``--record-mode hold`` to keep drawing the latest chunk until the next saved
record, which is useful for inspection but is not the exact per-frame inference
view from ``infer_and_visualize_traj.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_OPI_ROOT = _SCRIPT.parent.parent
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))

import visualize_mshab_relative_traj as _vis  # noqa: E402


DEFAULT_TRACE = (
    _OPI_ROOT
    / "mshab/mshab_exps/eval_seq_task/set_table/open/train/fridge/websocket_pi/"
    / "websocket_pi_inference_trace.pt"
)
DEFAULT_REPO = _OPI_ROOT / "datasets/mshab/mshab_lerobot_open_close"


def _init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _draw_polyline(
    bgr: np.ndarray,
    uv: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    thick_div: int = 64,
) -> None:
    if uv.size == 0:
        return
    h, w = bgr.shape[:2]
    line_thick = max(1, w // thick_div)
    dot_r = max(1, w // (thick_div * 2))
    for i in range(1, uv.shape[0]):
        p0 = (int(round(uv[i - 1, 0])), int(round(uv[i - 1, 1])))
        p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        cv2.line(bgr, p0, p1, color_bgr, line_thick, lineType=cv2.LINE_AA)
    for i in range(uv.shape[0]):
        p = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        cv2.circle(bgr, p, dot_r, color_bgr, -1, lineType=cv2.LINE_AA)
    start = (int(round(uv[0, 0])), int(round(uv[0, 1])))
    cv2.circle(bgr, start, max(2, w // 64), (255, 0, 0), 1, lineType=cv2.LINE_AA)


def _episode_video_path(repo_root: Path, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return repo_root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def _load_trace(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "records" not in obj:
        raise ValueError(f"{path} is not a websocket inference trace dict with a 'records' key")
    records = obj["records"]
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected trace['records'] to be a list, got {type(records).__name__}")
    return obj


def _record_xyz(record: dict, key: str) -> np.ndarray | None:
    if key not in record:
        return None
    arr = np.asarray(record[key], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] < 3:
        return None
    return arr[:, :3]


def _current_torso_qpos(record: dict) -> float | None:
    state = record.get("state")
    if state is None:
        return None
    arr = np.asarray(state).reshape(-1)
    if arr.size <= _vis.TORSO_STATE_IDX:
        return None
    return float(arr[_vis.TORSO_STATE_IDX])


def _record_for_frame(records: list[dict], frame_idx: int, mode: str) -> dict | None:
    if mode == "exact":
        for record in records:
            if int(record.get("step", -1)) == frame_idx:
                return record
        return None

    chosen: dict | None = None
    for record in records:
        step = int(record.get("step", -1))
        if step <= frame_idx:
            chosen = record
        else:
            break
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, default=DEFAULT_TRACE, help="Saved websocket_pi_inference_trace.pt path.")
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO, help="LeRobot dataset root containing videos/.")
    p.add_argument("--episode-index", type=int, default=31, help="Episode video index to visualize.")
    p.add_argument(
        "--video-key",
        type=str,
        default="observation.images.top",
        help="LeRobot video key under videos/chunk-XXX/. Default: top camera.",
    )
    p.add_argument("--video-path", type=Path, default=None, help="Override the resolved episode video path.")
    p.add_argument("--video-out", type=Path, required=True, help="Output MP4 path.")
    p.add_argument("--video-fps", type=float, default=None, help="Output FPS. Default: source video FPS.")
    p.add_argument(
        "--record-mode",
        choices=("exact", "hold"),
        default="exact",
        help="exact: annotate only frames at saved record.step; hold: draw latest saved chunk on later frames.",
    )
    p.add_argument(
        "--trace-frame-offset",
        type=int,
        default=0,
        help="Subtract this from record.step before matching video frames, useful if the trace/video start differs.",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many source video frames.")
    p.add_argument("--no-pred", dest="draw_pred", action="store_false", default=True, help="Do not draw pred_traj.")
    p.add_argument("--no-actual", dest="draw_actual", action="store_false", default=True, help="Do not draw actual_traj.")
    p.add_argument("--draw-history", action="store_true", help="Draw tcp_pose_history up to the current frame in cyan.")
    p.add_argument(
        "--camera-json",
        type=str,
        default=None,
        help="Camera JSON from scripts/dump_mshab_fetch_camera.py. If omitted, use fallback look-at camera.",
    )
    p.add_argument("--cam-eye", type=float, nargs=3, default=(-0.12, 0.0, 1.3), metavar=("X", "Y", "Z"))
    p.add_argument("--cam-target", type=float, nargs=3, default=(0.6, 0.0, 0.7), metavar=("X", "Y", "Z"))
    p.add_argument("--cam-up", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("UX", "UY", "UZ"))
    p.add_argument(
        "--torso-ref-qpos",
        type=float,
        default=_vis.FETCH_REST_TORSO,
        help=f"Reference torso_lift_joint value at camera dump time. Default: {_vis.FETCH_REST_TORSO}.",
    )
    p.add_argument(
        "--no-torso-compensation",
        dest="torso_compensation",
        action="store_false",
        default=True,
        help="Disable per-record torso-lift compensation.",
    )
    return p.parse_args()


def main() -> None:
    _init_logging()
    args = parse_args()

    trace_path = args.trace.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    video_path = (
        args.video_path.expanduser().resolve()
        if args.video_path is not None
        else _episode_video_path(repo_root, int(args.episode_index), str(args.video_key))
    )
    out_path = args.video_out.expanduser().resolve()

    trace = _load_trace(trace_path)
    records = sorted(trace["records"], key=lambda r: int(r.get("step", -1)))
    if args.trace_frame_offset:
        records = [{**r, "step": int(r.get("step", -1)) - int(args.trace_frame_offset)} for r in records]
    logging.info("Loaded %d trace records from %s", len(records), trace_path)
    logging.info("Using video %s", video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    out_fps = float(args.video_fps) if args.video_fps is not None else src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise SystemExit(f"Could not read source video size from {video_path}")

    r_base2cam, t_base2cam, k_from_file = _vis._load_camera_from_args(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for {out_path}")

    tcp_history = np.asarray(trace.get("tcp_pose_history", []), dtype=np.float64)
    if tcp_history.ndim == 3 and tcp_history.shape[-1] >= 3:
        tcp_history_xyz = tcp_history[:, 0, :3]
    else:
        tcp_history_xyz = None

    frame_idx = 0
    annotated = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= int(args.max_frames):
            break
        if (bgr.shape[1], bgr.shape[0]) != (width, height):
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

        record = _record_for_frame(records, frame_idx, str(args.record_mode))
        t_cam = t_base2cam
        q_torso = _current_torso_qpos(record) if (record is not None and args.torso_compensation) else None
        if args.torso_compensation and q_torso is not None:
            t_cam = _vis._torso_compensated_tvec(r_base2cam, t_base2cam, q_torso, float(args.torso_ref_qpos))

        if args.draw_history and tcp_history_xyz is not None:
            hist = tcp_history_xyz[: min(frame_idx + 1, tcp_history_xyz.shape[0])]
            if hist.shape[0] > 1:
                uv_hist = _vis._project_world_points(hist, (height, width), r_base2cam, t_cam, k_mat=k_from_file)
                _draw_polyline(bgr, uv_hist, (255, 255, 0), thick_div=128)

        if record is not None:
            if args.draw_actual:
                actual_xyz = _record_xyz(record, "actual_traj")
                if actual_xyz is not None:
                    uv_actual = _vis._project_world_points(actual_xyz, (height, width), r_base2cam, t_cam, k_mat=k_from_file)
                    _draw_polyline(bgr, uv_actual, (0, 0, 255), thick_div=96)
            if args.draw_pred:
                pred_xyz = _record_xyz(record, "pred_traj")
                if pred_xyz is not None:
                    uv_pred = _vis._project_world_points(pred_xyz, (height, width), r_base2cam, t_cam, k_mat=k_from_file)
                    _draw_polyline(bgr, uv_pred, (0, 255, 0), thick_div=64)
            annotated += 1

        overlay = f"episode={args.episode_index} frame={frame_idx}"
        if record is not None:
            overlay += f" record_step={int(record.get('step', -1))} mode={args.record_mode}"
        if q_torso is not None:
            overlay += f" torso={q_torso:.3f}/{float(args.torso_ref_qpos):.3f}"
        cv2.putText(bgr, overlay, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, "pred", (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr, "actual", (48, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        if args.draw_history:
            cv2.putText(bgr, "history", (106, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

        writer.write(bgr)
        frame_idx += 1

    cap.release()
    writer.release()
    logging.info("Wrote %d frames (%d annotated) to %s", frame_idx, annotated, out_path)


if __name__ == "__main__":
    main()
