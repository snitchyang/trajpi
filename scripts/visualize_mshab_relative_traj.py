#!/usr/bin/env python3
"""
Project the **future action-chunk trajectory** onto the **current top camera frame**, mirroring the
training data pipeline (``Repack`` + :class:`MshabTcpBaseToWorldActions`) exactly.

Geometry (critical!)
  :class:`MshabTcpBaseToWorldActions` turns ``tcp_pose_wrt_base[t…t+H-1]`` into chunk-TCP positions
  expressed in the **base-at-chunk-start** frame (``openpi/transforms.py``:
  ``_integrate_base_velocities_mshab`` seeds ``poses[0] = [0, 0, 0]``). So each chunk's 3-D points
  live in the robot base frame **at the current step t**, *not* in sim world.

  Therefore the camera extrinsic for projection must be in the **same** base-at-t frame, i.e.
  ``T_cam_base`` (OpenCV). Use :mod:`scripts.dump_mshab_fetch_camera` to export it; the resulting
  ``for_visualize_mshab`` block has ``"R", "t"`` that pre-multiply a base-frame point to get the
  OpenCV camera-frame point (``P_cam = R @ P_base + t``).

Cumulative-drift fix (per-frame torso compensation)
  The head camera rides on ``head_camera_link`` which sits above the prismatic ``torso_lift_joint``.
  The camera JSON is dumped at the Fetch rest keyframe (``qpos[torso_lift] = 0.386``), but during an
  episode the torso drops to ~0.17 m — a ~20 cm shift along base Z — which is the dominant source
  of the drift between the projected polyline and the real gripper.

  We compensate **per frame** using a single real-world-observable signal: the torso encoder at
  time ``t`` (LeRobot ``observation.state[0]`` — Fetch's 12-D state is ``qpos[3:15]``, so element 0
  is ``torso_lift_joint``). Because ``torso_lift_joint`` is prismatic along ``+Z_base`` and (with
  ``stationary_head=True``) only translates ``head_camera_link`` by ``Δq`` along base Z::

      t_cam_base(q) = t_cam_base(q_ref) - R_cam_base @ [0, 0, q_torso(t) - q_torso_ref]

  ``R_cam_base`` stays the same. ``q_torso_ref`` defaults to the Fetch rest keyframe (0.386), which
  matches ``dump_mshab_fetch_camera.py --steps 0``. Override with ``--torso-ref-qpos`` or disable
  via ``--no-torso-compensation``. Base planar motion within the chunk is still handled by
  :class:`MshabTcpBaseToWorldActions` (same integration as training).

Usage (from openpi root, with ``src`` on PYTHONPATH as for training)::

  # 1) Dump real fetch_head K + base-frame extrinsic from a reset of the same env family
  python scripts/dump_mshab_fetch_camera.py --out mshab_fetch_head_base.json

  # 2) Encode a verification video (each frame: future chunk projected into current top image)
  python scripts/visualize_mshab_relative_traj.py --config-name pi05_mstraj \\
    --camera-json mshab_fetch_head_base.json --video-out verify.mp4 \\
    --start-index 0 --num-frames 200

  # Single frame PNG
  python scripts/visualize_mshab_relative_traj.py --config-name pi05_mstraj \\
    --camera-json mshab_fetch_head_base.json --index 42 --out mshab_traj_vis.png

  # Fallback (no camera JSON): virtual look-at *in base frame* (+X forward, +Z up for fetch)
  python scripts/visualize_mshab_relative_traj.py --config-name pi05_mstraj \\
    --cam-eye -0.1 0.0 1.3 --cam-target 0.6 0.0 0.7 --video-out verify.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Package imports (``src`` must be on path, same as ``scripts/train_pytorch.py``)
_SCRIPT = Path(__file__).resolve()
_OPI_ROOT = _SCRIPT.parent.parent
_OPI_SRC = _OPI_ROOT / "src"
if _OPI_SRC.is_dir() and str(_OPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPI_SRC))

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

import openpi.training.config as openpi_train_config  # noqa: E402
from openpi.policies import libero_policy  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402

# Same repack as ``LeRobotMshabDataConfig``
MSHAB_REPACK: dict = {
    "observation/image": "observation.images.top",
    "observation/wrist_image": "observation.images.wrist",
    "observation/state": "observation.state",
    "actions": "action",
    "tcp_pose_wrt_base": "tcp_pose_wrt_base",
}

try:
    import torch  # type: ignore

    _TORCH = torch
except ImportError:  # pragma: no cover
    _TORCH = None


def _data_to_numpy(data: dict) -> dict:
    o: dict = {}
    for k, v in data.items():
        if isinstance(v, dict):
            o[k] = _data_to_numpy(v)
        elif _TORCH is not None and isinstance(v, _TORCH.Tensor):
            o[k] = v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):
            o[k] = v.numpy()
        elif hasattr(v, "detach"):
            o[k] = v.detach().cpu().numpy()
        else:
            o[k] = np.asarray(v)
    return o


def _look_at_to_world2cam(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build OpenCV **frame→camera** (R, t) with ``P_c = R @ P + t``, X right, Y down, Z forward.

    The source frame is whatever ``eye`` / ``target`` are expressed in. For projecting
    :class:`MshabTcpBaseToWorldActions` output, provide eye/target in the **current base** frame
    (chunk-start base at origin, +X forward, +Z up for fetch)."""
    eye = np.asarray(eye, dtype=np.float64).ravel()[:3]
    target = np.asarray(target, dtype=np.float64).ravel()[:3]
    u = np.asarray(world_up, dtype=np.float64).ravel()[:3]
    z_axis = target - eye
    zn = np.linalg.norm(z_axis)
    if zn < 1e-8:
        raise ValueError("cam eye and target are too close")
    z_axis = z_axis / zn
    x_axis = np.cross(u, z_axis)
    xn = np.linalg.norm(x_axis)
    if xn < 1e-6:
        u = u + np.array([1e-3, 0.0, 0.0], dtype=np.float64)
        x_axis = np.cross(u, z_axis)
        xn = np.linalg.norm(x_axis)
    x_axis = x_axis / xn
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    r_w2c = np.stack([x_axis, y_axis, z_axis], axis=0)
    t_w2c = -r_w2c @ eye.reshape(3, 1)
    return r_w2c, t_w2c


TORSO_STATE_IDX = 0
"""Index of ``torso_lift_joint`` inside LeRobot ``observation.state`` for Fetch.

The LeRobot converter stores ``qpos[3:15]`` (drops the 3 planar base joints), so element 0 of the
12-D state is ``torso_lift_joint``. Verified by comparing all-episode first frames (~0.385) against
Fetch's ``rest`` keyframe qpos ``[0,0,0, 0.386, ...]``.
"""

FETCH_REST_TORSO = 0.386
"""Fetch ``rest`` keyframe value of ``torso_lift_joint`` — matches ``env.reset()`` in
``scripts/dump_mshab_fetch_camera.py`` when ``--steps 0``."""


def _torso_compensated_tvec(
    r_base_to_cam: np.ndarray,
    t_base_to_cam: np.ndarray,
    q_torso: float,
    q_torso_ref: float,
) -> np.ndarray:
    """Shift the OpenCV translation ``t`` to reflect the current torso height.

    Given ``P_cam = R @ P_base + t`` with ``R, t`` calibrated at ``q_torso_ref`` and the torso is
    prismatic along base Z, the camera position in base frame shifts by ``[0, 0, q - q_ref]``.
    Equivalently ``t_new = t - R @ [0, 0, q - q_ref]``. ``R`` is unchanged.
    """
    delta = np.array([0.0, 0.0, float(q_torso) - float(q_torso_ref)], dtype=np.float64).reshape(3, 1)
    return np.asarray(t_base_to_cam, dtype=np.float64).reshape(3, 1) - r_base_to_cam @ delta


def _project_world_points(
    world_xyz: np.ndarray,
    wh: tuple[int, int],
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
    k_mat: np.ndarray | None = None,
) -> np.ndarray:
    h, w = wh
    if k_mat is not None:
        K = np.asarray(k_mat, dtype=np.float64)
    else:
        K = np.array(
            [[float(w), 0.0, 0.5 * (w - 1)], [0.0, float(h), 0.5 * (h - 1)], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    dist = np.zeros(4)
    pts = world_xyz.reshape(-1, 1, 3).astype(np.float64)
    rvec, _ = cv2.Rodrigues(r_w2c)
    tvec = t_w2c.reshape(3, 1)
    img_pts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return img_pts.reshape(-1, 2)


def _rt_w2c_from_cam2world_gl(T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P_c = R P_w + t; given T_wc: camera to world, 3x4 or 4x4."""
    T = np.asarray(T_wc, dtype=np.float64)
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :4] = T
        T = T4
    T_cw = np.linalg.inv(T)
    return T_cw[:3, :3], T_cw[:3, 3:4]


def _load_camera_json_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load OpenCV ``(R, t)`` + optional ``K`` from a camera JSON.

    ``R, t`` must satisfy ``P_cam = R @ P + t`` where ``P`` is a 3-D point in the **robot base
    frame** (the same frame as :class:`MshabTcpBaseToWorldActions` output). The preferred source is
    ``scripts/dump_mshab_fetch_camera.py`` which writes ``for_visualize_mshab = {K, R, t, frame:"base"}``.

    For backward compat we also accept: top-level ``K/R/t``; ``sensors[*].name=="fetch_head"``
    with ``R, t``; ``cam2world_gl`` 4x4 (inverted to get R,t — note this is sim-world, will NOT align
    with MshabTcpBaseToWorldActions unless the episode starts with the robot base at sim-world origin);
    or Sapien ``extrinsic_cv`` 3×4/4×4 (also sim-world, same caveat).
    """
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    cam: dict = dict(root)
    if "for_visualize_mshab" in root and isinstance(root["for_visualize_mshab"], dict):
        cam.update(root["for_visualize_mshab"])
    if "R" not in cam and "sensors" in root:
        for s in root.get("sensors", []):
            if isinstance(s, dict) and s.get("name") == "fetch_head":
                cam.update(s)
                break
    k_from_file: np.ndarray | None = None
    if "K" in cam and cam["K"] is not None:
        k_from_file = np.asarray(cam["K"], dtype=np.float64)
    r_mat: np.ndarray | None = None
    t_vec: np.ndarray | None = None
    frame_label: str | None = cam.get("frame") if isinstance(cam.get("frame"), str) else None
    if "R" in cam and "t" in cam:
        r_mat = np.asarray(cam["R"], dtype=np.float64)
        t_vec = np.asarray(cam["t"], dtype=np.float64).reshape(3, 1)
    elif cam.get("cam2world_gl") is not None:
        r_mat, t_vec = _rt_w2c_from_cam2world_gl(np.asarray(cam["cam2world_gl"], dtype=np.float64))
        frame_label = frame_label or "world"
    elif cam.get("extrinsic_cv") is not None:
        ex = np.asarray(cam["extrinsic_cv"], dtype=np.float64)
        r_mat = ex[:3, :3]
        t_vec = ex[:3, 3:4]
        frame_label = frame_label or "world"
    if r_mat is None or t_vec is None:
        raise ValueError(
            f"{path}: need one of: (R,t), or cam2world_gl, or extrinsic_cv, or for_visualize_mshab / sensors[fetch_head]"
        )
    if frame_label is not None and frame_label != "base":
        print(
            f"Warning: camera JSON {path!r} declares frame={frame_label!r}; MshabTcpBaseToWorldActions emits "
            f"points in the *current base* frame so the projection will misalign. Re-dump with the updated "
            f"scripts/dump_mshab_fetch_camera.py to get base-frame extrinsics.",
            file=sys.stderr,
        )
    return r_mat, t_vec, k_from_file


def _load_camera_from_args(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if args.camera_json:
        return _load_camera_json_file(str(args.camera_json))
    r_w2c, t_w2c = _look_at_to_world2cam(
        np.array(args.cam_eye, dtype=np.float64),
        np.array(args.cam_target, dtype=np.float64),
        np.array(args.cam_up, dtype=np.float64),
    )
    return r_w2c, t_w2c, None


def _apply_repack_pipeline(pipeline: _transforms.DataTransformFn, raw: dict) -> dict:
    """Apply the training `repack_transforms` chain. Visualization does not use language: if the sample has no
    ``prompt`` but the config repack maps ``"prompt": "prompt"``, we insert an empty string so repack does not fail.
    """
    d = _data_to_numpy(raw)
    if "prompt" not in d:
        d["prompt"] = ""  # visualization only; repack may still list "prompt" from training config
    return _data_to_numpy(pipeline(d))


def _world_positions_from_data(data: dict) -> np.ndarray:
    if "actions" not in data:
        raise KeyError("Repack+transform did not produce 'actions'")
    act = np.asarray(data["actions"], dtype=np.float64)
    if act.ndim == 1:
        act = act.reshape(1, -1)
    if act.shape[-1] < 3:
        raise ValueError(f"Expected actions with at least 3 dims (world xyz), got shape {act.shape}")
    return act[:, :3]


def _draw_traj_on_bgr(bgr: np.ndarray, uv: np.ndarray) -> None:
    """Draw polylines and keypoints; mutates bgr in place (BGR, uint8)."""
    h, w = bgr.shape[:2]
    n = uv.shape[0]
    for t in range(1, n):
        p0 = (int(round(uv[t - 1, 0])), int(round(uv[t - 1, 1])))
        p1 = (int(round(uv[t, 0])), int(round(uv[t, 1])))
        cv2.line(bgr, p0, p1, (0, 255, 0), max(1, w // 64), lineType=cv2.LINE_AA)
    for t in range(n):
        cx, cy = int(round(uv[t, 0])), int(round(uv[t, 1]))
        r = max(1, w // 128)
        cv2.circle(bgr, (cx, cy), r, (0, 165, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(
        bgr, (int(round(uv[0, 0])), int(round(uv[0, 1]))), max(2, w // 64), (255, 0, 0), 1, lineType=cv2.LINE_AA
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Train config name (``openpi.training.config.get_config``). Sets repo from DataConfig, "
        "``action_horizon`` from ``model.action_horizon``, repack/trajectory transforms from the data factory, "
        "and LeRobot ``delta_timestamps`` keys from ``action_sequence_keys``. "
        "Provide ``--repo-id`` only to override the dataset path.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Local LeRobot v2.1 root (data/, meta/…). Required if ``--config-name`` is not set; optional override if it is set.",
    )
    p.add_argument("--index", type=int, default=0, help="Global frame index (static PNG mode; same as __getitem__ index)")
    p.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Override action chunk size; default: model's ``action_horizon`` (with --config-name) or 10 (manual mode).",
    )
    p.add_argument(
        "--traj-fps",
        type=float,
        default=20.0,
        help="Legacy manual mode only: MshabTcpBaseToWorldActions fps. With ``--config-name``, fps is taken from "
        "``LeRobotMshabDataConfig.traj_action_fps`` inside the repack (ignored if that transform is absent).",
    )
    p.add_argument(
        "--cam-eye",
        type=float,
        nargs=3,
        default=(-0.12, 0.0, 1.3),
        metavar=("X", "Y", "Z"),
        help="Virtual camera position in **base frame** (meters; +X forward, +Y left, +Z up for fetch). Ignored if --camera-json is set.",
    )
    p.add_argument(
        "--cam-target",
        type=float,
        nargs=3,
        default=(0.6, 0.0, 0.7),
        metavar=("X", "Y", "Z"),
        help="Look-at point in **base frame** (meters). Ignored if --camera-json is set.",
    )
    p.add_argument(
        "--cam-up", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("UX", "UY", "UZ"), help="Up vector (same frame as eye/target)"
    )
    p.add_argument(
        "--camera-json",
        type=str,
        default=None,
        help="JSON from ``scripts/dump_mshab_fetch_camera.py`` with ``for_visualize_mshab = {K, R, t, frame:\"base\"}``; "
        "overrides the look-at fallback. Older dumps (sim-world R/t or cam2world_gl) will project to the wrong place "
        "and emit a warning — re-dump with the updated script.",
    )
    p.add_argument(
        "--video-out",
        type=str,
        default=None,
        help="If set, write an MP4: each frame = top image at t with the action chunk (t..t+H-1) trajectory overlaid",
    )
    p.add_argument("--start-index", type=int, default=0, help="First global frame index for --video-out")
    p.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of video frames to encode; default: min(500, len(ds)-start) for video mode",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output video FPS; default: dataset meta fps",
    )
    p.add_argument(
        "--out",
        type=str,
        default="mshab_relative_traj.png",
        help="Output path for the two-panel PNG (only when --video-out is not set, unless --summary-png is used alongside video)",
    )
    p.add_argument(
        "--summary-png",
        type=str,
        default=None,
        help="With --video-out, optionally write the same two-panel matplotlib figure to this path (uses --index)",
    )
    p.add_argument("--show", action="store_true", help="Show matplotlib window (static mode or --summary-png only)")
    p.add_argument(
        "--torso-ref-qpos",
        type=float,
        default=FETCH_REST_TORSO,
        help=f"Reference value of torso_lift_joint (meters) at which the camera JSON was captured. "
        f"Default {FETCH_REST_TORSO} (Fetch rest keyframe, matches ``dump_mshab_fetch_camera.py --steps 0``).",
    )
    p.add_argument(
        "--no-torso-compensation",
        dest="torso_compensation",
        action="store_false",
        default=True,
        help="Disable per-frame torso-lift compensation (reproduces the old drifting behavior).",
    )
    return p.parse_args()


def _current_torso_qpos(data: dict) -> float | None:
    """Pull ``q_torso`` (``torso_lift_joint``) from a repacked sample.

    ``data`` is the dict returned by :func:`_apply_repack_pipeline`; ``observation/state`` is the
    12-D Fetch proprioception (``qpos[3:15]``), so ``[TORSO_STATE_IDX]`` is the torso encoder.
    Returns ``None`` if the key is missing (e.g., custom repack that dropped state).
    """
    state = data.get("observation/state")
    if state is None:
        return None
    arr = np.asarray(state).reshape(-1)
    if arr.size <= TORSO_STATE_IDX:
        return None
    return float(arr[TORSO_STATE_IDX])


def _write_static_summary(
    *,
    repo_name: str,
    out_path: Path,
    data: dict,
    world_pos: np.ndarray,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
    k_from_file: np.ndarray | None,
    frame_index: int,
    traj_fps: float,
    show: bool,
    torso_ref: float,
    torso_compensation: bool,
) -> None:
    rel_pos = world_pos - world_pos[0:1, :]
    n = int(world_pos.shape[0])
    base = libero_policy._parse_image(data["observation/image"])
    h, w = int(base.shape[0]), int(base.shape[1])
    bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    t_b2c = t_w2c
    torso_note = "off"
    q_torso = _current_torso_qpos(data) if torso_compensation else None
    if torso_compensation and q_torso is not None:
        t_b2c = _torso_compensated_tvec(r_w2c, t_w2c, q_torso, torso_ref)
        torso_note = f"q={q_torso:.3f}  ref={torso_ref:.3f}  Δ={q_torso - torso_ref:+.3f}m"
    elif torso_compensation:
        torso_note = "unavailable (no observation/state)"
    uv = _project_world_points(world_pos, (h, w), r_w2c, t_b2c, k_mat=k_from_file)
    _draw_traj_on_bgr(bgr, uv)
    vis_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="tight")
    ax0, ax1 = axes
    ax0.imshow(vis_rgb)
    ax0.set_title("Top camera + future chunk TCP projected (base frame, t…t+H-1)")
    ax0.axis("off")
    ax0.text(
        0.01,
        0.99,
        f"index={frame_index}  H={n}  fps={traj_fps}\n"
        f"projection: base→camera (OpenCV)\n"
        f"torso comp: {torso_note}",
        transform=ax0.transAxes,
        fontsize=7,
        verticalalignment="top",
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, pad=2),
    )
    ax1.plot(rel_pos[:, 0], rel_pos[:, 1], "-o", color="C0", markersize=3, label="Δx, Δy (m)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Δx (m, rel. to chunk t=0)")
    ax1.set_ylabel("Δy (m, rel. to chunk t=0)")
    ax1.set_title("Chunk TCP (base frame, relative to t=0)")
    ax1.legend(loc="best", fontsize=8)
    z_txt = f"Δz range: [{rel_pos[:, 2].min():.4f}, {rel_pos[:, 2].max():.4f}] m"
    fig.suptitle(f"{repo_name}  |  {z_txt}", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _resolve_dataset_and_pipeline(
    args: argparse.Namespace,
) -> tuple[Path, int, tuple[str, ...], _transforms.DataTransformFn, float, str | None]:
    """Returns (repo, action_horizon, action_keys, repack pipeline, traj_fps label, config_name)."""
    if args.config_name:
        tc = openpi_train_config.get_config(args.config_name)
        data_cfg = tc.data.create(tc.assets_dirs, tc.model)
        repo_s = args.repo_id if args.repo_id is not None else data_cfg.repo_id
        if not repo_s:
            raise SystemExit(f"Config {args.config_name!r} has no repo_id; pass --repo-id.")
        action_horizon = args.action_horizon if args.action_horizon is not None else int(tc.model.action_horizon)
        pipeline: _transforms.DataTransformFn = _transforms.compose(data_cfg.repack_transforms.inputs)
        keys = tuple(str(k) for k in data_cfg.action_sequence_keys)
        traj_label: float
        if isinstance(tc.data, openpi_train_config.LeRobotMshabDataConfig):
            traj_label = float(tc.data.traj_action_fps)
        else:
            traj_label = float(args.traj_fps)
        return (
            Path(repo_s).expanduser().resolve(),
            action_horizon,
            keys,
            pipeline,
            traj_label,
            str(args.config_name),
        )
    if args.repo_id is None:
        raise SystemExit("Pass either --config-name (recommended) or --repo-id for the LeRobot dataset root.")
    action_horizon = int(args.action_horizon) if args.action_horizon is not None else 10
    repack = _transforms.RepackTransform(MSHAB_REPACK)
    tcp2world = _transforms.MshabTcpBaseToWorldActions(fps=float(args.traj_fps))
    pipeline = _transforms.compose([repack, tcp2world])
    return (
        Path(args.repo_id).expanduser().resolve(),
        action_horizon,
        ("action", "tcp_pose_wrt_base"),
        pipeline,
        float(args.traj_fps),
        None,
    )


def _encode_traj_video(
    *,
    ds: lerobot_dataset.LeRobotDataset,
    start: int,
    num: int,
    pipeline: _transforms.DataTransformFn,
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
    k_from_file: np.ndarray | None,
    out_path: Path,
    video_fps: float,
    meta_fps: float,
    torso_ref: float,
    torso_compensation: bool,
) -> int:
    """Returns number of frames written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None
    n_written = 0
    h = w = 0
    missing_state_warned = False
    for step_i, idx in enumerate(range(start, start + num)):
        if idx < 0 or idx >= len(ds):
            break
        try:
            data = _apply_repack_pipeline(pipeline, ds[idx])
            world_pos = _world_positions_from_data(data)
        except (KeyError, ValueError) as e:
            print(f"Warning: skip index {idx}: {e}", file=sys.stderr)
            continue
        base = libero_policy._parse_image(data["observation/image"])
        hi, wi = int(base.shape[0]), int(base.shape[1])
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        t_b2c = t_w2c
        q_torso = _current_torso_qpos(data) if torso_compensation else None
        if torso_compensation and q_torso is not None:
            t_b2c = _torso_compensated_tvec(r_w2c, t_w2c, q_torso, torso_ref)
        elif torso_compensation and not missing_state_warned:
            print(
                "Warning: observation/state missing from repacked sample; torso compensation disabled.",
                file=sys.stderr,
            )
            missing_state_warned = True
        uv = _project_world_points(world_pos, (hi, wi), r_w2c, t_b2c, k_mat=k_from_file)
        _draw_traj_on_bgr(bgr, uv)
        overlay = f"idx={idx}  H={world_pos.shape[0]}  chunk fps={meta_fps:.1f}"
        if torso_compensation and q_torso is not None:
            overlay += f"  torso={q_torso:.3f} (ref {torso_ref:.3f})"
        cv2.putText(
            bgr,
            overlay,
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
        if writer is None:
            h, w = hi, wi
            writer = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), (w, h))
        if (hi, wi) != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        assert writer is not None
        writer.write(bgr)
        n_written += 1
        if (step_i + 1) % 50 == 0:
            print(f"  encoded {step_i + 1}/{num} frames…", file=sys.stderr)
    if writer is not None:
        writer.release()
    if n_written == 0:
        print("Warning: no frames were encoded; check dataset indices and action/tcp alignment.", file=sys.stderr)
    return n_written


def main() -> None:
    args = _parse_args()
    repo, action_horizon, action_keys, pipeline, traj_label_fps, cfg_name = _resolve_dataset_and_pipeline(args)
    if not (repo / "meta").is_dir():
        print(f"Warning: {repo} may not be a LeRobot root (no meta/ dir).", file=sys.stderr)

    meta = lerobot_dataset.LeRobotDatasetMetadata(str(repo))
    meta_fps = float(meta.fps) if meta.fps else 20.0
    delta_t = 1.0 / meta_fps
    delta_keys = {k: [i * delta_t for i in range(action_horizon)] for k in action_keys}

    ds = lerobot_dataset.LeRobotDataset(
        str(repo),
        delta_timestamps=delta_keys,
    )
    r_w2c, t_w2c, k_from_file = _load_camera_from_args(args)
    if cfg_name:
        print(
            f"Using train config {cfg_name!r}: repo={repo}  H={action_horizon}  "
            f"delta_keys={list(action_keys)}  traj_fps(label)={traj_label_fps}",
            file=sys.stderr,
        )

    if args.video_out is not None:
        start = int(args.start_index)
        nmax = len(ds) - start
        if nmax <= 0:
            raise SystemExit(f"start_index {start} is past dataset len {len(ds)}")
        n_request = args.num_frames
        if n_request is None:
            n_request = min(500, nmax)
        else:
            n_request = min(int(n_request), nmax)
        v_fps = float(args.video_fps) if args.video_fps is not None else meta_fps
        out_v = Path(args.video_out).expanduser()
        print(f"Encoding {n_request} frames {start}…{start + n_request - 1} → {out_v} @ {v_fps} FPS", file=sys.stderr)
        n_w = _encode_traj_video(
            ds=ds,
            start=start,
            num=n_request,
            pipeline=pipeline,
            r_w2c=r_w2c,
            t_w2c=t_w2c,
            k_from_file=k_from_file,
            out_path=out_v,
            video_fps=v_fps,
            meta_fps=meta_fps,
            torso_ref=float(args.torso_ref_qpos),
            torso_compensation=bool(args.torso_compensation),
        )
        print(f"Wrote {n_w} frames to {out_v}", file=sys.stderr)
        if args.summary_png:
            idx = int(args.index)
            if idx < 0 or idx >= len(ds):
                raise SystemExit(f"index {idx} out of range for --summary-png (len={len(ds)})")
            dsum = _apply_repack_pipeline(pipeline, ds[idx])
            wp = _world_positions_from_data(dsum)
            sp = Path(args.summary_png).expanduser()
            _write_static_summary(
                repo_name=repo.name,
                out_path=sp,
                data=dsum,
                world_pos=wp,
                r_w2c=r_w2c,
                t_w2c=t_w2c,
                k_from_file=k_from_file,
                frame_index=idx,
                traj_fps=traj_label_fps,
                show=bool(args.show),
                torso_ref=float(args.torso_ref_qpos),
                torso_compensation=bool(args.torso_compensation),
            )
            print(f"Wrote summary {sp}", file=sys.stderr)
        return

    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"index {args.index} out of range (dataset len={len(ds)})")

    data = _apply_repack_pipeline(pipeline, ds[args.index])
    world_pos = _world_positions_from_data(data)
    out_path = Path(args.out).expanduser()
    _write_static_summary(
        repo_name=repo.name,
        out_path=out_path,
        data=data,
        world_pos=world_pos,
        r_w2c=r_w2c,
        t_w2c=t_w2c,
        k_from_file=k_from_file,
        frame_index=int(args.index),
        traj_fps=traj_label_fps,
        show=bool(args.show),
        torso_ref=float(args.torso_ref_qpos),
        torso_compensation=bool(args.torso_compensation),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
