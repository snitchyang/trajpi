#!/usr/bin/env python3
"""
Visualize a Fetch TCP trajectory computed only from LeRobot ``observation.state`` qpos and ``action``.

This is the qpos/action-only counterpart of ``visualize_mshab_relative_traj.py``.  It does not read
``tcp_pose_wrt_base``.  For each dataset frame it:

  1. Reads the current 12-D Fetch qpos from ``observation.state``.
  2. Reads an action chunk from ``action`` using LeRobot ``delta_timestamps``.
  3. Rolls qpos forward with the mshab normalized action layout:
       action[0:7]   arm joint deltas, scaled from [-1, 1] to [-0.1, 0.1] rad
       action[7]     gripper command, scaled from [-1, 1] to [-0.01, 0.05] m
       action[8:10]  reserved head dims, ignored by the dataset/action wrapper
       action[10]    torso_lift delta, scaled from [-1, 1] to [-0.1, 0.1] m
       action[11:13] base forward/angular velocity commands
  4. Runs a small URDF forward-kinematics pass from ``base_link`` to ``gripper_link``.
  5. Projects the resulting TCP path, expressed in the base-at-current-frame coordinate system,
     onto the current fetch_head image.

The 12-D qpos order is the ManiSkill Fetch order after dropping the 3 planar base joints:

  [torso_lift, head_pan, head_tilt,
   shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll,
   l_gripper_finger, r_gripper_finger]

Example:

  python scripts/visualize_mshab_qpos_action_traj.py --config-name pi05_mstraj \\
    --camera-json mshab_fetch_head_base.json --index 42 --out qpos_action_traj.png

  python scripts/visualize_mshab_qpos_action_traj.py --config-name pi05_mstraj \\
    --camera-json mshab_fetch_head_base.json --video-out qpos_action_traj.mp4 \\
    --start-index 0 --num-frames 200
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve()
_OPI_ROOT = _SCRIPT.parent.parent
_OPI_SRC = _OPI_ROOT / "src"
if _OPI_SRC.is_dir() and str(_OPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPI_SRC))
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

from openpi import transforms as _transforms  # noqa: E402
from openpi.policies import libero_policy  # noqa: E402
import openpi.training.config as openpi_train_config  # noqa: E402

import visualize_mshab_relative_traj as _vis  # noqa: E402


MSHAB_QPOS_ACTION_REPACK = {
    "observation/image": "observation.images.top",
    "observation/wrist_image": "observation.images.wrist",
    "observation/state": "observation.state",
    "actions": "action",
    "prompt": "prompt",
}

DEFAULT_URDF = _OPI_ROOT / "mshab" / "ManiSkill" / "mani_skill" / "assets" / "robots" / "fetch" / "fetch.urdf"

QPOS_NAMES = (
    "torso_lift_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "l_gripper_finger_joint",
    "r_gripper_finger_joint",
)

ARM_QPOS_SLICE = slice(3, 10)
LEFT_GRIPPER_QPOS_IDX = 10
RIGHT_GRIPPER_QPOS_IDX = 11
HEAD_PAN_QPOS_IDX = 1
HEAD_TILT_QPOS_IDX = 2
TORSO_QPOS_IDX = 0
FK_QPOS_SIGNS = np.ones(len(QPOS_NAMES), dtype=np.float64)
FK_QPOS_SIGNS[5] = -1.0
"""Convert LeRobot/SAPIEN qpos convention to the URDF axis convention used by this lightweight FK.

The upperarm roll sign is the only position-affecting mismatch. This was checked against the dataset
``tcp_pose_wrt_base`` column, but the script still computes trajectories only from qpos + action.
"""

ARM_DELTA_SCALE = 0.1
BODY_DELTA_SCALE = 0.1
GRIPPER_ACTION_LOW = -0.01
GRIPPER_ACTION_HIGH = 0.05


def _as_np_tree(data):
    return _vis._data_to_numpy(data)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def _make_T(R: np.ndarray | None = None, p: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None

    @property
    def origin_T(self) -> np.ndarray:
        return _make_T(_rpy_matrix(self.origin_rpy), self.origin_xyz)

    def motion_T(self, q: float) -> np.ndarray:
        if self.joint_type in {"revolute", "continuous"}:
            return _make_T(_axis_angle_matrix(self.axis, q))
        if self.joint_type == "prismatic":
            return _make_T(p=self.axis * float(q))
        return np.eye(4, dtype=np.float64)


class FetchUrdfKinematics:
    def __init__(self, urdf_path: Path, base_link: str = "base_link", tcp_link: str = "gripper_link"):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Fetch URDF not found: {self.urdf_path}")
        self.base_link = base_link
        self.tcp_link = tcp_link
        self.joints = self._load_joints(self.urdf_path)
        self.children_by_parent: dict[str, list[UrdfJoint]] = {}
        self.joints_by_name: dict[str, UrdfJoint] = {j.name: j for j in self.joints}
        for joint in self.joints:
            self.children_by_parent.setdefault(joint.parent, []).append(joint)
        self.path = self._find_path(base_link, tcp_link)
        if self.path is None:
            raise ValueError(f"Could not find URDF path from {base_link!r} to {tcp_link!r}")

    @staticmethod
    def _parse_vec(elem: ET.Element | None, attr: str, default: str) -> np.ndarray:
        text = default if elem is None else elem.attrib.get(attr, default)
        return np.fromstring(text, sep=" ", dtype=np.float64)

    @classmethod
    def _load_joints(cls, urdf_path: Path) -> list[UrdfJoint]:
        root = ET.parse(urdf_path).getroot()
        joints: list[UrdfJoint] = []
        for je in root.findall("joint"):
            origin = je.find("origin")
            axis = je.find("axis")
            limit = je.find("limit")
            lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None
            upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None
            joints.append(
                UrdfJoint(
                    name=je.attrib["name"],
                    joint_type=je.attrib.get("type", "fixed"),
                    parent=je.find("parent").attrib["link"],  # type: ignore[union-attr]
                    child=je.find("child").attrib["link"],  # type: ignore[union-attr]
                    origin_xyz=cls._parse_vec(origin, "xyz", "0 0 0"),
                    origin_rpy=cls._parse_vec(origin, "rpy", "0 0 0"),
                    axis=cls._parse_vec(axis, "xyz", "0 0 0"),
                    lower=lower,
                    upper=upper,
                )
            )
        return joints

    def _find_path(self, start: str, goal: str) -> list[UrdfJoint] | None:
        def dfs(link: str, path: list[UrdfJoint]) -> list[UrdfJoint] | None:
            if link == goal:
                return path
            for joint in self.children_by_parent.get(link, []):
                found = dfs(joint.child, [*path, joint])
                if found is not None:
                    return found
            return None

        return dfs(start, [])

    @staticmethod
    def qpos_to_joint_values(qpos12: np.ndarray) -> dict[str, float]:
        q = np.asarray(qpos12, dtype=np.float64).reshape(-1)
        if q.size < 12:
            raise ValueError(f"Expected 12-D Fetch qpos, got shape {qpos12.shape}")
        q = q.copy()
        q[: len(FK_QPOS_SIGNS)] *= FK_QPOS_SIGNS
        return {name: float(q[i]) for i, name in enumerate(QPOS_NAMES)}

    def clip_qpos(self, qpos12: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos12, dtype=np.float64).copy().reshape(-1)
        for i, name in enumerate(QPOS_NAMES):
            joint = self.joints_by_name.get(name)
            if joint is None:
                continue
            if joint.lower is not None:
                q[i] = max(q[i], joint.lower)
            if joint.upper is not None:
                q[i] = min(q[i], joint.upper)
        return q

    def tcp_pose_base(self, qpos12: np.ndarray) -> np.ndarray:
        values = self.qpos_to_joint_values(qpos12)
        T = np.eye(4, dtype=np.float64)
        assert self.path is not None
        for joint in self.path:
            q = values.get(joint.name, 0.0)
            T = T @ joint.origin_T @ joint.motion_T(q)
        return T


def _base_pose_T(xyth: np.ndarray) -> np.ndarray:
    x, y, th = [float(v) for v in xyth[:3]]
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _step_base_pose(xyth: np.ndarray, action_row: np.ndarray, dt: float, v_scale: float, w_scale: float) -> np.ndarray:
    x, y, th = [float(v) for v in xyth[:3]]
    v = float(action_row[11]) * float(v_scale)
    w = float(action_row[12]) * float(w_scale)
    return np.array([x + v * np.cos(th) * dt, y + v * np.sin(th) * dt, th + w * dt], dtype=np.float64)


def _scale_normalized(value: np.ndarray | float, low: float, high: float) -> np.ndarray | float:
    v = np.clip(value, -1.0, 1.0)
    return low + (v + 1.0) * 0.5 * (high - low)


def _apply_action_to_qpos(qpos12: np.ndarray, action_row: np.ndarray) -> np.ndarray:
    q = np.asarray(qpos12, dtype=np.float64).copy()
    a = np.asarray(action_row, dtype=np.float64).reshape(-1)
    if a.size < 13:
        raise ValueError(f"Expected 13-D mshab action, got shape {action_row.shape}")
    q[ARM_QPOS_SLICE] += np.clip(a[0:7], -1.0, 1.0) * ARM_DELTA_SCALE
    # action[8:10] are head placeholders and are zeroed by FetchActionWrapper(stationary_head=True).
    q[TORSO_QPOS_IDX] += float(np.clip(a[10], -1.0, 1.0)) * BODY_DELTA_SCALE
    gripper_qpos = float(_scale_normalized(a[7], GRIPPER_ACTION_LOW, GRIPPER_ACTION_HIGH))
    q[LEFT_GRIPPER_QPOS_IDX] = gripper_qpos
    q[RIGHT_GRIPPER_QPOS_IDX] = gripper_qpos
    return q


def _tcp_traj_from_qpos_actions(
    fk: FetchUrdfKinematics,
    qpos12: np.ndarray,
    actions: np.ndarray,
    *,
    fps: float,
    base_v_scale: float,
    base_w_scale: float,
    include_current: bool,
    clip_limits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: roll qpos by controller target deltas, not by simulated physics."""
    q = np.asarray(qpos12, dtype=np.float64).reshape(-1).copy()
    act = np.asarray(actions, dtype=np.float64)
    if act.ndim == 1:
        act = act.reshape(1, -1)
    if q.size < 12:
        raise ValueError(f"Expected 12-D qpos, got shape {qpos12.shape}")
    if act.shape[-1] < 13:
        raise ValueError(f"Expected action last dim >= 13, got shape {act.shape}")

    dt = 1.0 / float(fps)
    base = np.zeros(3, dtype=np.float64)
    points: list[np.ndarray] = []
    qseq: list[np.ndarray] = []

    def append_point() -> None:
        q_eval = fk.clip_qpos(q) if clip_limits else q
        T = _base_pose_T(base) @ fk.tcp_pose_base(q_eval)
        points.append(T[:3, 3].copy())
        qseq.append(q_eval.copy())

    if include_current:
        append_point()

    for row in act:
        q = _apply_action_to_qpos(q, row)
        if clip_limits:
            q = fk.clip_qpos(q)
        base = _step_base_pose(base, row, dt, base_v_scale, base_w_scale)
        append_point()

    return np.stack(points, axis=0), np.stack(qseq, axis=0)


def _tcp_traj_from_qpos_sequence_actions(
    fk: FetchUrdfKinematics,
    qpos_seq12: np.ndarray,
    actions: np.ndarray,
    *,
    fps: float,
    base_v_scale: float,
    base_w_scale: float,
    clip_limits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Use recorded future qpos for the arm/body and action only for base velocity integration."""
    qseq = np.asarray(qpos_seq12, dtype=np.float64)
    act = np.asarray(actions, dtype=np.float64)
    if qseq.ndim == 1:
        qseq = qseq.reshape(1, -1)
    if act.ndim == 1:
        act = act.reshape(1, -1)
    if qseq.shape[-1] < 12:
        raise ValueError(f"Expected qpos sequence last dim >= 12, got shape {qseq.shape}")
    if act.shape[-1] < 13:
        raise ValueError(f"Expected action last dim >= 13, got shape {act.shape}")

    n = min(qseq.shape[0], act.shape[0])
    dt = 1.0 / float(fps)
    base = np.zeros(3, dtype=np.float64)
    points: list[np.ndarray] = []
    qout: list[np.ndarray] = []
    for i in range(n):
        q_eval = fk.clip_qpos(qseq[i]) if clip_limits else qseq[i]
        T = _base_pose_T(base) @ fk.tcp_pose_base(q_eval)
        points.append(T[:3, 3].copy())
        qout.append(q_eval.copy())
        base = _step_base_pose(base, act[i], dt, base_v_scale, base_w_scale)
    return np.stack(points, axis=0), np.stack(qout, axis=0)


def _apply_repack(pipeline: _transforms.DataTransformFn, raw: dict) -> dict:
    d = _as_np_tree(raw)
    if "prompt" not in d:
        d["prompt"] = ""
    return _as_np_tree(pipeline(d))


def _resolve_dataset(args: argparse.Namespace) -> tuple[Path, int, _transforms.DataTransformFn, str | None]:
    if args.config_name:
        tc = openpi_train_config.get_config(args.config_name)
        data_cfg = tc.data.create(tc.assets_dirs, tc.model)
        repo_s = args.repo_id if args.repo_id is not None else data_cfg.repo_id
        if not repo_s:
            raise SystemExit(f"Config {args.config_name!r} has no repo_id; pass --repo-id.")
        horizon = args.action_horizon if args.action_horizon is not None else int(tc.model.action_horizon)
        return Path(repo_s).expanduser().resolve(), horizon, _transforms.RepackTransform(MSHAB_QPOS_ACTION_REPACK), args.config_name

    if args.repo_id is None:
        raise SystemExit("Pass either --config-name or --repo-id.")
    horizon = int(args.action_horizon) if args.action_horizon is not None else 10
    return Path(args.repo_id).expanduser().resolve(), horizon, _transforms.RepackTransform(MSHAB_QPOS_ACTION_REPACK), None


def _project_points_with_depth(
    xyz: np.ndarray,
    wh: tuple[int, int],
    r_w2c: np.ndarray,
    t_w2c: np.ndarray,
    k_mat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    uv = _vis._project_world_points(xyz, wh, r_w2c, t_w2c, k_mat=k_mat)
    cam_xyz = (np.asarray(r_w2c, dtype=np.float64) @ np.asarray(xyz, dtype=np.float64).T) + np.asarray(
        t_w2c, dtype=np.float64
    ).reshape(3, 1)
    return uv, cam_xyz[2]


def _visible_uv_mask(
    uv: np.ndarray,
    depth: np.ndarray,
    image_shape: tuple[int, int],
    *,
    margin_px: int = 1,
    min_depth: float = 1e-4,
) -> np.ndarray:
    h, w = image_shape
    u = uv[:, 0]
    v = uv[:, 1]
    return (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(depth)
        & (depth > min_depth)
        & (u >= -margin_px)
        & (u <= (w - 1 + margin_px))
        & (v >= -margin_px)
        & (v <= (h - 1 + margin_px))
    )


def _draw_polyline(
    bgr: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray | None = None,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
) -> None:
    if uv.size == 0:
        return
    h, w = bgr.shape[:2]
    if depth is None:
        visible = np.isfinite(uv).all(axis=1)
    else:
        visible = _visible_uv_mask(uv, depth, (h, w))
    max_segment_px = 0.45 * float(max(h, w))
    for i in range(1, uv.shape[0]):
        if not (visible[i - 1] and visible[i]):
            continue
        if np.linalg.norm(uv[i] - uv[i - 1]) > max_segment_px:
            continue
        p0 = (int(round(uv[i - 1, 0])), int(round(uv[i - 1, 1])))
        p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        cv2.line(bgr, p0, p1, color_bgr, max(1, w // 64), lineType=cv2.LINE_AA)
    for i, (u, v) in enumerate(uv):
        if not visible[i]:
            continue
        color = (255, 0, 0) if i == 0 else (0, 165, 255)
        cv2.circle(bgr, (int(round(u)), int(round(v))), max(1, w // 128), color, -1, lineType=cv2.LINE_AA)


def _sample_to_traj(
    *,
    fk: FetchUrdfKinematics,
    data: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(data["observation/state"], dtype=np.float64)
    actions = np.asarray(data["actions"], dtype=np.float64)
    if qpos.ndim >= 2:
        return _tcp_traj_from_qpos_sequence_actions(
            fk,
            qpos,
            actions,
            fps=float(args.traj_fps),
            base_v_scale=float(args.base_v_scale),
            base_w_scale=float(args.base_w_scale),
            clip_limits=bool(args.clip_limits),
        )
    return _tcp_traj_from_qpos_actions(
        fk,
        qpos.reshape(-1),
        actions,
        fps=float(args.traj_fps),
        base_v_scale=float(args.base_v_scale),
        base_w_scale=float(args.base_w_scale),
        include_current=bool(args.include_current),
        clip_limits=bool(args.clip_limits),
    )


def _camera_tvec_for_sample(data: dict, r_b2c: np.ndarray, t_b2c: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not args.torso_compensation:
        return t_b2c
    qpos = np.asarray(data.get("observation/state"), dtype=np.float64)
    if qpos.ndim >= 2:
        qpos = qpos[0]
    qpos = qpos.reshape(-1)
    if qpos.size <= TORSO_QPOS_IDX:
        return t_b2c
    return _vis._torso_compensated_tvec(r_b2c, t_b2c, float(qpos[TORSO_QPOS_IDX]), float(args.torso_ref_qpos))


def _write_static(
    *,
    repo_name: str,
    data: dict,
    xyz: np.ndarray,
    qseq: np.ndarray,
    r_b2c: np.ndarray,
    t_b2c: np.ndarray,
    k_mat: np.ndarray | None,
    frame_index: int,
    out_path: Path,
    show: bool,
    args: argparse.Namespace,
) -> None:
    base_rgb = libero_policy._parse_image(data["observation/image"])
    h, w = base_rgb.shape[:2]
    t_cam = _camera_tvec_for_sample(data, r_b2c, t_b2c, args)
    uv, depth = _project_points_with_depth(xyz, (h, w), r_b2c, t_cam, k_mat=k_mat)
    bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    _draw_polyline(bgr, uv, depth)
    vis_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rel = xyz - xyz[0:1]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), layout="tight")
    axes[0].imshow(vis_rgb)
    axes[0].axis("off")
    axes[0].set_title("fetch_head + qpos/action TCP trajectory")
    axes[0].text(
        0.01,
        0.99,
        f"index={frame_index}  points={len(xyz)}  fps={args.traj_fps}\n"
        f"state=qpos12, action=arm/body/base",
        transform=axes[0].transAxes,
        fontsize=7,
        va="top",
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, pad=2),
    )

    axes[1].plot(rel[:, 0], rel[:, 1], "-o", markersize=3)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Delta x (m)")
    axes[1].set_ylabel("Delta y (m)")
    axes[1].set_title("TCP path in base-at-current frame")

    t = np.arange(qseq.shape[0])
    axes[2].plot(t, qseq[:, TORSO_QPOS_IDX], label="torso")
    axes[2].plot(t, qseq[:, ARM_QPOS_SLICE], alpha=0.65)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("trajectory point")
    axes[2].set_ylabel("qpos")
    axes[2].set_title("Rolled qpos (torso + arm)")
    axes[2].legend(loc="best", fontsize=7)

    fig.suptitle(f"{repo_name} | computed from qpos + action only", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _encode_video(
    *,
    ds: lerobot_dataset.LeRobotDataset,
    pipeline: _transforms.DataTransformFn,
    fk: FetchUrdfKinematics,
    r_b2c: np.ndarray,
    t_b2c: np.ndarray,
    k_mat: np.ndarray | None,
    start: int,
    num: int,
    out_path: Path,
    video_fps: float,
    args: argparse.Namespace,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    n_written = 0
    out_h = out_w = 0
    for step_i, idx in enumerate(range(start, start + num)):
        if idx < 0 or idx >= len(ds):
            break
        try:
            data = _apply_repack(pipeline, ds[idx])
            xyz, _ = _sample_to_traj(fk=fk, data=data, args=args)
        except Exception as e:  # noqa: BLE001 - keep long videos robust to a bad row.
            print(f"Warning: skip index {idx}: {e}", file=sys.stderr)
            continue

        base_rgb = libero_policy._parse_image(data["observation/image"])
        h, w = base_rgb.shape[:2]
        t_cam = _camera_tvec_for_sample(data, r_b2c, t_b2c, args)
        uv, depth = _project_points_with_depth(xyz, (h, w), r_b2c, t_cam, k_mat=k_mat)
        bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        _draw_polyline(bgr, uv, depth)
        overlay = f"idx={idx} points={len(xyz)} qpos+action FK"
        cv2.putText(bgr, overlay, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        if writer is None:
            out_h, out_w = h, w
            writer = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), (out_w, out_h))
        if (h, w) != (out_h, out_w):
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(bgr)
        n_written += 1
        if (step_i + 1) % 50 == 0:
            print(f"  encoded {step_i + 1}/{num} frames...", file=sys.stderr)
    if writer is not None:
        writer.release()
    return n_written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", type=str, default=None, help="openpi training config name, e.g. pi05_mstraj")
    p.add_argument("--repo-id", type=str, default=None, help="Override / provide LeRobot dataset root")
    p.add_argument("--action-horizon", type=int, default=None, help="Action chunk horizon; default from config model")
    p.add_argument("--index", type=int, default=0, help="Global frame index for PNG mode")
    p.add_argument("--out", type=str, default="mshab_qpos_action_traj.png", help="Output PNG path")
    p.add_argument("--show", action="store_true", help="Show matplotlib window in PNG mode")
    p.add_argument("--video-out", type=str, default=None, help="If set, write an MP4 instead of only a PNG")
    p.add_argument("--start-index", type=int, default=0, help="First global frame index for video mode")
    p.add_argument("--num-frames", type=int, default=None, help="Number of video frames; default min(500, remaining)")
    p.add_argument("--video-fps", type=float, default=None, help="Output video FPS; default dataset FPS")
    p.add_argument("--traj-fps", type=float, default=20.0, help="Control/action FPS used for base velocity integration")
    p.add_argument("--base-v-scale", type=float, default=1.0, help="Scale applied to action[11] before base integration")
    p.add_argument(
        "--base-w-scale",
        type=float,
        default=float(np.pi),
        help="Scale applied to action[12] before base integration. Default matches MshabTcpBaseToWorldActions.",
    )
    p.add_argument("--urdf", type=str, default=str(DEFAULT_URDF), help="Fetch URDF path")
    p.add_argument("--camera-json", type=str, default=None, help="Camera JSON from scripts/dump_mshab_fetch_camera.py")
    p.add_argument(
        "--cam-eye",
        type=float,
        nargs=3,
        default=(-0.12, 0.0, 1.3),
        help="Fallback virtual camera eye in base frame when --camera-json is absent",
    )
    p.add_argument(
        "--cam-target",
        type=float,
        nargs=3,
        default=(0.6, 0.0, 0.7),
        help="Fallback virtual camera target in base frame when --camera-json is absent",
    )
    p.add_argument("--cam-up", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="Fallback camera up vector")
    p.add_argument("--torso-ref-qpos", type=float, default=_vis.FETCH_REST_TORSO, help="Reference torso qpos for camera JSON")
    p.add_argument("--no-torso-compensation", dest="torso_compensation", action="store_false", default=True)
    p.add_argument("--no-include-current", dest="include_current", action="store_false", default=True)
    p.add_argument("--no-clip-limits", dest="clip_limits", action="store_false", default=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo, horizon, pipeline, cfg_name = _resolve_dataset(args)
    meta = lerobot_dataset.LeRobotDatasetMetadata(str(repo))
    meta_fps = float(meta.fps) if meta.fps else float(args.traj_fps)
    delta_t = 1.0 / meta_fps
    delta_times = [i * delta_t for i in range(horizon)]
    ds = lerobot_dataset.LeRobotDataset(
        str(repo),
        delta_timestamps={
            "action": delta_times,
            "observation.state": delta_times,
        },
    )
    fk = FetchUrdfKinematics(Path(args.urdf).expanduser())
    r_b2c, t_b2c, k_mat = _vis._load_camera_from_args(args)

    if cfg_name:
        print(f"Using config {cfg_name!r}: repo={repo} horizon={horizon}", file=sys.stderr)
    print(f"Using qpos/action FK only; URDF={fk.urdf_path}", file=sys.stderr)

    if args.video_out:
        start = int(args.start_index)
        if start >= len(ds):
            raise SystemExit(f"start-index {start} out of range for dataset len {len(ds)}")
        num = min(int(args.num_frames) if args.num_frames is not None else 500, len(ds) - start)
        vfps = float(args.video_fps) if args.video_fps is not None else meta_fps
        out_v = Path(args.video_out).expanduser()
        n = _encode_video(
            ds=ds,
            pipeline=pipeline,
            fk=fk,
            r_b2c=r_b2c,
            t_b2c=t_b2c,
            k_mat=k_mat,
            start=start,
            num=num,
            out_path=out_v,
            video_fps=vfps,
            args=args,
        )
        print(f"Wrote {n} frames to {out_v}")
        return

    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"index {args.index} out of range for dataset len {len(ds)}")
    data = _apply_repack(pipeline, ds[args.index])
    xyz, qseq = _sample_to_traj(fk=fk, data=data, args=args)
    out_path = Path(args.out).expanduser()
    _write_static(
        repo_name=repo.name,
        data=data,
        xyz=xyz,
        qseq=qseq,
        r_b2c=r_b2c,
        t_b2c=t_b2c,
        k_mat=k_mat,
        frame_index=int(args.index),
        out_path=out_path,
        show=bool(args.show),
        args=args,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
