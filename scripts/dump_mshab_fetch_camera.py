#!/usr/bin/env python3
"""
Export ManiSkill / Sapien **fetch_head** (and **fetch_hand**) camera K and extrinsics **in the robot
base frame** for use with OpenCV :func:`cv2.projectPoints` and
``scripts/visualize_mshab_relative_traj.py``.

Why base-frame?
  ``MshabTcpBaseToWorldActions`` (training / visualization) expresses chunk TCP positions in the
  **base-at-chunk-start** frame (see ``openpi/transforms.py``: ``_integrate_base_velocities_mshab``
  starts ``poses[0] = [0, 0, 0]``). Projecting these points therefore requires the camera's
  extrinsic in that same base frame, i.e., ``T_cam_base``. A sim-world extrinsic does not align with
  future-chunk TCPs.

Implementation
  After ``env.reset`` (+ optional ``--steps`` no-op steps), we read:
    - ``intrinsic_cv`` → ``K`` (3x3)
    - ``extrinsic_cv`` → ``T_cam_world`` (OpenCV world → camera; 3x4 or 4x4)
    - ``cam2world_gl``  → ``T_world_cam`` (for reference only)
    - ``base_link.pose`` → ``T_world_base`` (4x4, SAPIEN quat convention ``[w, x, y, z]``)

  Then ``T_cam_base = T_cam_world @ T_world_base``; the output ``"R"``, ``"t"`` in
  ``for_visualize_mshab`` are the 3x3, 3 parts of that (OpenCV, **base → camera**).

Limitations
  * The head camera is mounted on ``head_camera_link`` (after ``torso_lift_joint``). If the torso
    moves during the episode, ``T_cam_base`` changes — a single static dump is only an approximation.
    Use ``stationary_torso=True`` during data generation for the best alignment, or extend this script
    to do per-frame FK from ``observation.state``.
  * Pick ``--torso-qpos`` (cm from joint min) if you want the dump at a specific torso height instead
    of the default reset qpos.

Usage (from ``openpi`` root, with GPU / ManiSkill / mshab on PYTHONPATH)::

  cd /path/to/openpi
  python scripts/dump_mshab_fetch_camera.py --out mshab_fetch_head_base.json

  # Shorter, default paths match mshab ``run_environment.py`` (set_table / open / train)
  python scripts/dump_mshab_fetch_camera.py --out cam.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Paths: openpi/ scripts -> openpi, mshab package, ManiSkill
_OPI = Path(__file__).resolve().parent.parent
_MSHAB_ROOT = _OPI / "mshab"
if _MSHAB_ROOT.is_dir():
    sys.path.insert(0, str(_MSHAB_ROOT))
_MS_DIR = _MSHAB_ROOT / "ManiSkill"
if _MS_DIR.is_dir():
    sys.path.insert(0, str(_MS_DIR))


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pad_to_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :4] = T
        return T4
    raise ValueError(f"Unsupported matrix shape: {T.shape}")


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """SAPIEN / ManiSkill quaternion is (w, x, y, z)."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    w, x, y, z = q[:4]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_struct_to_T(pose) -> np.ndarray:
    """Convert a ManiSkill / SAPIEN Pose (possibly batched) to a 4x4 matrix."""
    p = _to_numpy(pose.p)
    q = _to_numpy(pose.q)
    if p.ndim == 2:
        p = p[0]
    if q.ndim == 2:
        q = q[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_R(q)
    T[:3, 3] = p[:3]
    return T


def _get_base_pose_in_world(env, base_link_name: str) -> tuple[np.ndarray, str]:
    """Try a list of likely base-link names; fall back to the articulation root pose."""
    from mani_skill.utils import sapien_utils  # late import inside function

    robot = env.unwrapped.agent.robot
    candidates = [base_link_name, "base_link", "base", "torso_fixed_link"]
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            link = sapien_utils.get_obj_by_name(robot.get_links(), name)
        except Exception:
            link = None
        if link is not None:
            return _pose_struct_to_T(link.pose), name
    return _pose_struct_to_T(robot.pose), "<articulation_root>"


def _params_to_record(name: str, p: dict, T_world_base: np.ndarray) -> dict:
    K = _to_numpy(p["intrinsic_cv"])
    if K.ndim == 3:
        K = K[0]
    ex = _to_numpy(p.get("extrinsic_cv"))
    if ex is not None and ex.size > 0 and ex.ndim == 3:
        ex = ex[0]
    c2w = _to_numpy(p.get("cam2world_gl"))
    if c2w is not None and c2w.size > 0 and c2w.ndim == 3:
        c2w = c2w[0]

    T_cam_world = _pad_to_4x4(ex) if ex is not None and ex.size > 0 else np.linalg.inv(_pad_to_4x4(c2w))
    T_world_base_4 = _pad_to_4x4(T_world_base)
    T_cam_base = T_cam_world @ T_world_base_4
    T_base_cam = np.linalg.inv(T_cam_base)
    R_cb = T_cam_base[:3, :3]
    t_cb = T_cam_base[:3, 3:4]
    R_cw = T_cam_world[:3, :3]
    t_cw = T_cam_world[:3, 3:4]
    return {
        "name": name,
        "K": K.tolist(),
        "R": R_cb.tolist(),
        "t": t_cb.reshape(3).tolist(),
        "T_cam_base": T_cam_base.tolist(),
        "T_base_cam": T_base_cam.tolist(),
        "R_cam_world": R_cw.tolist(),
        "t_cam_world": t_cw.reshape(3).tolist(),
        "extrinsic_cv_world": (None if ex is None else ex.tolist()),
        "cam2world_gl": c2w.tolist() if c2w is not None else None,
    }


def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Output JSON path")
    p.add_argument(
        "--env-id", type=str, default="OpenSubtaskTrain-v0", help="Registered mshab / ManiSkill env id"
    )
    p.add_argument("--task", type=str, default="set_table", help="rearrange task name for default plan path")
    p.add_argument("--subtask", type=str, default="open", help="rearrange subtask (open, pick, …)")
    p.add_argument("--split", type=str, default="train", help="train or val for spawn/plan files")
    p.add_argument("--sim-backend", type=str, default="gpu", help="ManiSkill sim backend (gpu|cpu)")
    p.add_argument("--steps", type=int, default=0, help="Number of no-op action steps after reset (camera pose may change)")
    p.add_argument("--seed", type=int, default=0, help="Reset seed")
    p.add_argument(
        "--base-link-name",
        type=str,
        default="base_link",
        help="Robot base link name used for base-frame extrinsic (tries this first, then common fallbacks)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    from mani_skill import ASSET_DIR

    import mshab.envs  # noqa: F401  — register envs
    from mshab.envs.planner import plan_data_from_file

    rearr = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_path = rearr / "task_plans" / args.task / args.subtask / args.split / "all.json"
    if not plan_path.is_file():
        raise SystemExit(
            f"Plan not found: {plan_path}\n"
            f"Set --task --subtask --split or edit paths; ASSET_DIR={ASSET_DIR}"
        )
    plan_data = plan_data_from_file(plan_path)
    spawn_data_fp = rearr / "spawn_data" / args.task / args.subtask / args.split / "spawn_data.pt"
    if not spawn_data_fp.is_file():
        raise SystemExit(f"spawn_data not found: {spawn_data_fp}")

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgbd",
        sim_backend=args.sim_backend,
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        shader_dir="minimal",
        max_episode_steps=200,
        reward_mode="normalized_dense",
        task_plans=plan_data.plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=str(spawn_data_fp),
        require_build_configs_repeated_equally_across_envs=False,
    )

    try:
        env.reset(seed=args.seed)
        for _ in range(int(args.steps)):
            z = np.zeros((1, env.action_space.shape[1]), dtype=np.float32)
            env.step(z)
        T_world_base, base_link_used = _get_base_pose_in_world(env, args.base_link_name)
        p_all = env.unwrapped.get_sensor_params()
        if not p_all:
            raise RuntimeError("No sensors: get_sensor_params() is empty")
        sensors_out: list[dict] = []
        for name, p in p_all.items():
            if name not in ("fetch_head", "fetch_hand"):
                continue
            rec = _params_to_record(name, p, T_world_base)
            sensors_out.append(rec)
        if not sensors_out:
            available = list(p_all.keys())
            raise RuntimeError(f"No fetch_head / fetch_hand in get_sensor_params. Available: {available}")

        head = next(s for s in sensors_out if s["name"] == "fetch_head")
        payload = {
            "source": (
                "ManiSkill get_sensor_params() at reset(+steps); R,t = OpenCV extrinsic in *robot base* frame "
                "(T_cam_base = extrinsic_cv @ T_world_base). Use with MshabTcpBaseToWorldActions output (base-at-chunk-start)."
            ),
            "env_id": args.env_id,
            "task": args.task,
            "subtask": args.subtask,
            "split": args.split,
            "sim_backend": args.sim_backend,
            "steps_after_reset": int(args.steps),
            "base_link_used": base_link_used,
            "T_world_base": _pad_to_4x4(T_world_base).tolist(),
            "sensors": sensors_out,
            "for_visualize_mshab": {
                "K": head["K"],
                "R": head["R"],
                "t": head["t"],
                "frame": "base",
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}  (base_link_used={base_link_used})", file=sys.stderr)
        print("Use with:", file=sys.stderr)
        print(f"  python scripts/visualize_mshab_relative_traj.py --camera-json {out_path} --config-name <CFG> --video-out out.mp4", file=sys.stderr)
    finally:
        env.close()


if __name__ == "__main__":
    main()
