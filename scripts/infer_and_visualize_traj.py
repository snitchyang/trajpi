#!/usr/bin/env python3
"""
Run inference with a trained Pi0 / Pi05 PyTorch checkpoint on N consecutive samples from the
training dataset and render an MP4 where each frame is the top camera image at time ``t`` with
the **model-predicted action chunk** projected onto it.

What this script does (per frame, index ``t``):
  1. Pull ``ds[t]`` from the LeRobot dataset (same delta_timestamps/action_sequence_keys as training).
  2. Run the full training-time transform chain on it:
         repack_transforms.inputs
         → data_transforms.inputs
         → Normalize(norm_stats)
         → model_transforms.inputs (resize 224, tokenize, pad to model action_dim)
     This produces a single unbatched sample ready for the model.
  3. Stack a batch dim, move to device, build ``Observation`` and call
     ``model.sample_actions(device, obs, num_steps=--num-flow-steps)``.
  4. Reverse the output side of the pipeline:
         model_transforms.outputs → Unnormalize(norm_stats) → data_transforms.outputs
     For ``LeRobotMshabDataConfig(traj_action=True)`` this yields ``(H, 8)`` = world-frame
     TCP xyz+quat + gripper expressed in the base-at-chunk-start frame.
  5. Project the predicted ``(H, 3)`` xyz onto the current top image using the same camera
     extrinsics + per-frame torso compensation as :mod:`scripts.visualize_mshab_relative_traj`.
  6. Optionally draw the GT chunk (from the repack-only pipeline) for side-by-side comparison.

Usage:
  # Default checkpoint layout from the config (config.checkpoint_dir / <step>)
  python scripts/infer_and_visualize_traj.py pi05_mstraj \
    --exp-name=openpi --checkpoint-step=20000 \
    --num-samples 200 --video-out pred.mp4 \
    --camera-json mshab_fetch_head_base.json

  # Custom weights (file or dir containing model.safetensors)
  python scripts/infer_and_visualize_traj.py pi05_mstraj \
    --weights /path/to/20000/model.safetensors \
    --num-samples 200 --video-out pred.mp4 \
    --camera-json mshab_fetch_head_base.json

  # Overlay GT polyline as well (red)
  python scripts/infer_and_visualize_traj.py pi05_mstraj \
    --weights ... --num-samples 200 --video-out compare.mp4 \
    --camera-json mshab_fetch_head_base.json --overlay-gt

Notes:
  - Requires the same dataset + norm stats that the checkpoint was trained with; we pull both
    from ``_config.get_config(<config_name>)``.
  - For non-mshab configs, ``pred_actions[:, :3]`` may not be world-frame xyz. This projection
    code is specialised to the mshab ``traj_action=True`` setup.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_OPI_ROOT = _SCRIPT.parent.parent
_OPI_SRC = _OPI_ROOT / "src"
if _OPI_SRC.is_dir() and str(_OPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPI_SRC))
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

from openpi.models import model as _model  # noqa: E402
from openpi.policies import libero_policy  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.training.eval_pytorch as eval_lib  # noqa: E402

import visualize_mshab_relative_traj as _vis  # noqa: E402


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_input_pipeline(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    """Full training-time input chain: repack → data_transforms → Normalize → model_transforms."""
    return _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _build_output_pipeline(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    """Reverse chain for predicted actions: model_transforms.outputs → Unnormalize → data_transforms.outputs.

    For mshab ``traj_action=True``, ``data_transforms.outputs`` is ``LiberoOutputs(action_dims=8)``,
    so the final ``actions`` are shape ``(H, 8)`` = xyz + quat + gripper (world/base-at-t0 frame).
    """
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )


def _build_repack_pipeline(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    """repack_transforms.inputs only.

    For mshab with ``traj_action=True`` this already includes ``MshabTcpBaseToWorldActions``, so the
    ``actions`` entry here is GT world-frame TCP (H, 8), and ``observation/image`` is the unresized
    top frame we render onto.
    """
    return _transforms.compose(list(data_config.repack_transforms.inputs))


def _apply_with_prompt(fn: _transforms.DataTransformFn, raw: dict, default_prompt: str | None = None) -> dict:
    """Apply a transform chain on a LeRobot raw sample. Inject an empty ``prompt`` if absent so
    downstream repack configs that map ``"prompt": "prompt"`` do not KeyError during viz-only use."""
    d = _vis._data_to_numpy(raw)
    if default_prompt is not None:
        d["prompt"] = default_prompt
    elif "prompt" not in d:
        d["prompt"] = ""
    return fn(d)


def _batchify_torch(sample: dict, device: torch.device) -> dict:
    """Add a leading batch dim to each leaf and convert to torch tensors on ``device``.

    Mirrors training DataLoader's ``_collate_fn`` (``np.stack([arr], axis=0)``) + ``torch.as_tensor``:
      - 0-d leaf (e.g. scalar ``np.True_`` from ``LiberoInputs.image_mask``) → ``(1,)``.
      - n-d leaf with shape ``S`` → ``(1, *S)``.

    Do *not* pre-promote scalars via ``np.ascontiguousarray`` — that enforces ``ndim>=1`` and ends
    up giving ``image_mask`` shape ``(1, 1)``, which fails ``Observation.image_masks``' jaxtyping
    contract ``Bool[Tensor, '*b']``.

    Skips any residual string leaves (they should have been consumed by ``TokenizePrompt``).
    """
    out: dict = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            out[k] = _batchify_torch(v, device)
            continue
        if isinstance(v, str) or (isinstance(v, np.ndarray) and v.dtype.kind in {"U", "S", "O"}):
            continue
        arr = np.asarray(v)
        stacked = np.stack([arr], axis=0)
        out[k] = torch.as_tensor(stacked).to(device)
    return out


@torch.no_grad()
def _predict_actions(
    model,
    device: torch.device,
    full_sample: dict,
    output_pipeline: _transforms.DataTransformFn,
    num_flow_steps: int,
) -> np.ndarray:
    """Run ``sample_actions`` then reverse the normalize/pad pipeline. Returns ``(H, action_dims)``."""
    batch = _batchify_torch(full_sample, device)
    obs = _model.Observation.from_dict(batch)
    pred = model.sample_actions(device, obs, num_steps=num_flow_steps)
    pred_np = pred[0].detach().cpu().float().numpy()

    state_np = np.asarray(full_sample["state"], dtype=np.float32)
    out = output_pipeline({"actions": pred_np, "state": state_np})
    return np.asarray(out["actions"], dtype=np.float64)


def _draw_polyline(bgr: np.ndarray, uv: np.ndarray, color_bgr: tuple[int, int, int], *, thick_div: int = 64) -> None:
    """Draw a polyline + dots on ``bgr`` in place."""
    if uv.size == 0:
        return
    h, w = bgr.shape[:2]
    line_thick = max(1, w // thick_div)
    dot_r = max(1, w // (thick_div * 2))
    n = uv.shape[0]
    for t in range(1, n):
        p0 = (int(round(uv[t - 1, 0])), int(round(uv[t - 1, 1])))
        p1 = (int(round(uv[t, 0])), int(round(uv[t, 1])))
        cv2.line(bgr, p0, p1, color_bgr, line_thick, lineType=cv2.LINE_AA)
    for t in range(n):
        cx = int(round(uv[t, 0]))
        cy = int(round(uv[t, 1]))
        cv2.circle(bgr, (cx, cy), dot_r, color_bgr, -1, lineType=cv2.LINE_AA)
    cv2.circle(bgr, (int(round(uv[0, 0])), int(round(uv[0, 1]))), max(2, w // 64), (255, 0, 0), 1, lineType=cv2.LINE_AA)


def _resolve_weights(
    args: argparse.Namespace,
    config: _config.TrainConfig,
    model,
    device: torch.device,
) -> str:
    if args.weights:
        p = eval_lib.load_weights_file(model, pathlib.Path(args.weights), device)
        return f"custom:{p}"
    root = (
        pathlib.Path(args.checkpoint_root).expanduser().resolve()
        if args.checkpoint_root is not None
        else config.checkpoint_dir
    )
    step = eval_lib.load_eval_checkpoint(model, root, step=args.checkpoint_step, device=device)
    return f"step:{step}@{root}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config_name", help="Train config name, e.g. pi05_mstraj")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="model.safetensors file or a directory containing it. Overrides --checkpoint-root.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Checkpoint root (parent of numeric step dirs). Default: config.checkpoint_dir.",
    )
    p.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Numeric step subdir under checkpoint root. Default: latest.",
    )
    p.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Override exp_name (only used when resolving the default checkpoint_dir layout).",
    )

    p.add_argument("--num-samples", type=int, default=200, help="Number of dataset indices to infer + render.")
    p.add_argument("--start-index", type=int, default=0, help="First dataset index.")
    p.add_argument("--sample-stride", type=int, default=1, help="Index stride between consecutive frames.")
    p.add_argument("--repo-id", type=str, default=None, help="Override DataConfig.repo_id.")
    p.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Override model.action_horizon (also drives delta_timestamps for the dataset).",
    )
    p.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Force a prompt for all samples. If omitted, the dataset's task-derived prompt is used "
        "(when ``prompt_from_task=True``) or an empty string.",
    )

    p.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    p.add_argument("--num-flow-steps", type=int, default=10, help="Euler steps inside sample_actions.")

    p.add_argument("--video-out", type=str, required=True, help="Output MP4 path.")
    p.add_argument("--video-fps", type=float, default=None, help="Output FPS. Default: dataset meta fps.")

    p.add_argument(
        "--camera-json",
        type=str,
        default=None,
        help="Camera JSON from scripts/dump_mshab_fetch_camera.py (base-frame R,t,K). If omitted, "
        "falls back to the look-at virtual camera (--cam-eye/--cam-target in base frame).",
    )
    p.add_argument(
        "--cam-eye",
        type=float,
        nargs=3,
        default=(-0.12, 0.0, 1.3),
        metavar=("X", "Y", "Z"),
        help="Fallback virtual camera eye in base frame (ignored when --camera-json is set).",
    )
    p.add_argument(
        "--cam-target",
        type=float,
        nargs=3,
        default=(0.6, 0.0, 0.7),
        metavar=("X", "Y", "Z"),
        help="Fallback virtual camera look-at target in base frame.",
    )
    p.add_argument(
        "--cam-up",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("UX", "UY", "UZ"),
        help="Fallback virtual camera up vector.",
    )

    p.add_argument(
        "--torso-ref-qpos",
        type=float,
        default=_vis.FETCH_REST_TORSO,
        help=f"Reference torso_lift_joint value (meters) at which the camera JSON was captured. "
        f"Default {_vis.FETCH_REST_TORSO} (Fetch rest keyframe).",
    )
    p.add_argument(
        "--no-torso-compensation",
        dest="torso_compensation",
        action="store_false",
        default=True,
        help="Disable per-frame torso-lift compensation (reproduces the drifting behavior).",
    )

    p.add_argument(
        "--overlay-gt",
        action="store_true",
        help="Also draw the dataset's GT chunk trajectory (red) for visual comparison.",
    )
    return p.parse_args()


def main() -> None:
    _init_logging()
    args = parse_args()

    base = _config.get_config(args.config_name)
    config = dataclasses.replace(
        base,
        exp_name=args.exp_name if args.exp_name is not None else base.exp_name,
        resume=False,
        wandb_enabled=False,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available; falling back to CPU")
        device = torch.device("cpu")

    logging.info("Building model for config=%s on %s", args.config_name, device)
    model = eval_lib.build_pi0_pytorch_for_eval(config, device, use_torch_compile=False)
    weights_tag = _resolve_weights(args, config, model, device)
    logging.info("Loaded weights: %s", weights_tag)
    model.eval()

    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = args.repo_id or data_config.repo_id
    if repo_id is None:
        raise SystemExit("No repo_id in DataConfig; pass --repo-id.")
    if data_config.norm_stats is None:
        raise SystemExit(
            "DataConfig.norm_stats is None; cannot un-normalize predicted actions. "
            "Run scripts/compute_norm_stats.py for this config first."
        )
    action_horizon = int(args.action_horizon) if args.action_horizon is not None else int(config.model.action_horizon)
    logging.info("repo_id=%s  action_horizon=%d", repo_id, action_horizon)

    meta = lerobot_dataset.LeRobotDatasetMetadata(str(repo_id))
    meta_fps = float(meta.fps) if meta.fps else 20.0
    delta_t = 1.0 / meta_fps
    delta_keys = {k: [i * delta_t for i in range(action_horizon)] for k in data_config.action_sequence_keys}
    raw_ds = lerobot_dataset.LeRobotDataset(str(repo_id), delta_timestamps=delta_keys)
    if data_config.prompt_from_task:
        raw_ds = _data_loader.TransformedDataset(raw_ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])

    input_pipeline = _build_input_pipeline(data_config)
    output_pipeline = _build_output_pipeline(data_config)
    repack_pipeline = _build_repack_pipeline(data_config)

    r_base2cam, t_base2cam, k_from_file = _vis._load_camera_from_args(args)

    stride = max(1, int(args.sample_stride))
    num = int(args.num_samples)
    start = int(args.start_index)
    raw_len = len(raw_ds)
    indices = [start + stride * i for i in range(num)]
    indices = [i for i in indices if 0 <= i < raw_len]
    if not indices:
        raise SystemExit(f"No valid indices (len(ds)={raw_len}, start={start}, stride={stride}, num={num}).")

    video_fps = float(args.video_fps) if args.video_fps is not None else meta_fps
    out_path = Path(args.video_out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None
    n_written = 0
    missing_state_warned = False

    logging.info(
        "Encoding %d frames %d..%d (stride %d) → %s @ %.2f fps (flow_steps=%d, overlay_gt=%s, torso_comp=%s)",
        len(indices),
        indices[0],
        indices[-1],
        stride,
        out_path,
        video_fps,
        args.num_flow_steps,
        bool(args.overlay_gt),
        bool(args.torso_compensation),
    )

    for step_i, idx in enumerate(indices):
        try:
            raw = raw_ds[idx]
        except Exception as e:  # noqa: BLE001
            logging.warning("skip idx=%d: cannot load raw sample (%s)", idx, e)
            continue

        try:
            repack_sample = _apply_with_prompt(repack_pipeline, raw, default_prompt=args.default_prompt)
            full_sample = _apply_with_prompt(input_pipeline, raw, default_prompt=args.default_prompt)
        except Exception as e:  # noqa: BLE001
            logging.warning("skip idx=%d: transform chain failed (%s)", idx, e)
            continue

        try:
            base_img = libero_policy._parse_image(repack_sample["observation/image"])
        except Exception as e:  # noqa: BLE001
            logging.warning("skip idx=%d: missing/unparseable observation/image (%s)", idx, e)
            continue
        hi, wi = int(base_img.shape[0]), int(base_img.shape[1])
        bgr = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)

        t_cam = t_base2cam
        q_torso = _vis._current_torso_qpos(repack_sample) if args.torso_compensation else None
        if args.torso_compensation and q_torso is not None:
            t_cam = _vis._torso_compensated_tvec(r_base2cam, t_base2cam, q_torso, args.torso_ref_qpos)
        elif args.torso_compensation and not missing_state_warned:
            logging.warning("observation/state missing; disabling torso compensation for remaining frames.")
            missing_state_warned = True

        try:
            pred_actions = _predict_actions(
                model=model,
                device=device,
                full_sample=full_sample,
                output_pipeline=output_pipeline,
                num_flow_steps=int(args.num_flow_steps),
            )
        except Exception as e:  # noqa: BLE001
            logging.warning("skip idx=%d: inference failed (%s)", idx, e)
            continue

        if pred_actions.ndim != 2 or pred_actions.shape[-1] < 3:
            logging.warning(
                "skip idx=%d: predicted actions shape %s cannot be projected (need >=3 dims)",
                idx,
                pred_actions.shape,
            )
            continue
        pred_xyz = pred_actions[:, :3]

        if args.overlay_gt and "actions" in repack_sample:
            try:
                gt_actions = np.asarray(repack_sample["actions"], dtype=np.float64)
                if gt_actions.ndim == 2 and gt_actions.shape[-1] >= 3:
                    uv_gt = _vis._project_world_points(
                        gt_actions[:, :3], (hi, wi), r_base2cam, t_cam, k_mat=k_from_file
                    )
                    _draw_polyline(bgr, uv_gt, (0, 0, 255), thick_div=96)
            except Exception as e:  # noqa: BLE001
                logging.debug("GT overlay skipped at idx=%d: %s", idx, e)

        uv_pred = _vis._project_world_points(pred_xyz, (hi, wi), r_base2cam, t_cam, k_mat=k_from_file)
        _draw_polyline(bgr, uv_pred, (0, 255, 0), thick_div=64)

        overlay = f"idx={idx} H={pred_xyz.shape[0]} flow={args.num_flow_steps}"
        if args.torso_compensation and q_torso is not None:
            overlay += f" torso={q_torso:.3f}/{args.torso_ref_qpos:.3f}"
        cv2.putText(bgr, overlay, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(bgr, "pred", (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        if args.overlay_gt:
            cv2.putText(bgr, "gt", (46, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (wi, hi))
            if not writer.isOpened():
                raise SystemExit(f"Failed to open VideoWriter for {out_path}")
        if (hi, wi) != (bgr.shape[0], bgr.shape[1]):
            bgr = cv2.resize(bgr, (wi, hi), interpolation=cv2.INTER_AREA)
        writer.write(bgr)
        n_written += 1
        if (step_i + 1) % 25 == 0:
            logging.info("encoded %d / %d frames (last idx=%d)", step_i + 1, len(indices), idx)

    if writer is not None:
        writer.release()

    if n_written == 0:
        logging.warning("no frames were encoded; check dataset indices / repack alignment.")
    else:
        logging.info("wrote %d frames → %s", n_written, out_path)


if __name__ == "__main__":
    main()
