#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
from typing import Any

import jax
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from openpi import transforms as _transforms
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.eval_pytorch as pi_eval_lib

_EVAL_TRAJ = pathlib.Path(__file__).resolve().with_name("eval_trajectory_decoders.py")
if not _EVAL_TRAJ.is_file():
    raise FileNotFoundError(f"Expected trajectory eval helper at {_EVAL_TRAJ}")


def _load_traj_eval_module():
    import importlib.util

    name = "eval_trajectory_decoders"
    spec = importlib.util.spec_from_file_location(name, _EVAL_TRAJ)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {_EVAL_TRAJ}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_TRAJ_EVAL = _load_traj_eval_module()


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate PI-predicted actions through a trained trajectory predictor: "
            "observation -> PI action chunk, state + action chunk -> predicted traj, compare to GT traj."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config_name", help="OpenPI train config name used for PI model and dataset, e.g. pi05_mshab")
    p.add_argument("--traj-ckpt-path", required=True, help="Trajectory predictor model_*.pt checkpoint.")
    p.add_argument(
        "--traj-model-config-json",
        default=None,
        help="Optional train_trajectory_decoders config.json if traj checkpoint lacks embedded config.",
    )

    p.add_argument(
        "--exp-name",
        default=None,
        help="PI experiment name under checkpoint_base_dir/name/<exp>. Not required with --weights/--checkpoint-root.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Override PI checkpoint root containing numeric step dirs with model.safetensors.",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Load this PI model.safetensors file, or a directory containing model.safetensors.",
    )
    p.add_argument("--checkpoint-step", type=int, default=None, help="PI checkpoint step; default: latest.")
    p.add_argument("--batch-size", type=int, default=None, help="Override global batch size.")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    p.add_argument("--num-flow-steps", type=int, default=10, help="Euler steps for PI sample_actions.")
    p.add_argument("--max-batches", type=int, default=None, help="Limit batches for quick evaluation.")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    p.add_argument("--torch-compile", action="store_true", help="Use torch.compile for PI sampling.")
    return p.parse_args(argv)


def _force_traj_action_data(config: _config.TrainConfig) -> _config.TrainConfig:
    if not hasattr(config.data, "train_traj") or not hasattr(config.data, "traj_action"):
        raise TypeError(f"Config data factory {type(config.data)!r} does not support train_traj/traj_action")
    data = dataclasses.replace(config.data, train_traj=False, traj_action=True)
    return dataclasses.replace(config, data=data)


def _load_pi_model(args: argparse.Namespace, config: _config.TrainConfig, device: torch.device):
    model = pi_eval_lib.build_pi0_pytorch_for_eval(config, device, use_torch_compile=args.torch_compile)
    if args.weights:
        wpath = pi_eval_lib.load_weights_file(model, pathlib.Path(args.weights), device)
        used_step: int | str = f"custom:{wpath}"
    elif args.checkpoint_root:
        used_step = pi_eval_lib.load_eval_checkpoint(
            model,
            pathlib.Path(args.checkpoint_root).expanduser().resolve(),
            step=args.checkpoint_step,
            device=device,
        )
    else:
        used_step = pi_eval_lib.load_eval_checkpoint(
            model,
            config.checkpoint_dir,
            step=args.checkpoint_step,
            device=device,
        )
    logging.info("Loaded PI checkpoint: %s", used_step)
    model.eval()
    return model, used_step


def _load_trajectory_predictor(args: argparse.Namespace, device: torch.device):
    ckpt = _TRAJ_EVAL._load_checkpoint(args.traj_ckpt_path, map_location=device)
    eval_cfg = _TRAJ_EVAL.EvalConfig(
        ckpt_path=args.traj_ckpt_path,
        model_config_json=args.traj_model_config_json,
        device=str(device),
    )
    model_cfg = _TRAJ_EVAL._model_config_from_weight(eval_cfg, ckpt)
    if model_cfg.task != "trajectory_predictor":
        raise ValueError(f"Expected a trajectory_predictor checkpoint, got task={model_cfg.task!r}")
    model = _TRAJ_EVAL.build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    logging.info(
        "Loaded trajectory predictor: horizon=%d action_dim=%d trajectory_dim=%d",
        model_cfg.horizon,
        model_cfg.action_dim,
        model_cfg.trajectory_dim,
    )
    return model, model_cfg


def _build_output_pipeline(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )


def _actions_from_pi_sample(
    sample_out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Pi0 with ``traj2actions`` returns ``(traj_chunk, action_chunk)``; trajectory predictor needs actions."""
    if isinstance(sample_out, tuple):
        if len(sample_out) != 2:
            raise ValueError(
                f"Expected sample_actions to return a Tensor or (traj, actions); got tuple of len {len(sample_out)}"
            )
        return sample_out[1]
    return sample_out


def _apply_output_pipeline_batch(
    actions: torch.Tensor,
    state: torch.Tensor,
    output_pipeline: _transforms.DataTransformFn,
    device: torch.device,
) -> torch.Tensor:
    actions_np = actions.detach().cpu().float().numpy()
    state_np = state.detach().cpu().float().numpy()
    out_actions = []
    for action_i, state_i in zip(actions_np, state_np, strict=True):
        out = output_pipeline({"actions": action_i, "state": state_i})
        out_actions.append(torch.as_tensor(out["actions"], dtype=torch.float32))
    return torch.stack(out_actions, dim=0).to(device)


def _denormalize_state(
    state: torch.Tensor,
    data_config: _config.DataConfig,
    device: torch.device,
) -> torch.Tensor:
    if data_config.norm_stats is None:
        return state
    if "state" not in data_config.norm_stats:
        raise KeyError("data_config.norm_stats does not contain `state`; cannot denormalize observation.state")
    state_np = state.detach().cpu().float().numpy()
    unnormalize = _transforms.Unnormalize(
        {"state": data_config.norm_stats["state"]},
        use_quantiles=data_config.use_quantile_norm,
    )
    out = unnormalize({"state": state_np})
    return torch.as_tensor(out["state"], dtype=torch.float32, device=device)


def _strip_gripper_if_needed(actions: torch.Tensor, target_dim: int) -> torch.Tensor:
    if actions.shape[-1] == target_dim:
        return actions
    if actions.shape[-1] >= 13 and target_dim == 12:
        raw_mshab = actions[..., :13]
        return torch.cat([raw_mshab[..., :7], raw_mshab[..., 8:13]], dim=-1)
    if actions.shape[-1] > target_dim:
        return actions[..., :target_dim]
    raise ValueError(f"Cannot adapt PI actions with shape {tuple(actions.shape)} to target_dim={target_dim}")


def _trim_train_traj_actions(actions: torch.Tensor, target_dim: int) -> torch.Tensor:
    """train_traj data already removed gripper before padding, so the first target_dim columns are used."""
    if actions.shape[-1] < target_dim:
        raise ValueError(f"GT actions shape {tuple(actions.shape)} is shorter than target_dim={target_dim}")
    return actions[..., :target_dim]


def _match_horizon(x: torch.Tensor, horizon: int, name: str) -> torch.Tensor:
    if x.shape[1] < horizon:
        raise ValueError(f"{name} horizon {x.shape[1]} is shorter than required predictor horizon {horizon}")
    return x[:, :horizon]


def _match_last_dim(x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if x.shape[-1] < dim:
        raise ValueError(f"{name} last dim {x.shape[-1]} is shorter than required dim {dim}")
    return x[..., :dim]


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    base = _config.get_config(args.config_name)
    config = dataclasses.replace(
        base,
        exp_name=args.exp_name if args.exp_name is not None else base.exp_name,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else base.num_workers,
        resume=False,
        wandb_enabled=False,
    )
    config = _force_traj_action_data(config)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available; falling back to CPU")
        device = torch.device("cpu")

    pi_model, pi_step = _load_pi_model(args, config, device)
    traj_model, traj_cfg = _load_trajectory_predictor(args, device)
    data_config = config.data.create(config.assets_dirs, config.model)
    output_pipeline = _build_output_pipeline(data_config)

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)

    total_se = 0.0
    total_ae = 0.0
    total_el = 0
    total_samples = 0
    total_action_se = 0.0
    total_action_el = 0
    total_action_se_per_step: torch.Tensor | None = None
    total_action_el_per_step: torch.Tensor | None = None
    n_batches = 0

    try:
        loader_len = len(loader)  # type: ignore[arg-type]
    except TypeError:
        loader_len = None
    pbar_total = min(args.max_batches, loader_len) if args.max_batches is not None and loader_len is not None else args.max_batches or loader_len

    pbar = tqdm(loader, desc="eval pi->traj", total=pbar_total, unit="batch")
    for batch_idx, batch in enumerate(pbar):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if not isinstance(batch, tuple) or len(batch) != 3:
            raise ValueError(
                "Expected dataloader to yield (observation, traj, actions). "
                "This script forces train_traj=False and traj_action=True; check LeRobotMshabDataConfig."
            )
        observation, gt_traj, gt_actions = batch
        observation = jax.tree.map(lambda x: x.to(device), observation)
        gt_traj = gt_traj.to(device=device, dtype=torch.float32)
        gt_actions = gt_actions.to(device=device, dtype=torch.float32)

        state_for_output = observation.state.to(device=device, dtype=torch.float32)
        pi_actions = _actions_from_pi_sample(
            pi_model.sample_actions(device, observation, num_steps=args.num_flow_steps)
        )
        pi_actions = _apply_output_pipeline_batch(pi_actions, state_for_output, output_pipeline, device)
        pi_actions = _strip_gripper_if_needed(pi_actions.to(dtype=torch.float32), traj_cfg.action_dim)
        pi_actions = _match_horizon(pi_actions, traj_cfg.horizon, "PI actions")
        gt_actions = _apply_output_pipeline_batch(gt_actions, state_for_output, output_pipeline, device)
        ng_actions = _strip_gripper_if_needed(gt_actions.to(dtype=torch.float32), traj_cfg.action_dim)
        ng_actions = _match_horizon(ng_actions, traj_cfg.horizon, "NG actions")
        gt_traj = _match_horizon(gt_traj, traj_cfg.horizon, "GT traj")
        gt_traj = _match_last_dim(gt_traj, traj_cfg.trajectory_dim, "GT traj")

        qpos = _denormalize_state(state_for_output, data_config, device)[..., : traj_cfg.qpos_dim]
        pred_traj = traj_model(qpos, pi_actions)
        ng_traj = traj_model(qpos, ng_actions)
        if pred_traj.shape[1] < 2 or gt_traj.shape[1] < 2:
            raise ValueError("Trajectory loss alignment requires horizon >= 2.")
        pred_traj = pred_traj[:, :-1]
        ng_traj = ng_traj[:, :-1]
        gt_traj = gt_traj[:, 1:]
        err = pred_traj - gt_traj
        se = F.mse_loss(pred_traj, gt_traj, reduction="none")
        se_ng = F.mse_loss(ng_traj, gt_traj, reduction="none")
        ae = err.abs()

        total_se += se.sum().item()
        total_ae += ae.sum().item()
        total_el += se.numel()
        total_samples += int(qpos.shape[0])
        n_batches += 1

        gt_actions = _strip_gripper_if_needed(gt_actions, traj_cfg.action_dim)
        gt_actions = _match_horizon(gt_actions, traj_cfg.horizon, "GT actions")
        action_loss = float("nan")
        if gt_actions.shape[-1] == pi_actions.shape[-1]:
            action_se = F.mse_loss(pi_actions, gt_actions, reduction="none")
            action_loss = float(action_se.mean().item())
            total_action_se += action_se.sum().item()
            total_action_el += action_se.numel()
            action_se_per_step = action_se.sum(dim=(0, 2)).detach().cpu()
            action_el_per_step = torch.full_like(
                action_se_per_step,
                fill_value=action_se.shape[0] * action_se.shape[2],
            )
            if total_action_se_per_step is None:
                total_action_se_per_step = torch.zeros_like(action_se_per_step)
                total_action_el_per_step = torch.zeros_like(action_el_per_step)
            total_action_se_per_step += action_se_per_step
            total_action_el_per_step += action_el_per_step

        running_mse = total_se / max(total_el, 1)
        pbar.set_postfix(
            traj_loss=float(se.mean().item()),
            ng_loss=float(se_ng.mean().item()),
            action_loss=action_loss,
            running=float(running_mse),
            refresh=False,
        )

    traj_mse = total_se / max(total_el, 1)
    traj_mae = total_ae / max(total_el, 1)
    out = {
        "traj_loss": float(traj_mse),
        "traj_mse": float(traj_mse),
        "traj_mae": float(traj_mae),
        "num_batches": int(n_batches),
        "num_samples": int(total_samples),
        "num_elements": int(total_el),
        "config_name": args.config_name,
        "pi_checkpoint": str(pi_step),
        "traj_ckpt_path": str(pathlib.Path(args.traj_ckpt_path).expanduser().resolve()),
        "horizon": int(traj_cfg.horizon),
        "traj_target_offset": int(traj_cfg.traj_target_offset),
    }
    if total_action_el:
        out["pi_action_mse_vs_gt_actions"] = float(total_action_se / total_action_el)
    if total_action_se_per_step is not None and total_action_el_per_step is not None:
        per_step = total_action_se_per_step / torch.clamp(total_action_el_per_step, min=1)
        out["pi_action_mse_per_step"] = [float(x) for x in per_step.tolist()]
    return out


def main(argv: list[str] | None = None) -> None:
    init_logging()
    args = parse_args(argv)
    if args.exp_name is None and args.checkpoint_root is None and args.weights is None:
        logging.error("Provide PI --exp-name, --checkpoint-root, or --weights")
        sys.exit(2)
    metrics = run_eval(args)
    logging.info("traj_loss=%.8f", metrics["traj_loss"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
