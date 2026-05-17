#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader
import tyro
from tqdm.auto import tqdm

# Reuse dataset / model construction from the training script (same file dir).
_TRAIN = Path(__file__).resolve().with_name("train_trajectory_decoders.py")
if not _TRAIN.is_file():
    raise FileNotFoundError(f"Expected training script at {_TRAIN}")


def _load_training_module():
    import importlib.util

    name = "train_trajectory_decoders"
    spec = importlib.util.spec_from_file_location(name, _TRAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {_TRAIN}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_TRAIN_MOD = _load_training_module()
TrainConfig = _TRAIN_MOD.TrainConfig
build_model = _TRAIN_MOD.build_model
MshabTrajectoryChunkDataset = _TRAIN_MOD.MshabTrajectoryChunkDataset
resolve_repo_id = _TRAIN_MOD.resolve_repo_id


@dataclasses.dataclass
class EvalConfig:
    """Evaluate trajectory decoder / predictor weights on a dataset resolved from a config."""

    ckpt_path: str
    """Path to model_*.pt from train_trajectory_decoders."""

    config_name: str | None = "pi05_mshab"
    """OpenPI training config used only to locate the evaluation dataset."""

    repo_id: str | None = None
    """Optional dataset path/repo. If set, this overrides config_name."""

    model_config_json: str | None = None
    """Optional training config.json for model construction when checkpoint lacks config."""

    split: Literal["val", "train"] = "val"
    batch_size: int = 256
    num_workers: int = 4
    device: str = "cuda"
    max_batches: int | None = None


def _train_config_from_dict(d: dict[str, Any]) -> TrainConfig:
    field_names = {f.name for f in dataclasses.fields(TrainConfig)}
    allowed_task = {"trajectory_decoder", "trajectory_predictor"}
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k not in field_names:
            continue
        if k == "task" and v not in allowed_task:
            raise ValueError(f"Invalid task in config: {v!r}; expected one of {allowed_task}")
        if k == "mlp_hidden_dims" and isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    return TrainConfig(**kwargs)


def _load_checkpoint(ckpt_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(ckpt_path).expanduser().resolve()
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)!r} from {path}")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {path} must contain 'model_state_dict'")
    return ckpt


def _load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config JSON must contain an object, got {type(cfg)!r}")
    return cfg


def _model_config_from_weight(eval_cfg: EvalConfig, ckpt: dict[str, Any]) -> TrainConfig:
    ckpt_cfg = ckpt.get("config")
    d: dict[str, Any] = dict(ckpt_cfg) if isinstance(ckpt_cfg, dict) else {}
    if not d and eval_cfg.model_config_json is not None:
        d = _load_json_config(eval_cfg.model_config_json)
    if not d:
        raise ValueError(
            "No model config found in checkpoint and --model-config-json was not provided. "
            "Use a checkpoint saved by train_trajectory_decoders.py or pass that run's config.json."
        )
    return _train_config_from_dict(d)


def _dataset_repo_from_config(eval_cfg: EvalConfig, model_cfg: TrainConfig, ckpt: dict[str, Any]) -> str:
    if eval_cfg.repo_id is not None:
        return eval_cfg.repo_id
    if eval_cfg.config_name is not None:
        return resolve_repo_id(dataclasses.replace(model_cfg, config_name=eval_cfg.config_name, repo_id=None))

    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict) and isinstance(ckpt_cfg.get("resolved_repo_id"), str):
        return ckpt_cfg["resolved_repo_id"]
    if model_cfg.repo_id is not None:
        return model_cfg.repo_id
    if model_cfg.config_name is not None:
        return resolve_repo_id(model_cfg)
    raise ValueError("Could not resolve dataset. Pass --config-name or --repo-id.")


@torch.no_grad()
def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    ckpt = _load_checkpoint(cfg.ckpt_path, map_location=device)
    model_cfg = _model_config_from_weight(cfg, ckpt)
    repo_id = _dataset_repo_from_config(cfg, model_cfg, ckpt)

    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = MshabTrajectoryChunkDataset(
        repo_id,
        horizon=model_cfg.horizon,
        trajectory_dim=model_cfg.trajectory_dim,
        traj_target_offset=model_cfg.traj_target_offset,
        split=cfg.split,
        val_fraction=model_cfg.val_fraction,
        seed=model_cfg.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    mse_num = 0.0
    mae_num = 0.0
    count = 0
    loss_sum = 0.0
    sample_count = 0
    pbar = tqdm(loader, desc=f"eval {cfg.split}", unit="batch", leave=True)
    for b_idx, batch in enumerate(pbar):
        if cfg.max_batches is not None and b_idx >= cfg.max_batches:
            break
        q = batch["qpos"].to(device)
        a = batch["actions"].to(device)
        t = batch["trajectory"].to(device)
        if model_cfg.task == "trajectory_decoder":
            pred = model(q, t)
            target = a
        else:
            pred = model(q, a)
            target = t
        err = pred - target
        mse_num += (err**2).sum().item()
        mae_num += err.abs().sum().item()
        count += int(err.numel())
        batch_size = int(q.shape[0])
        loss_sum += float((err**2).mean().item()) * batch_size
        sample_count += batch_size
        pbar.set_postfix(
            loss=float((err**2).mean().item()),
            mae=float(err.abs().mean().item()),
            refresh=False,
        )

    mse = mse_num / count if count else float("nan")
    mae = mae_num / count if count else float("nan")
    loss = loss_sum / sample_count if sample_count else float("nan")
    return {
        "loss": float(loss),
        "mse": float(mse),
        "mae": float(mae),
        "task": model_cfg.task,
        "split": cfg.split,
        "repo_id": repo_id,
        "horizon": model_cfg.horizon,
        "traj_target_offset": model_cfg.traj_target_offset,
        "num_elements": int(count),
        "num_samples": int(sample_count),
        "dataset_samples": int(len(ds)),
    }


def main() -> None:
    cfg = tyro.cli(EvalConfig)
    metrics = run_eval(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
