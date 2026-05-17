#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import random
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tyro
import wandb
from tqdm.auto import tqdm
import time

from openpi import transforms as _transforms
from openpi.models_pytorch.decoders import TrajectoryActionDecoder
from openpi.models_pytorch.decoders import TrajectoryPredictor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_mshab_gripper(action: np.ndarray) -> np.ndarray:
    """13-D MSHAB action -> 12-D action without gripper.

    Original layout is [arm7, gripper1, body3, base2]. We drop index 7.
    """
    return np.concatenate([action[..., :7], action[..., 8:13]], axis=-1)


class MshabTrajectoryChunkDataset(Dataset):
    def __init__(
        self,
        repo_id: str | Path,
        *,
        horizon: int,
        trajectory_dim: int,
        traj_target_offset: int,
        traj_action_fps: float,
        split: Literal["train", "val"],
        val_fraction: float,
        seed: int,
    ):
        self.repo_id = Path(repo_id).expanduser().resolve()
        self.horizon = horizon
        self.trajectory_dim = trajectory_dim
        self.traj_target_offset = traj_target_offset
        self.traj_transform = _transforms.MshabTcpBaseToWorldActions(
            fps=traj_action_fps,
            require_gripper=False,
        )
        if traj_target_offset < 0:
            raise ValueError(f"traj_target_offset must be >= 0; got {traj_target_offset}")
        parquet_paths = sorted((self.repo_id / "data").glob("chunk-*/*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found under {self.repo_id / 'data'}")

        rng = np.random.default_rng(seed)
        self.episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        all_samples: list[tuple[int, int]] = []
        for path in parquet_paths:
            table = pq.read_table(path, columns=["observation.state", "action", "tcp_pose_wrt_base"])
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            tcp_pose_wrt_base = np.asarray(table["tcp_pose_wrt_base"].to_pylist(), dtype=np.float32)
            if len(states) < horizon + traj_target_offset:
                continue
            ep_idx = len(self.episodes)
            self.episodes.append((states, actions, tcp_pose_wrt_base))
            all_samples.extend((ep_idx, start) for start in range(0, len(states) - horizon - traj_target_offset + 1))

        indices = np.arange(len(all_samples))
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_fraction))
        if split == "val":
            selected = indices[:n_val]
        else:
            selected = indices[n_val:]
        self.samples = [all_samples[int(i)] for i in selected]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.samples[idx]
        states, raw_actions, tcp_pose_wrt_base = self.episodes[ep_idx]
        stop = start + self.horizon
        # qpos[t0] + action[t0:t0+H-1] -> trajectory[t0+offset:t0+offset+H-1].
        traj_start = start + self.traj_target_offset
        traj_stop = stop + self.traj_target_offset
        transformed = self.traj_transform(
            {
                "actions": raw_actions[start:traj_stop],
                "tcp_pose_wrt_base": tcp_pose_wrt_base[start:traj_stop],
            }
        )
        actions = strip_mshab_gripper(raw_actions[start:stop]).astype(np.float32)
        traj = np.asarray(transformed["traj"], dtype=np.float32)[self.traj_target_offset :, : self.trajectory_dim]
        return {
            "qpos": torch.from_numpy(states[start]),
            "actions": torch.from_numpy(actions),
            "trajectory": torch.from_numpy(traj),
        }


@dataclasses.dataclass
class TrainConfig:
    task: Literal["trajectory_decoder", "trajectory_predictor"] = "trajectory_predictor"
    repo_id: str | None = None
    config_name: str | None = "pi05_mshab"
    output_dir: str = f"outputs/{task}/{time.strftime('%Y%m%d_%H%M%S')}"
    wandb_project: str = f"openpi-{task}"
    wandb_name: str | None = None
    wandb_mode: str = "online"

    qpos_dim: int = 12
    action_dim: int = 12
    trajectory_dim: int = 7
    horizon: int = 32
    traj_target_offset: int = 1
    traj_action_fps: float = 20.0
    encoder_type: Literal["transformer", "mlp"] = "transformer"
    latent_dim: int = 256
    mlp_hidden_dims: tuple[int, ...] = (512, 512)
    transformer_hidden_dim: int = 256
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_n_head: int = 8
    use_positional_encoding: bool = True
    concat_most_recent_obs: bool = False

    seed: int = 0
    device: str = "cuda"
    batch_size: int = 1024
    num_workers: int = 4
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    val_fraction: float = 0.1
    log_interval: int = 50
    ckpt_save_interval: int = 5


def resolve_repo_id(cfg: TrainConfig) -> str:
    if cfg.repo_id is not None:
        return cfg.repo_id
    if cfg.config_name is None:
        raise ValueError("Pass either repo_id or config_name")
    import openpi.training.config as openpi_train_config

    train_cfg = openpi_train_config.get_config(cfg.config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.repo_id is None:
        raise ValueError(f"Config {cfg.config_name!r} does not define repo_id")
    return data_cfg.repo_id


def build_model(cfg: TrainConfig) -> nn.Module:
    kwargs = dict(
        qpos_dim=cfg.qpos_dim,
        action_dim=cfg.action_dim,
        trajectory_dim=cfg.trajectory_dim,
        horizon=cfg.horizon,
        encoder_type=cfg.encoder_type,
        mlp_hidden_dims=cfg.mlp_hidden_dims,
        transformer_hidden_dim=cfg.transformer_hidden_dim,
        transformer_num_layers=cfg.transformer_num_layers,
        transformer_dim_feedforward=cfg.transformer_dim_feedforward,
        transformer_n_head=cfg.transformer_n_head,
        use_positional_encoding=cfg.use_positional_encoding,
        latent_dim=cfg.latent_dim,
        concat_most_recent_obs=cfg.concat_most_recent_obs,
    )
    if cfg.task == "trajectory_decoder":
        return TrajectoryActionDecoder(**kwargs)
    if cfg.task == "trajectory_predictor":
        return TrajectoryPredictor(**kwargs)
    raise ValueError(f"Unsupported task {cfg.task}")


def forward_loss(model: nn.Module, batch: dict[str, torch.Tensor], task: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = batch["qpos"].to(device)
    actions = batch["actions"].to(device)
    trajectory = batch["trajectory"].to(device)
    if task == "trajectory_decoder":
        pred = model(qpos, trajectory)
        target = actions
    elif task == "trajectory_predictor":
        pred = model(qpos, actions)
        target = trajectory
    else:
        raise ValueError(f"Unsupported task {task}")
    return torch.mean((pred - target) ** 2), pred


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        loss, _ = forward_loss(model, batch, task, device)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    repo_id = resolve_repo_id(cfg)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg) | {"resolved_repo_id": repo_id}, f, indent=2)

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
        config=dataclasses.asdict(cfg) | {"resolved_repo_id": repo_id},
        dir=str(output_dir),
    )

    train_ds = MshabTrajectoryChunkDataset(
        repo_id,
        horizon=cfg.horizon,
        trajectory_dim=cfg.trajectory_dim,
        traj_target_offset=cfg.traj_target_offset,
        traj_action_fps=cfg.traj_action_fps,
        split="train",
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    val_ds = MshabTrajectoryChunkDataset(
        repo_id,
        horizon=cfg.horizon,
        trajectory_dim=cfg.trajectory_dim,
        traj_target_offset=cfg.traj_target_offset,
        traj_action_fps=cfg.traj_action_fps,
        split="val",
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    global_step = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        train_pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{cfg.max_epochs}",
            unit="batch",
            leave=True,
        )
        for batch_idx, batch in enumerate(train_pbar):
            loss, _ = forward_loss(model, batch, cfg.task, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            train_pbar.set_postfix(
                loss=float(loss.item()),
                lr=optimizer.param_groups[0]["lr"],
                refresh=False,
            )

            if global_step % cfg.log_interval == 0:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/epoch": epoch,
                        "train/batch": batch_idx,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            global_step += 1

        val_loss = evaluate(model, val_loader, cfg.task, device)
        print(f"epoch={epoch} val/loss={val_loss:.8f}", flush=True)
        wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
        if (epoch + 1) % cfg.ckpt_save_interval == 0 or epoch == cfg.max_epochs - 1:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": dataclasses.asdict(cfg) | {"resolved_repo_id": repo_id},
                },
                output_dir / f"model_{epoch + 1}.pt",
            )


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
