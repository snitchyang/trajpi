from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import torch
from torch import Tensor

from openpi.models_pytorch.decoders.trajectory_predictor import TrajectoryPredictor


class ActionTrajectoryAligner:
    """Align policy actions to trajectory predictor outputs with test-time gradient descent."""

    def __init__(
        self,
        *,
        predictor_ckpt_path: str,
        device: str,
        num_steps: int = 8,
        lr: float = 1,
        model_config_json: str | None = None,
        traj_target_offset: int | None = None,
        gripper_index_to_drop: int | None = 7,
    ):
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")

        self._device = torch.device(device)
        self._num_steps = num_steps
        self._lr = lr
        self._gripper_index_to_drop = gripper_index_to_drop
        self._predictor, predictor_cfg = self._load_predictor(
            predictor_ckpt_path=predictor_ckpt_path,
            model_config_json=model_config_json,
        )
        self._predictor.eval()
        for p in self._predictor.parameters():
            p.requires_grad_(False)

        self._qpos_dim = int(predictor_cfg["qpos_dim"])
        self._action_dim = int(predictor_cfg["action_dim"])
        self._traj_dim = int(predictor_cfg["trajectory_dim"])
        self._horizon = int(predictor_cfg["horizon"])
        ckpt_offset = int(predictor_cfg.get("traj_target_offset", 1))
        self._traj_target_offset = ckpt_offset if traj_target_offset is None else int(traj_target_offset)

    def align(
        self,
        *,
        state: np.ndarray,
        actions: np.ndarray,
        traj: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Optimize actions to minimize MSE between predictor(actions) and policy trajectory."""
        state_t = self._to_batch_tensor(state)
        actions_t = self._to_batch_tensor(actions)
        traj_t = self._to_batch_tensor(traj)

        qpos = state_t[..., : self._qpos_dim].to(dtype=torch.float32, device=self._device)
        target_traj = traj_t[..., : self._traj_dim].to(dtype=torch.float32, device=self._device)
        target_traj = target_traj[:, : self._horizon]

        optimized_actions = actions_t.to(dtype=torch.float32, device=self._device).clone().detach()
        optimized_actions = optimized_actions[:, : self._horizon]
        optimized_actions.requires_grad_(True)
        optimizer = torch.optim.SGD([optimized_actions], lr=self._lr)

        initial_loss = None
        final_loss = None
        for _ in range(self._num_steps):
            optimizer.zero_grad()
            predictor_actions = self._adapt_actions_for_predictor(optimized_actions)
            predicted_traj = self._predictor(qpos, predictor_actions)
            predicted_traj, aligned_target = self._align_prediction_with_target(predicted_traj, target_traj)
            # loss = torch.mean((predicted_traj - aligned_target) ** 2)
            # Use L1 loss
            loss = torch.mean(torch.abs(predicted_traj - aligned_target))
            if initial_loss is None:
                initial_loss = float(loss.item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        # Use target trajectory gripper (dim 8, index 7) for the optimized actions.
        with torch.no_grad():
            if traj_t.shape[-1] > 7 and optimized_actions.shape[-1] > 7:
                t_end = min(traj_t.shape[1], optimized_actions.shape[1])
                optimized_actions[:, :t_end, 7] = traj_t[:, :t_end, 7]

        out = optimized_actions.detach().cpu().numpy()
        if np.ndim(actions) == 2:
            out = out[0]

        return out, {
            "enabled": 1.0,
            "initial_loss": 0.0 if initial_loss is None else initial_loss,
            "final_loss": 0.0 if final_loss is None else final_loss,
            "num_steps": float(self._num_steps),
            "lr": float(self._lr),
        }

    def _align_prediction_with_target(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        # Predictor is trained so its step t predicts next-step trajectory (aligned to PI traj at t+1).
        # Therefore, compute loss with PI trajectory [1:] against predictor trajectory [:-1].
        offset = self._traj_target_offset
        if offset <= 0:
            return pred, target
        if pred.shape[1] <= offset or target.shape[1] <= offset:
            raise ValueError(
                f"Cannot apply traj_target_offset={offset} with pred/target horizon "
                f"{pred.shape[1]}/{target.shape[1]}"
            )
        return pred[:, :-offset], target[:, offset:]

    def _adapt_actions_for_predictor(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] == self._action_dim:
            return actions
        if (
            self._gripper_index_to_drop is not None
            and actions.shape[-1] >= self._gripper_index_to_drop + 1
            and actions.shape[-1] - 1 == self._action_dim
        ):
            idx = self._gripper_index_to_drop
            return torch.cat([actions[..., :idx], actions[..., idx + 1 :]], dim=-1)
        if actions.shape[-1] > self._action_dim:
            return actions[..., : self._action_dim]
        raise ValueError(
            f"Cannot adapt action dim from {actions.shape[-1]} to predictor action_dim={self._action_dim}"
        )

    def _to_batch_tensor(self, x: np.ndarray) -> Tensor:
        t = torch.as_tensor(x)
        if t.ndim == 1:
            return t.unsqueeze(0)
        if t.ndim == 2:
            return t.unsqueeze(0)
        return t

    def _load_predictor(self, *, predictor_ckpt_path: str, model_config_json: str | None) -> tuple[TrajectoryPredictor, dict]:
        ckpt_path = pathlib.Path(predictor_ckpt_path).expanduser().resolve()
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Predictor checkpoint must be a dict, got {type(ckpt)!r}")
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Predictor checkpoint is missing model_state_dict: {ckpt_path}")

        cfg = ckpt.get("config")
        if not isinstance(cfg, dict):
            cfg = {}
        if not cfg and model_config_json is not None:
            with pathlib.Path(model_config_json).expanduser().resolve().open("r", encoding="utf-8") as f:
                loaded_cfg = json.load(f)
            if not isinstance(loaded_cfg, dict):
                raise ValueError("Predictor model config json must contain an object")
            cfg = loaded_cfg
        if not cfg:
            default_cfg_path = ckpt_path.with_name("config.json")
            if default_cfg_path.is_file():
                with default_cfg_path.open("r", encoding="utf-8") as f:
                    loaded_cfg = json.load(f)
                if not isinstance(loaded_cfg, dict):
                    raise ValueError(f"Predictor config at {default_cfg_path} must contain an object")
                cfg = loaded_cfg

        if not cfg:
            raise ValueError(
                "No predictor config found in checkpoint. Provide model_config_json, place config.json next to "
                "predictor checkpoint, or use checkpoint saved by train_trajectory_decoders.py"
            )
        if cfg.get("task") not in (None, "trajectory_predictor"):
            raise ValueError(f"Expected trajectory_predictor checkpoint, got task={cfg.get('task')!r}")

        predictor = TrajectoryPredictor(
            qpos_dim=int(cfg.get("qpos_dim", 12)),
            action_dim=int(cfg.get("action_dim", 12)),
            trajectory_dim=int(cfg.get("trajectory_dim", 7)),
            horizon=int(cfg.get("horizon", 32)),
            encoder_type=str(cfg.get("encoder_type", "transformer")),
            mlp_hidden_dims=tuple(cfg.get("mlp_hidden_dims", (512, 512))),
            transformer_hidden_dim=int(cfg.get("transformer_hidden_dim", 256)),
            transformer_num_layers=int(cfg.get("transformer_num_layers", 4)),
            transformer_dim_feedforward=int(cfg.get("transformer_dim_feedforward", 1024)),
            transformer_n_head=int(cfg.get("transformer_n_head", 8)),
            use_positional_encoding=bool(cfg.get("use_positional_encoding", True)),
            latent_dim=int(cfg.get("latent_dim", 256)),
            concat_most_recent_obs=bool(cfg.get("concat_most_recent_obs", False)),
        ).to(self._device)
        predictor.load_state_dict(ckpt["model_state_dict"], strict=True)
        return predictor, cfg
