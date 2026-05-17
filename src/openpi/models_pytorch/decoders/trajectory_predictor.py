from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.decoders.common import ConditionedChunkModel


class TrajectoryPredictor(nn.Module):
    """Predict a future TCP trajectory from current qpos and an action chunk.

    This is the learned forward-dynamics side of the pair:

    ``qpos_t, action_{t:t+H-1} -> tcp_trajectory_{t:t+H-1}``

    It intentionally matches :class:`TrajectoryActionDecoder`'s architecture so the two modules can
    be trained independently or used as inverse/forward consistency losses.

    The default ``action_dim=12`` assumes the gripper dimension has been removed from the MSHAB
    action chunk.
    """

    def __init__(
        self,
        *,
        qpos_dim: int = 12,
        action_dim: int = 12,
        trajectory_dim: int = 7,
        horizon: int = 32,
        encoder_type: str = "transformer",
        mlp_hidden_dims: tuple[int, ...] = (512, 512),
        transformer_hidden_dim: int = 256,
        transformer_num_layers: int = 4,
        transformer_dim_feedforward: int = 1024,
        transformer_n_head: int = 8,
        use_positional_encoding: bool = True,
        latent_dim: int = 256,
        concat_most_recent_obs: bool = False,
    ):
        super().__init__()
        self.qpos_dim = qpos_dim
        self.action_dim = action_dim
        self.trajectory_dim = trajectory_dim
        self.horizon = horizon
        self.model = ConditionedChunkModel(
            qpos_dim=qpos_dim,
            input_dim=action_dim,
            output_dim=trajectory_dim,
            horizon=horizon,
            encoder_type=encoder_type,
            mlp_hidden_dims=mlp_hidden_dims,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_n_head=transformer_n_head,
            use_positional_encoding=use_positional_encoding,
            latent_dim=latent_dim,
            concat_most_recent_obs=concat_most_recent_obs,
        )

    def forward(self, qpos: Tensor, actions: Tensor) -> Tensor:
        """Return a trajectory chunk with shape ``(B, H, trajectory_dim)``.

        ``qpos`` may be ``(qpos_dim,)`` or ``(B, qpos_dim)``.
        ``actions`` may be ``(H, action_dim)`` or ``(B, H, action_dim)``.
        """
        return self.model(qpos, actions)


class MlpTrajectoryPredictor(nn.Module):
    """Flattened MLP baseline for qpos + action chunk -> TCP trajectory.

    Use this as a minimal forward-dynamics baseline. It requires a fixed horizon.
    """

    def __init__(
        self,
        *,
        qpos_dim: int = 12,
        action_dim: int = 12,
        trajectory_dim: int = 7,
        horizon: int = 32,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        self.qpos_dim = qpos_dim
        self.action_dim = action_dim
        self.trajectory_dim = trajectory_dim
        self.horizon = horizon
        dims = [qpos_dim + horizon * action_dim, *hidden_dims, horizon * trajectory_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, qpos: Tensor, actions: Tensor) -> Tensor:
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if qpos.shape[-1] != self.qpos_dim:
            raise ValueError(f"qpos last dim must be {self.qpos_dim}; got {qpos.shape[-1]}")
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"actions must have shape (B, {self.horizon}, {self.action_dim}); got {tuple(actions.shape)}")
        x = torch.cat([qpos, actions.flatten(start_dim=1)], dim=-1)
        return self.net(x).view(qpos.shape[0], self.horizon, self.trajectory_dim)
