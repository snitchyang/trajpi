from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.decoders.common import ConditionedChunkModel


class TrajectoryActionDecoder(nn.Module):
    """Decode an executable action chunk from current qpos and a desired TCP trajectory.

    This mirrors the umi-on-legs sequence-encoder style: qpos is repeated across the horizon and
    concatenated with each trajectory waypoint, a history encoder compresses the full sequence into
    one latent, and an MLP head predicts the full action chunk.

    Args:
        qpos_dim: Dimension of robot proprioceptive qpos, e.g. 12 for MSHAB Fetch without base DoFs.
        trajectory_dim: Dimension of each trajectory waypoint, e.g. 3 for xyz or 7 for xyz+quat.
        action_dim: Dimension of each action. Defaults to 12 for MSHAB actions without gripper.
        horizon: Fixed chunk length.
    """

    def __init__(
        self,
        *,
        qpos_dim: int = 12,
        trajectory_dim: int = 7,
        action_dim: int = 12,
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
        self.trajectory_dim = trajectory_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.model = ConditionedChunkModel(
            qpos_dim=qpos_dim,
            input_dim=trajectory_dim,
            output_dim=action_dim,
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

    def forward(self, qpos: Tensor, trajectory: Tensor) -> Tensor:
        """Return an action chunk with shape ``(B, H, action_dim)``.

        ``qpos`` may be ``(qpos_dim,)`` or ``(B, qpos_dim)``.
        ``trajectory`` may be ``(H, trajectory_dim)`` or ``(B, H, trajectory_dim)``.
        """
        return self.model(qpos, trajectory)


class MlpTrajectoryActionDecoder(nn.Module):
    """Flattened MLP baseline for qpos + trajectory -> action chunk.

    This is useful as a strong sanity-check baseline before using the Transformer decoder.
    It requires a fixed horizon.
    """

    def __init__(
        self,
        *,
        qpos_dim: int = 12,
        trajectory_dim: int = 7,
        action_dim: int = 12,
        horizon: int = 32,
        hidden_dims: tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        self.qpos_dim = qpos_dim
        self.trajectory_dim = trajectory_dim
        self.action_dim = action_dim
        self.horizon = horizon
        dims = [qpos_dim + horizon * trajectory_dim, *hidden_dims, horizon * action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, qpos: Tensor, trajectory: Tensor) -> Tensor:
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)
        if qpos.shape[-1] != self.qpos_dim:
            raise ValueError(f"qpos last dim must be {self.qpos_dim}; got {qpos.shape[-1]}")
        if trajectory.shape[1:] != (self.horizon, self.trajectory_dim):
            raise ValueError(
                f"trajectory must have shape (B, {self.horizon}, {self.trajectory_dim}); got {tuple(trajectory.shape)}"
            )
        x = torch.cat([qpos, trajectory.flatten(start_dim=1)], dim=-1)
        return self.net(x).view(qpos.shape[0], self.horizon, self.action_dim)
