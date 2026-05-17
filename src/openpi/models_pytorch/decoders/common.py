from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor
from torch import nn


def ensure_batched_vector(x: Tensor, dim: int, name: str) -> Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[-1] != dim:
        raise ValueError(f"{name} must have shape ({dim},) or (B, {dim}); got {tuple(x.shape)}")
    return x


def ensure_batched_sequence(x: Tensor, dim: int, name: str) -> Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != dim:
        raise ValueError(f"{name} must have shape (H, {dim}) or (B, H, {dim}); got {tuple(x.shape)}")
    return x


class HistoryEncoder(nn.Module):
    """Direct port of umi-on-legs' history encoder interface."""

    def __init__(
        self,
        obs_dim: int,
        history_len: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.net = self.setup_net(obs_dim, history_len)

    @abstractmethod
    def setup_net(self, obs_dim: int, history_len: int) -> nn.Module:
        pass

    def forward(self, obs: Tensor) -> Tensor:
        if len(obs.shape) == 2 and obs.shape[1] == (self.history_len * self.obs_dim):
            obs = obs.view(-1, self.history_len, self.obs_dim)
        elif len(obs.shape) == 1 and obs.shape[0] == (self.history_len * self.obs_dim):
            obs = obs.view(1, self.history_len, self.obs_dim)
        if obs.shape[1] != self.history_len or obs.shape[2] != self.obs_dim:
            raise ValueError(f"obs must be (B, {self.history_len}, {self.obs_dim}); got {tuple(obs.shape)}")
        return self.net(obs.permute(0, 2, 1))


class MLPHistoryEncoder(HistoryEncoder):
    def __init__(self, obs_dim: int, history_len: int, hidden_dims: tuple[int, ...]):
        self.hidden_dims = hidden_dims
        super().__init__(obs_dim, history_len)

    def setup_net(self, obs_dim: int, history_len: int) -> nn.Module:
        all_dims = [obs_dim * history_len, *self.hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs.view(-1, self.history_len * self.obs_dim))


class TransformerHistoryEncoder(HistoryEncoder):
    def __init__(
        self,
        obs_dim: int,
        history_len: int,
        hidden_dim: int,
        num_layers: int,
        dim_feedforward: int,
        n_head: int,
        use_positional_encoding: bool,
        output_latent_dim: int,
        concat_most_recent_obs: bool,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        super().__init__(obs_dim, history_len)
        self.in_proj = nn.Linear(obs_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_latent_dim)
        self.concat_most_recent_obs = concat_most_recent_obs
        if use_positional_encoding:
            self.pos_enc = nn.Parameter(torch.randn(history_len + 1, hidden_dim))
        else:
            self.pos_enc = None

    def setup_net(self, _obs_dim: int, _history_len: int) -> nn.Module:
        return nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.n_head,
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
            ),
            num_layers=self.num_layers,
        )

    def forward(self, obs: Tensor) -> Tensor:
        if len(obs.shape) == 2 and obs.shape[1] == (self.history_len * self.obs_dim):
            obs = obs.view(-1, self.history_len, self.obs_dim)
        elif len(obs.shape) == 1 and obs.shape[0] == (self.history_len * self.obs_dim):
            obs = obs.view(1, self.history_len, self.obs_dim)
        if obs.shape[1] != self.history_len or obs.shape[2] != self.obs_dim:
            raise ValueError(f"obs must be (B, {self.history_len}, {self.obs_dim}); got {tuple(obs.shape)}")
        obs_embed = self.in_proj(obs)
        if self.pos_enc is not None:
            obs_embed = obs_embed + self.pos_enc[: obs_embed.shape[1]]
        cls_token = torch.zeros(obs.shape[0], 1, self.hidden_dim, device=obs.device, dtype=obs.dtype)
        embs = self.net(torch.cat([cls_token, obs_embed], dim=1))
        cls_emb = embs[:, 0]
        out_emb = self.out_proj(cls_emb)
        if self.concat_most_recent_obs:
            out_emb = torch.cat([out_emb, obs[:, -1]], dim=1)
        return out_emb


class ConditionedChunkModel(nn.Module):
    """UMI-style history encoder plus MLP head for full-chunk prediction."""

    def __init__(
        self,
        *,
        qpos_dim: int,
        input_dim: int,
        output_dim: int,
        horizon: int,
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.obs_dim = qpos_dim + input_dim
        self.encoder_type = encoder_type

        if encoder_type == "mlp":
            self.encoder = MLPHistoryEncoder(
                obs_dim=self.obs_dim,
                history_len=horizon,
                hidden_dims=(*mlp_hidden_dims, latent_dim),
            )
            head_input_dim = latent_dim
        elif encoder_type == "transformer":
            self.encoder = TransformerHistoryEncoder(
                obs_dim=self.obs_dim,
                history_len=horizon,
                hidden_dim=transformer_hidden_dim,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                n_head=transformer_n_head,
                use_positional_encoding=use_positional_encoding,
                output_latent_dim=latent_dim,
                concat_most_recent_obs=concat_most_recent_obs,
            )
            head_input_dim = latent_dim + (self.obs_dim if concat_most_recent_obs else 0)
        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type!r}; expected 'transformer' or 'mlp'")

        head_dims = [head_input_dim, *mlp_hidden_dims, horizon * output_dim]
        head_layers: list[nn.Module] = []
        for i in range(len(head_dims) - 1):
            head_layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            if i < len(head_dims) - 2:
                head_layers.append(nn.ReLU())
        self.head = nn.Sequential(*head_layers)

    def forward(self, qpos: Tensor, sequence: Tensor) -> Tensor:
        qpos = ensure_batched_vector(qpos, self.qpos_dim, "qpos")
        sequence = ensure_batched_sequence(sequence, self.input_dim, "sequence")
        if qpos.shape[0] != sequence.shape[0]:
            raise ValueError(f"qpos batch {qpos.shape[0]} does not match sequence batch {sequence.shape[0]}")
        if sequence.shape[1] != self.horizon:
            raise ValueError(f"Expected sequence horizon {self.horizon}; got {sequence.shape[1]}")
        qpos_seq = qpos[:, None, :].expand(-1, self.horizon, -1)
        obs = torch.cat([qpos_seq, sequence], dim=-1)
        latent = self.encoder(obs)
        return self.head(latent).view(qpos.shape[0], self.horizon, self.output_dim)
