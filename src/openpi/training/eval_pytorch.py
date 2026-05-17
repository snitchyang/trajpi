"""PyTorch evaluation helpers for Pi0 / Pi05 (flow matching → x0 / action MSE)."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any, TextIO

import jax
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config


def get_latest_checkpoint_step(checkpoint_dir: pathlib.Path) -> int | None:
    steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(steps) if steps else None


def build_pi0_pytorch_for_eval(
    config: _config.TrainConfig,
    device: torch.device,
    *,
    use_torch_compile: bool = False,
) -> pi0_pytorch.PI0Pytorch:
    """Match `scripts/train_pytorch.py` model construction (no DDP, no checkpointing).

    By default ``use_torch_compile=False`` so eval does not wrap ``sample_actions`` in
    ``torch.compile``. The repo default ``pytorch_compile_mode="max-autotune"`` triggers
    long Triton autotune on the first batch, so logs (first sample / batch) appear only
    after that finishes.
    """
    compile_mode = None
    if use_torch_compile and isinstance(config.model, pi0_config.Pi0Config):
        compile_mode = config.model.pytorch_compile_mode

    if not isinstance(config.model, pi0_config.Pi0Config):
        model_cfg = pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            traj_dim=getattr(config.model, "traj_dim", 8),
            traj2actions=getattr(config.model, "traj2actions", False),
            pytorch_compile_mode=compile_mode,
        )
    else:
        model_cfg = dataclasses.replace(
            config.model,
            dtype=config.pytorch_training_precision,
            pytorch_compile_mode=compile_mode,
        )

    model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.eval()
    return model


def load_weights_file(model: pi0_pytorch.PI0Pytorch, weights_path: pathlib.Path, device: torch.device) -> pathlib.Path:
    """Load a single ``model.safetensors`` from a file path, or from ``<dir>/model.safetensors`` if a directory is given."""
    p = weights_path.expanduser().resolve()
    if p.is_dir():
        p = p / "model.safetensors"
    if not p.is_file():
        raise FileNotFoundError(f"Expected a model.safetensors file or a directory containing it: {weights_path}")
    safetensors.torch.load_model(model, str(p), device=str(device))
    logging.info("Loaded eval weights from %s", p)
    return p


def load_eval_checkpoint(
    model: pi0_pytorch.PI0Pytorch,
    checkpoint_dir: pathlib.Path,
    *,
    step: int | None,
    device: torch.device,
) -> int:
    """Load ``model.safetensors`` from ``checkpoint_dir / <step>``. Returns the step used."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if step is None:
        latest = get_latest_checkpoint_step(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"No step subdirectories found under {checkpoint_dir}")
        step = latest

    ckpt_dir = checkpoint_dir / str(step)
    weights = ckpt_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing model weights: {weights}")

    safetensors.torch.load_model(model, str(weights), device=str(device))
    logging.info("Loaded eval weights from %s", weights)
    return step


@torch.no_grad()
def eval_x0_action_mse(
    model: pi0_pytorch.PI0Pytorch,
    loader: Any,
    device: torch.device,
    *,
    num_flow_steps: int = 10,
    max_batches: int | None = None,
    log_per_batch: bool = True,
    log_per_sample: bool = False,
    sample_log_file: TextIO | None = None,
    sample_log_header: bool = True,
) -> dict[str, float]:
    """Run `sample_actions` (noise → x0) and compute global mean MSE against dataset actions.

    This matches the training-time flow path: the model predicts velocity `v_t`;
    integrating with `num_flow_steps` Euler steps from Gaussian noise yields an estimate
    of clean actions `x0`, compared here to ground-truth `actions` from the loader.

    Args:
        log_per_batch: Log one line per batch (batch index, batch mean MSE, running global MSE).
        log_per_sample: Log one line per sample in each batch (global index, per-sample mean MSE).
        sample_log_file: If set, per-sample lines are written here only (no ``logging`` spam).
            If ``None`` and ``log_per_sample`` is true, each sample is logged with ``logging.info``.
        sample_log_header: If True and ``sample_log_file`` is set, write a CSV header first.
    """
    model.eval()
    total_se = 0.0
    total_el = 0
    n_batches = 0
    global_sample = 0

    out: TextIO | None = sample_log_file
    wrote_header = False

    def _write_sample_line(idx: int, mse_val: float) -> None:
        line = f"{idx}\t{mse_val:.8f}\n"
        if out is not None:
            nonlocal wrote_header
            if sample_log_header and not wrote_header:
                out.write("sample_index\tx0_action_mse_mean_over_horizon_dims\n")
                wrote_header = True
            out.write(line)
            out.flush()
        else:
            logging.info("sample=%d x0_mse=%.8f", idx, mse_val)

    try:
        loader_len = len(loader)  # type: ignore[arg-type]
    except TypeError:
        loader_len = None
    if max_batches is not None:
        pbar_total = min(max_batches, loader_len) if loader_len is not None else max_batches
    else:
        pbar_total = loader_len
    for i, (observation, actions) in enumerate(tqdm(loader, desc="eval", total=pbar_total)):
        if max_batches is not None and i >= max_batches:
            break

        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(device=device, dtype=torch.float32)

        pred = model.sample_actions(device, observation, num_steps=num_flow_steps)

        se = F.mse_loss(pred, actions, reduction="none")
        total_se += se.sum().item()
        total_el += se.numel()
        n_batches += 1

        batch_mse = se.mean().item()
        running_mse = total_se / max(total_el, 1)
        bsize = int(actions.shape[0])

        if log_per_batch:
            logging.info(
                "batch=%d size=%d batch_mse=%.8f running_mse=%.8f",
                i,
                bsize,
                batch_mse,
                running_mse,
            )

        if log_per_sample:
            # Mean MSE over action_horizon and action_dim for each batch element
            reduce_dims = tuple(range(1, se.ndim))
            per_sample = se.mean(dim=reduce_dims)
            for b in range(bsize):
                mse_b = float(per_sample[b].item())
                idx = global_sample + b
                _write_sample_line(idx, mse_b)

        global_sample += bsize

    mean_mse = total_se / max(total_el, 1)
    return {
        "x0_action_mse": mean_mse,
        "num_batches": float(n_batches),
        "num_elements": float(total_el),
        "num_samples": float(global_sample),
    }
