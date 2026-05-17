#!/usr/bin/env python3
"""
Evaluate a Pi0 / Pi05 PyTorch checkpoint on the training dataset.

Uses the same `create_data_loader` pipeline as `scripts/train_pytorch.py`, runs
`PI0Pytorch.sample_actions` (flow matching from noise to x0), and reports mean
squared error between predicted actions and ground-truth actions.

Example:
  cd /path/to/openpi
  python scripts/eval_pytorch.py pi05_mshab \\
    --exp-name=mshab \\
    --checkpoint-step=5000

  # Latest checkpoint, cap batches for a quick smoke test
  python scripts/eval_pytorch.py pi05_mshab --exp-name=mshab --max-batches=50

  # Per-sample lines go to a TSV (default: one line per sample to the logger)
  python scripts/eval_pytorch.py pi05_mshab --exp-name=mshab --sample-log-file=eval_x0_mse.tsv

  # Custom weights (file or directory that contains model.safetensors) — no --exp-name needed
  python scripts/eval_pytorch.py pi05_mshab \\
    --weights=/path/to/5000/model.safetensors

  # Custom checkpoint root (parent of numeric step dirs), still use --checkpoint-step or latest
  python scripts/eval_pytorch.py pi05_mshab \\
    --checkpoint-root=/path/to/outputs/checkpoints/pi05_mshab/my_run
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys

import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.eval_pytorch as eval_lib


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config_name", help="Train config name, e.g. pi05_mshab (controls data + model layout)")
    p.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name under checkpoint_base_dir/name/<exp> (default layout). "
        "Not required if --weights or --checkpoint-root is set.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Override checkpoint root directory (must contain numeric step subdirs with model.safetensors).",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Load this model.safetensors file, or a directory that contains model.safetensors. "
        "Overrides --checkpoint-root / default exp checkpoint layout.",
    )
    p.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step folder under checkpoint root; default: latest (ignored if --weights is set)",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override global batch size (default: config)")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    p.add_argument("--num-flow-steps", type=int, default=10, help="Euler steps in sample_actions")
    p.add_argument("--max-batches", type=int, default=None, help="Limit batches for faster eval")
    p.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    p.add_argument(
        "--no-per-batch",
        action="store_true",
        help="Disable per-batch log lines (batch index, batch MSE, running MSE)",
    )
    p.add_argument(
        "--no-per-sample",
        action="store_true",
        help="Disable one line per dataset sample (per-sample mean MSE over horizon x action dim)",
    )
    p.add_argument(
        "--sample-log-file",
        type=str,
        default=None,
        help="If set, write per-sample TSV here instead of logging each sample",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use model's torch.compile on sample_actions (slow first batch: Inductor autotune)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    init_logging()
    args = parse_args(argv)

    if args.exp_name is None and args.checkpoint_root is None and args.weights is None:
        logging.error("Provide --exp-name (default layout), or --checkpoint-root, or --weights")
        sys.exit(2)

    base = _config.get_config(args.config_name)
    config = dataclasses.replace(
        base,
        exp_name=args.exp_name if args.exp_name is not None else base.exp_name,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else base.num_workers,
        resume=False,
        wandb_enabled=False,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available; falling back to CPU")
        device = torch.device("cpu")

    model = eval_lib.build_pi0_pytorch_for_eval(config, device, use_torch_compile=args.torch_compile)

    if args.weights:
        wpath = eval_lib.load_weights_file(model, pathlib.Path(args.weights), device)
        used_step: int | str = f"custom:{wpath}"
        logging.info("Evaluating weights file %s", wpath)
    elif args.checkpoint_root:
        ckpt_root = pathlib.Path(args.checkpoint_root).expanduser().resolve()
        logging.info("Checkpoint root (override): %s", ckpt_root)
        used_step = eval_lib.load_eval_checkpoint(
            model,
            ckpt_root,
            step=args.checkpoint_step,
            device=device,
        )
        logging.info("Evaluating checkpoint step %s", used_step)
    else:
        ckpt_root = config.checkpoint_dir
        logging.info("Checkpoint root: %s", ckpt_root)
        used_step = eval_lib.load_eval_checkpoint(
            model,
            ckpt_root,
            step=args.checkpoint_step,
            device=device,
        )
        logging.info("Evaluating checkpoint step %s", used_step)

    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)

    sample_fh = None
    try:
        if args.sample_log_file:
            sample_fh = open(args.sample_log_file, "w", encoding="utf-8")

        metrics = eval_lib.eval_x0_action_mse(
            model,
            loader,
            device,
            num_flow_steps=args.num_flow_steps,
            max_batches=args.max_batches,
            log_per_batch=not args.no_per_batch,
            log_per_sample=not args.no_per_sample,
            sample_log_file=sample_fh,
        )
    finally:
        if sample_fh is not None:
            sample_fh.close()

    logging.info(
        "x0_action_mse=%.8f over %d batches, %d samples (%d scalar elements)",
        metrics["x0_action_mse"],
        int(metrics["num_batches"]),
        int(metrics.get("num_samples", 0)),
        int(metrics["num_elements"]),
    )
    print(
        f"x0_action_mse={metrics['x0_action_mse']:.8f} "
        f"batches={int(metrics['num_batches'])} samples={int(metrics.get('num_samples', 0))}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
