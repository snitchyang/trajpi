"""Convert delta-action normalization stats to approximate absolute-action stats.

This uses only marginal statistics from norm_stats.json. The converted means are exact
for action_abs = action_delta + state, while std/quantiles are approximations because
the original file does not contain state/action covariance or joint distributions.
"""

import argparse
import copy
import json
import math
from pathlib import Path


DEFAULT_STATS_DIR = Path("/data/user/wzhang834/users/vick/trace_mobile/openpi/datasets/mshab/mshab_lerobot_open_close")
DEFAULT_INPUT = DEFAULT_STATS_DIR / "norm_stats_bak.json"
DEFAULT_OUTPUT = DEFAULT_STATS_DIR / "norm_stats.json"

# 1-based user layout:
# action dims 1-7 map to state dims 1-7; action dims 9-11 map to state dims 10-12.
DEFAULT_ACTION_INDICES = tuple(range(7)) + tuple(range(8, 11))
DEFAULT_STATE_INDICES = tuple(range(7)) + tuple(range(9, 12))


def _require_key(stats, key):
    if key not in stats:
        raise KeyError(f"Missing required stat field: {key}")
    return list(stats[key])


def _convert_stats(norm_stats, action_indices=DEFAULT_ACTION_INDICES, state_indices=DEFAULT_STATE_INDICES):
    if len(action_indices) != len(state_indices):
        raise ValueError("action_indices and state_indices must have the same length.")

    converted = copy.deepcopy(norm_stats)
    stats = converted["norm_stats"]
    state_stats = stats["state"]
    action_stats = stats["actions"]

    state_mean = _require_key(state_stats, "mean")
    state_std = _require_key(state_stats, "std")
    action_mean = _require_key(action_stats, "mean")
    action_std = _require_key(action_stats, "std")

    state_q01 = state_stats.get("q01")
    state_q99 = state_stats.get("q99")
    action_q01 = action_stats.get("q01")
    action_q99 = action_stats.get("q99")

    if state_q01 is not None:
        state_q01 = list(state_q01)
    if state_q99 is not None:
        state_q99 = list(state_q99)
    if action_q01 is not None:
        action_q01 = list(action_q01)
    if action_q99 is not None:
        action_q99 = list(action_q99)

    for action_idx, state_idx in zip(action_indices, state_indices):
        action_mean[action_idx] = action_mean[action_idx] + state_mean[state_idx]
        action_std[action_idx] = math.sqrt(action_std[action_idx] ** 2 + state_std[state_idx] ** 2)

        if action_q01 is not None and state_q01 is not None:
            action_q01[action_idx] = action_q01[action_idx] + state_q01[state_idx]
        if action_q99 is not None and state_q99 is not None:
            action_q99[action_idx] = action_q99[action_idx] + state_q99[state_idx]

    action_stats["mean"] = action_mean
    action_stats["std"] = action_std
    if action_q01 is not None:
        action_stats["q01"] = action_q01
    if action_q99 is not None:
        action_stats["q99"] = action_q99

    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open() as f:
        norm_stats = json.load(f)

    converted = _convert_stats(norm_stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(converted, f, indent=2)
        f.write("\n")

    print(f"Wrote converted absolute-action norm stats to: {args.output}")


if __name__ == "__main__":
    main()
