#!/usr/bin/env bash
# 用训练好的 trajectory predictor 评估 PI 模型：观测 -> PI 动作块 -> 预测轨迹，与 GT 轨迹对比。
# 下方配置固定；更多选项见 scripts/eval_pi_with_trajectory_predictor.py，可通过本脚本追加参数。
module load ffmpeg/5.1.7
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_NAME="pi05_mstraj"
TRAJ_CKPT="/data/user/wzhang834/users/vick/trace_mobile/openpi/outputs/trajectory_predictor/20260503_000019/model_30.pt"
# PI 权重：目录内需有 model.safetensors（与 eval_pytorch.load_weights_file 一致）
PI_CKPT_PATH="/data/user/wzhang834/users/vick/trace_mobile/openpi/outputs/checkpoints/pi05_mstraj/mshab_traj_table_open_close_0426/20000"

# 若需在特定 conda 环境中运行，取消注释并修改路径与环境名：
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
固定配置: CONFIG_NAME=$CONFIG_NAME
  --traj-ckpt-path=$TRAJ_CKPT（相对仓库根目录）
  --weights=$PI_CKPT_PATH（PI checkpoint 目录）

用法: $(basename "$0") [额外参数...]

额外参数会追加在固定参数之后；若再传 --weights / --checkpoint-root 等会覆盖或与 argparse 规则冲突时请避免重复。
例如:
  $(basename "$0") --max-batches 20
  $(basename "$0") --device cuda:0

完整参数说明请运行:
  python3 $SCRIPT_DIR/eval_pi_with_trajectory_predictor.py --help
EOF
  exit 0
fi

exec python3 "$SCRIPT_DIR/eval_pi_with_trajectory_predictor.py" \
  "$CONFIG_NAME" \
  --traj-ckpt-path "$TRAJ_CKPT" \
  --weights "$PI_CKPT_PATH" \
  "$@"
