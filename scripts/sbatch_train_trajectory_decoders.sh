#!/usr/bin/env bash
#SBATCH --job-name=traj-decoder
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=outputs/slurm_logs/%x-%j.out
#SBATCH --error=outputs/slurm_logs/%x-%j.err

module load ffmpeg/5.1.7

source /data/user/zyang736/miniconda3/etc/profile.d/conda.sh

cd /data/user/wzhang834/users/vick/trace_mobile/openpi
mkdir -p outputs/slurm_logs
conda activate trajpi
python scripts/train_trajectory_decoders.py \
  --task trajectory_predictor\
  --config-name traj_predictor \
  --wandb-mode disabled \
  "$@"
