#!/bin/bash
#SBATCH --job-name=mshab_test
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCh --mem=100G

# 与参考一致使用在线 wandb 时保持注释；离线调试可取消下一行注释
export WANDB_MODE=offline

module load ffmpeg/5.1.7

source /data/user/zyang736/miniconda3/etc/profile.d/conda.sh
conda activate /data/user/zyang736/miniconda3/envs/trajpi

cd /data/user/wzhang834/users/vick/trace_mobile/openpi

python scripts/compute_norm_stats.py --config_name=pi05_mstraj