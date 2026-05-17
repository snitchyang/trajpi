#!/bin/bash
#SBATCH --job-name=mshab_test
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48

# 与参考一致使用在线 wandb 时保持注释；离线调试可取消下一行注释
export WANDB_MODE=offline

module load ffmpeg/5.1.7

source /data/user/zyang736/miniconda3/etc/profile.d/conda.sh
conda activate /data/user/zyang736/miniconda3/envs/trajpi

cd /data/user/wzhang834/users/vick/trace_mobile/openpi

# batch_size 为全局 batch；与 #SBATCH --gres 一致：4 卡时每卡 32 -> --batch_size=128；8 卡 -> 256
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi05_mstraj \
    --exp_name=mshab_traj_table_open_close_delta_0501_chunk32 \
    --project_name=mshab_traj_table_open_close \
    --batch_size=128  \
    --num_workers=8 \
    --num_train_steps=5000 \
    --save_interval=2500 \
    # --resume \
# 仅在已有同名实验 checkpoint 目录、且要继续训练时追加: --resume
