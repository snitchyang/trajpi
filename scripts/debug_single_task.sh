#!/bin/bash
#SBATCH --job-name=mshab_test
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48

# 与参考一致使用在线 wandb 时保持注释；离线调试可取消下一行注释
export WANDB_MODE=offline

# 单进程调试只用一张卡；若 Slurm 申请了多卡，其余卡空闲
export CUDA_VISIBLE_DEVICES=0

module load ffmpeg/5.1.7

source /data/user/zyang736/miniconda3/etc/profile.d/conda.sh
conda activate /data/user/zyang736/miniconda3/envs/trajpi

cd /data/user/wzhang834/users/vick/trace_mobile/openpi

# debugpy：进程在 5678 端口等待调试器 attach 后才开始跑训练（VS Code / Cursor「附加到进程」或 debugpy connect）
# 集群上从本机连：ssh -L 5678:计算节点:5678 user@login 后，在本机用「Python: Remote Attach」连 localhost:5678
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client scripts/train_pytorch.py pi05_mstraj \
    --exp_name=mshab_action_debug \
    --project_name=mshab_action_table_pp_debug \
    --batch_size=2 \
    --num_workers=1 \
    --num_train_steps=30000 \
    --save_interval=2500 \
# 仅在已有同名实验 checkpoint 目录、且要继续训练时追加: --resume
