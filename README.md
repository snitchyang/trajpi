# trajpi

基于 [openpi](https://github.com/Physical-Intelligence/openpi) 的研究分支，针对 **MSHAB (ManiSkill) 移动操作仿真器**中的 **Fetch 机器人**进行扩展。核心创新在于引入 **轨迹-动作对偶 (Trajectory-Action Duality)** 机制：模型同时预测 TCP 轨迹和动作序列，并通过独立的轨迹预测网络在推理时进行动作精炼。

## 核心特性

- **轨迹预测 (traj2actions)**：扩展 π₀.₅ 模型，使其在预测动作块的同时预测世界坐标系下的 TCP 轨迹 (xyz + quaternion + gripper, 8D)
- **推理时动作精炼 (Action-Trajectory Alignment)**：通过梯度下降使预训练的 `TrajectoryPredictor` 输出与 VLA 预测轨迹对齐，从而优化动作
- **MSHAB Fetch 机器人支持**：自定义数据管线处理 Fetch 的 13D 动作空间 (`arm7 + gripper1 + body3 + base2`)，自动积分底盘速度完成 base→world 坐标变换
- **轨迹可视化工具链**：相机投影、轨迹叠加视频、动作/轨迹分布分析等

## 硬件需求

| 模式 | 显存需求 | 示例 GPU |
|---|---|---|
| 推理 | > 8 GB | RTX 4090 |
| LoRA 微调 | > 22.5 GB | RTX 4090 |
| 全参微调 | > 70 GB | A100 (80GB) / H100 |

## 安装

```bash
git clone --recurse-submodules <repo_url>
git submodule update --init --recursive
```

### uv (推荐)

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### conda

```bash
bash scripts/install_conda.sh
# 可选：开发工具 / RLDS 支持
# bash scripts/install_conda.sh --with-dev --with-rlds
```

创建的 conda 环境名为 `trajpi`。

### PyTorch transformers 补丁

PyTorch 模型需要修改 `transformers` 库以支持 AdaRMS、精度控制和 KV cache：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

> **注意**：使用 uv 默认 hardlink 模式时此修改会永久影响 uv cache。可通过 `uv cache clean transformers` 撤销。

## 关键配置

所有配置定义在 `src/openpi/training/config.py`：

| 配置名 | 说明 |
|---|---|
| `pi05_mshab` | MSHAB Fetch 标准训练，LoRA，动作空间 13D |
| `pi05_mstraj` | **核心配置** — π₀.₅ + 轨迹预测 (`traj2actions=True`)，含推理时动作精炼 |
| `traj_predictor` | 训练轨迹预测网络 (forward dynamics: state + actions → trajectory) |
| `traj_decoder` | 训练轨迹解码网络 (inverse dynamics: state + trajectory → actions) |
| `mshab_test` | 快速测试用配置 |

## 训练流程

### 流程一：标准动作训练 (pi05_mshab)

```bash
# 1. 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_mshab

# 2. 多 GPU DDP 训练
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_mshab --exp_name <name> --batch_size 128
```

### 流程二：轨迹感知训练 (pi05_mstraj)

```bash
# 1. 先训练轨迹预测网络
uv run scripts/train_trajectory_decoders.py --task trajectory_predictor \
    --config-name traj_predictor --wandb-mode disabled

# 2. 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_mstraj

# 3. 训练带轨迹预测的 π₀.₅
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py pi05_mstraj --exp_name <name> --batch_size 128
```

### SLURM 集群提交

```bash
sbatch train_single_task_ddp.sh    # 训练
sbatch compute_norm_stats.sh       # 归一化统计
sbatch scripts/sbatch_train_trajectory_decoders.sh  # 轨迹网络
```

## 推理与评估

```bash
# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_mstraj \
    --policy.dir=checkpoints/pi05_mstraj/<exp>/<step>

# 通过轨迹预测器评估 PI 动作
uv run scripts/eval_pi_with_trajectory_predictor.py pi05_mstraj \
    --traj-ckpt-path <predictor.pt> --weights <pi_checkpoint_dir>
```

## 可视化工具

| 脚本 | 功能 |
|---|---|
| `scripts/visualize_mshab_relative_traj.py` | 将未来动作块的 TCP 轨迹投影到相机图像，生成 PNG/MP4 |
| `scripts/visualize_mshab_qpos_action_traj.py` | 通过 URDF 正运动学从 qpos + actions 可视化 Fetch TCP 轨迹 |
| `scripts/visualize_mshab_trace_traj.py` | 将 websocket 推理 trace 叠加到 MSHAB episode 视频 |
| `scripts/infer_and_visualize_traj.py` | 完整推理 + 轨迹投影渲染 MP4 |
| `scripts/analyze_mshab_action_trajectory_distribution.py` | 分析动作/轨迹分布，生成统计图 |

## 相机标定

- `mshab_fetch_head_base.json` — Fetch 头部相机在 robot base 坐标系下的内外参 (K, R, t, T_cam_base, T_base_cam)
- `mshab_fetch_head_opencv.json` — OpenCV world-frame 坐标系约定的相机参数

## 项目结构

```
src/openpi/
├── models/
│   └── pi0_config.py          # traj_dim, traj2actions, enable_action_traj_alignment 等扩展
├── models_pytorch/
│   ├── decoders/
│   │   ├── common.py              # ConditionedChunkModel 共享骨干
│   │   ├── trajectory_predictor.py # Forward dynamics: (state, actions) → trajectory
│   │   └── trajectory_action_decoder.py # Inverse dynamics: (state, trajectory) → actions
│   └── pi0_pytorch.py         # traj2actions 模式扩展
├── policies/
│   └── action_traj_alignment.py  # 推理时梯度下降动作精炼
├── transforms.py              # MshabTcpBaseToWorldActions, DropActionDimensions, MappedAbsoluteActions
└── training/
    └── config.py              # LeRobotMshabDataConfig + 所有 MSHAB 配置
scripts/
├── train_trajectory_decoders.py       # 训练轨迹预测/解码网络
├── eval_trajectory_decoders.py        # 评估轨迹网络
├── eval_pi_with_trajectory_predictor.py # 端到端评估
├── infer_and_visualize_traj.py        # 推理 + 可视化
└── ...                                 # 其他可视化与分析脚本
```

## 上游 openpi

本项目基于 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi)，保留了其全部功能：

- **模型**：π₀, π₀-FAST, π₀.₅ (JAX + PyTorch 双后端)
- **机器人平台**：ALOHA, DROID, LIBERO, UR5 等
- **特性**：LoRA 微调、远程推理、policy server、权重转换等

详见上游 [README](https://github.com/Physical-Intelligence/openpi)。
