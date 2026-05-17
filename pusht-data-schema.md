# PushT Expert HDF5 Schema (`pusht_expert_train.h5`)

本文描述 LeWM / stable-worldmodel 使用的 PushT 专家轨迹 HDF5 **扁平表**格式。目标：在任意仿真中采集数据后，写出 **相同列名、dtype、形状语义与 episode 索引约定** 的文件，以便现有加载代码（如 `swm.data.HDF5Dataset`）可直接使用。

---

## 文件与顶层结构

- **格式**：HDF5（`.h5`）
- **布局**：**仅根组 `/`**，所有数据为 **顶层 Dataset**，**无嵌套 Group**。
- **一行一步**：全局第 `t` 行 = 某个 episode 内第 `step_idx[t]` 步；该步对应观测（图像 + 低维量）及 **在该步执行的动作** `action[t]`（约定与常见 RL 转储一致：存储 transition 时动作与当前观测对齐）。

---

## 必选 Dataset（列）

所有数组在 **第 0 维**上对齐，长度相同，记为 **`N`**（全局步数）。

| Dataset       | 形状              | dtype    | 含义 |
|---------------|-------------------|----------|------|
| `pixels`      | `(N, H, W, 3)`    | `uint8`  | RGB，**通道在最后维（NHWC）**。参考数据：`H=W=224`。 |
| `action`      | `(N, A)`          | `float32`| 连续动作。PushT 参考：**`A=2`**（平面 2 维）。 |
| `state`       | `(N, S)`          | `float32`| 任务相关低维状态。参考：**`S=7`**。 |
| `proprio`     | `(N, P)`          | `float32`| 本体感受 / 低维观测子向量。参考：**`P=4`**。 |
| `episode_idx` | `(N,)`            | `int64`  | 该行所属 episode 的全局编号，从 **0** 递增到 **`E-1`**。 |
| `step_idx`    | `(N,)`            | `int64`  | 该行在该 episode 内的步索引，**每个 episode 从 0 到 `L_e-1`**。 |
| `ep_len`      | `(E,)`            | `int32`  | 第 `e` 条轨迹的长度（步数）`L_e`。 |
| `ep_offset`   | `(E,)`            | `int64`  | 第 `e` 条轨迹在扁平表中的 **起始行索引**。 |

参考数据统计（仅供对齐量级）：`E≈18685`，`N≈2.34×10^6`，`ep_len` 约 **49–246** 步。

---

## Episode 索引不变量（实现时请自检）

设 episode 总数为 **`E`**，全局步数为 **`N`**。

1. **`sum(ep_len) == N`**
2. **`ep_offset[0] == 0`**，且 **`ep_offset[e] = sum(ep_len[0:e])`**（即前缀和，等价于 `np.concatenate([[0], np.cumsum(ep_len[:-1]])`)）。
3. 对任意 episode `e`，令 `start = ep_offset[e]`，`L = ep_len[e]`：
   - `episode_idx[start : start+L]` 全为 **`e`**
   - `step_idx[start : start+L]` 为 **`0, 1, …, L-1`**
4. **`pixels` / `action` / `state` / `proprio`** 在 `[start : start+L)` 上与上述 `episode_idx`、`step_idx` 同行对齐。

---

## `pixels` 存储说明

参考文件对 **`pixels`** 使用了 **HDF5 Blosc** 压缩滤镜（非 gzip）。写入时可选：

- **不压缩**：最易移植，`h5py` 无需插件即可读取。
- **Blosc**：体积更小；读取端需兼容 Blosc（例如 Python 侧常用 `hdf5plugin`）。

若要与参考文件完全一致，可对 `pixels` 使用 Blosc；若优先跨环境简单可读，建议 **`pixels` 不压缩**。

---

## 写入顺序建议（伪代码）

采集多条轨迹后：

1. 将每条轨迹按时间顺序追加到各 `(N, …)` 数组（`pixels`, `action`, `state`, `proprio`）。
2. 同步写入 `episode_idx`（当前 episode id）、`step_idx`（0…L-1）。
3. 拼好 `ep_len[e] = L_e`，再计算 **`ep_offset`** 为满足上节不变量的前缀和。
4. 以 **`chunks`** 适度分块（可选，利于随机读）；参考：`action` 类似 `(1000, A)`，`pixels` 类似 `(100, H, W, 3)`。

---

## 与仓库配置的对应关系

训练配置通过数据集名（不含 `.h5`）指向该文件，并声明要加载的键，例如：`pixels`, `action`, `proprio`, `state`（见 `config/train/data/pusht.yaml`）。新环境若维度 **A/S/P** 与参考一致，可直接替换轨迹内容；若维度不同，需同步修改下游模型与配置中的维度，但 **HDF5 列名与 episode 四件套语义** 建议保持不变以便复用管线。
