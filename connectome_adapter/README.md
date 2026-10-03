# 通用真实连接图适配器

此包把已下载的 MaleCNS v1.0 全图和 DOOMFLY v6 神经模型接到通用 RGB 场景，不启动 Doom 游戏，也不产生模拟的神经活动数据。客户端和神经计算使用独立进程，避免主平台与 Python 3.11 神经依赖冲突。

实际输入路径为：RGB PNG → 上游近似视网膜映射 → 166,700 个 LIF 神经元、25,582,938 条聚合有向边 → 固定 BCI 神经读出。`native_action` 的 `turn / forward / attack` 是上游人工校准的控制信号，每个场景必须显式定义如何把这些值映射到自己的动作，不能称为跨场景天然通用的神经动作。

## 客户端接口

```python
from connectome_adapter import ConnectomeClient

with ConnectomeClient(learning=True, timeout=120) as brain:
    # RGB uint8 NumPy 数组、RGB PIL Image 或已经编码的 RGB PNG bytes。
    result = brain.step(rgb, duration_ms=28.6,
                        reward_event={"aversive": collision_event},
                        neural_drive={"retina_left": 0.2, "retina_right": 0.3,
                                      "sugar": 0.0})
    native = result["native_action"]
    # 场景自行实现动作映射，不要直接假设 attack 适用于任意环境。
    scenario_action = map_for_this_scenario(native)

    brain.save("runs/my_connectome/checkpoints")
    brain.reset(preserve_weights=True)
    brain.restore("runs/my_connectome/checkpoints")
    brain.set_learning(False)  # 评估时冻结权重。
```

客户端初始化可配置 `python`、`log_dir`、`checkpoint` 和单请求超时 `timeout`。默认解释器为 `/root/.cache/cyberfly/doom-env/bin/python`。仅发送 PNG bytes 时客户端只需 Python 标准库；数组/PIL 输入编码需要 Pillow。

`step` 返回：

- `input_rgb_sha256`：脑端实际解码并送入仿真的 RGB uint8 原始字节摘要；配合 `input_step`、真实神经时间与实际电流摘要校验同一闭环事件，避免误用上一帧结果。
- `neural`：真实神经元与边数、神经时间、此步及累计放电数、放电向量 SHA256、计算时间。
- `native_action`、`readouts`：真实放电驱动的上游固定解码结果和读出神经元。
- `learning`：是否启用、可塑边数、改变边数、效能统计、模型记忆 SHA256。
- `stimulus`：明确外部事件次数、实际已交付刺激时长、待交付时长和来源标记。
- `neural_drive`：本次直接输入的归一化幅度、附加电流、三个真实神经元组的放电数、原生计算器最后一段的总驱动电流、累计刺激剂量。

## 直接连接语言模型或可训练输入策略

除了 RGB，`step(..., neural_drive=...)` 接受三个可独立连续控制的神经输入端口。调用方可以由语言模型产生经过类型和范围校验的输入，也可以用 PPO 训练输入策略。适配器把这些数字转换为真实已识别神经元的附加电流，再运行完整连接图；没有直接设置动作或放电数量。下述通用接口另外允许刺激已存在的运动神经元，这属于明确的工程干预。

| 端口 | 真实目标 | 输入范围及附加电流 |
| --- | --- | --- |
| `retina_left` | 上游已映射 R1–R6，MaleCNS `rootSide=L` | 0–1 → 0–10 mV 等效电流 |
| `retina_right` | 上游已映射 R1–R6，MaleCNS `rootSide=R` | 0–1 → 0–10 mV 等效电流 |
| `sugar` | 上游 `brain.sugar`，逐个核验为 LB3c | 0–1 → 0–30 mV 等效电流 |

`brain.describe()['input_ports']` 包含每个端口的全部 MaleCNS 神经元 ID、数量、注释来源、范围和生物学限制。未知端口、布尔幅度、NaN、越界值都会拒绝。三个端口只作用于当前请求的神经时间区间，下一请求未给出的端口为零。另有下述覆盖全图的通用实验电流接口。

这是额外的神经电流输入。它与 RGB 视网膜驱动、R8 输入和 PPL101 厌恶脉冲相加，幅度是人为选择的工程参数。糖输入只表示向同源推断的 LB3c 感觉细胞注入电流，不能据此宣称形成糖奖赏、正强化或语言理解。语言模型自身也没有变成果蝇神经元；它是可以接入这些端口的外部输入来源。

## 全部神经元读取与写入

`group_catalog()` / `groups()` 返回确定性的完整目录。当前数据有 **23,163 个细群**（`superclass / type / somaSide / rootSide`）和 **47 个宏群**（`superclass / rootSide`），两套分区各自覆盖全部 **166,700 个神经元**。缺少注释的细胞也保留在 `unclassified` 群中。`groups` 和 `macro_groups` 各按自己的 `index` 排列，含 `group_id`、`count` 和注释字段；对应 `partition_sha256` / `macro_partition_sha256` 可用于策略空间身份校验。离线目录为 `artifacts/full_brain/catalog.json`，可运行 `python -m connectome_adapter.full_state` 重建。

```python
catalog = brain.groups()
macro = catalog["macro_groups"][0]
result = brain.step(
    rgb, duration_ms=28.6,
    general_stimulation=[{"group_id": macro["group_id"], "current_mv": 2.0}],
    include_group_stats=True,
)
all_groups = result["group_state"]  # 默认 47 宏群，完整覆盖全脑。
fine_groups = brain.group_state(level="fine")  # 23,163 细群。
arrays, descriptor = brain.read_arrays()
assert len(arrays["ids"]) == len(arrays["v"]) == 166700

# 持久导出全部神经状态及全部边；显式目录中的文件不会自动删除。
export = brain.read_state("artifacts/my_brain_export", include_graph=True)
details = brain.neuron_details([str(arrays["ids"][0])])
```

`general_stimulation` 每次最多 16 项，可使用以下任一形式。所有 ID 必须真实存在；宏群与细群均可定点选择。电流仅持续本次请求，下一次必须再次给出。

```python
{"group_id": "实际 g_... 或 m_... ID", "current_mv": -5.0}
{"neuron_ids": ["实际原始节点 ID"], "current_mv": 5.0}
{"group_currents_mv": [0.0] * catalog["group_count"]}
{"macro_group_currents_mv": [0.0] * catalog["macro_group_count"]}
```

两个 dense 形式分别按细群或宏群的 `index` 顺序填写，长度必须精确匹配目录；一项就能对全脑每群设置不同电流。额外电流的工程范围是 **−30 至 +30 mV 模型等效电流**，允许抑制或兴奋。通用电流与三个旧端口按神经元相加，总额外电流越界时拒绝；PPL 脉冲独立相加，保持精确时序。没有增加神经元、删边、改连接拓扑或直接写入放电结果。请求返回目标数量、电流范围及向量摘要。

此接口也允许刺激真实输出神经元，因此单看动作或任务分数不能证明复杂连接拓扑发挥了作用；如要检验全脑计算的独特贡献，需要限制为感觉输入，并另做消融或普通策略对照。任意群电流是实验干预，不能凭细胞名称把语言概念声称为天然生物语义。

`group_state()` 返回真实的 `group_ids / counts / spikes / cumulative_spikes / voltage_mean / voltage_min / voltage_max` 对齐数组，以及神经时间、分区摘要与累计计数完整性。宏群反馈已经覆盖全脑；只把相关细群另外交给语言模型时，应明确模型读取的是全脑聚合反馈。

`read_state()` / `snapshot()` 返回 NPZ 描述符，避免将 166,700 项数组编码成 JSON。默认文件名为 `artifacts/full_brain/state-<uuid>.npz`，完整写入后原子更名，`temporary: true` 表示调用方读取后应删除。`read_arrays()` 返回 NumPy 数组并负责删除默认临时文件。指定输出目录则为持久导出。

| NPZ 字段 | 真实数据 |
| --- | --- |
| `ids` | 全部原始节点 ID，int64 |
| `v` | 每个神经元当前膜电位，float32，模型 mV |
| `counts` / `cumulative_counts` | 上次完整请求放电数 / 适配器累计放电数 |
| `drive` | 最后计算分段的实际总驱动，包括 RGB、tonic、额外电流及该分段的 PPL |
| `group_index` / `macro_group_index` | 每个节点在两套完整分区中的索引 |
| `soma_xyz` / `soma_valid` | 原始 `somaLocation` 坐标 / 是否存在 |
| `ptr` / `post` / `weight` | `include_graph=True` 时，完整 25,582,938 条聚合边的出边 CSR 与当前模型效能 |

CSR 中 `ptr[i]:ptr[i+1]` 是 `ids[i]` 的全部出边，`post` 是目标节点数组下标，`weight` 与每条边对齐，表示当前带符号模型效能，不等于解剖学接触数。描述符还提供原始图和 `edges.arrow` 来源。`neuron_details()` 每次接受最多 256 个原始 ID，返回注释、真实当前状态、出边总数和明确截断为前 32 条的出边样本；完整边始终可以从 CSR 导出读取。

139,662 个节点具有原始 soma 坐标，27,038 个缺少，缺失值保持 NaN。坐标单位尚未独立校准；任何替代网络布局必须明确标记，不能伪称为解剖位置。

## 强化事件的准确语义

强化事件仅接受 `reward_event={"aversive": bool}`。不接受任意数字奖励，不会把任何负数或语言模型文字偷偷转换为“真实损伤”。当场景明确报告厌恶事件时，适配器在当前请求神经区间开始，向两个已识别 PPL101 节点施加 +4 mV 等效电流，持续 2,000 个 0.1 ms 子步，即 200 ms；连续事件延长窗口。适配器保留上游 `DamageTraining.step()` 的精确脉冲分段规则，在每段将直接端口和 PPL 刺激合并后调用原生 `rgb_step()`，**不调用带虚构血量的 `observe()`**。

所有这些事件标记为 `engineered_external_event`。脉冲日志记录计划、每个神经区间实际交付的步数、输入图像 SHA256、学习开关和重置。外部环境事件与游戏里的真实血量损失不是同一个字段。

`reset(preserve_weights=True)` 清除快速神经状态和解码滤波器，保留学习的权重与记忆效能状态，取消尚未交付的脉冲并记录取消量。指定 `False` 才会清空已学的可塑性记忆。`set_learning(False)` 冻结权重，但仍可呈现输入和刺激，以便做冻结对照。

## 存档与跨场景转移

`save(path)` 使用新代次目录及原子 `latest.json` 指针保存完整 `brain.npz`、解码滤波状态、外部脉冲状态和各文件 SHA256。schema 3 保留 schema 2 的端口 ID/电流尺度签名、最近输入和累计刺激剂量，并增加全节点累计放电计数、分区签名与最近通用电流请求。`restore(path)` 支持该指针目录或具体代次目录，核验文件摘要以及上游脑模型/连接图配置签名，也兼容旧 schema 1 / 2。旧档案没有逐节点累计放电历史时，返回 `cumulative_counts_complete: false`，不会编造历史；重置后从零完整计数。恢复的最近输入用于审计，不会自动保持到下一请求。读取存档只用于可信本地文件。

也支持读取原 DOOMFLY `TrainingCheckpoints` 的 `checkpoints` 指针目录或某个代次。此时保留神经与学权状态，取消原场景待交付的刺激，不把之前的真实 Doom 伤害统计重新命名为本场景的人工事件，并保留当前客户端的学习开关。适配器原生存档则会恢复当时的学习开关；调用方要冻结评估，应在 `restore()` 之后再次 `set_learning(False)`。切换场景不代表已有权重能够立即在新任务有效工作。

## 协作停止与中断

`request_stop()` 要求完成当前神经请求后停止接受新 `step`，随后仍允许 `save()`；`wait_until_idle(timeout)` 可等待请求边界，`clear_stop()` 允许继续推进。在主线程使用默认 SIGINT 处理器时，第一次 Ctrl+C 等当前完整响应到达再抛 `KeyboardInterrupt`，worker 保持可存档，已完成的结果在 `last_interrupted_request` 中。再次 Ctrl+C 会终止该客户端自己启动的 worker。客户端独立进程组不影响其他服务；`close()` 只清理自己持有的进程。中断只保证完整神经区间的边界，场景身体状态是否同步保存由场景负责。

真实 SIGINT 实测已完成：200 ms 全图步骤中触发中断，PPL 脉冲完整交付；捕获中断后存档、恢复及继续推进结果精确一致，协作停止也拒绝新步且允许保存，worker 正常退出。证据：`artifacts/brain-interrupt-d93c48d407c1/evidence.json`；复现：`python -m connectome_adapter.validate_interrupt`。

## 已完成的实测

2026-09-13，实测加载完整图，4 次 RGB 区间推进共 257.2 ms 神经时间，记录 191,504 次实际放电。一个显式厌恶事件分段累计交付精确 200.0 ms。存档恢复后神经时间和记忆 SHA256 一致；保权重置通过；冻结后再次呈现刺激未改变记忆 SHA256。测试 worker 已正常关闭。

机器可读证据：`artifacts/connectome-adapter-validation/evidence.json`。这验证的是接口与神经计算实际发生，不能证明跨场景迁移成功、条件反射形成或任务表现提升。

直接输入对照也已完成：从同一个完整脑存档、同一张黑 RGB 图像推进 100 ms，纯 RGB 为 28,529 次放电；左眼 1,107 个 R1–R6 注入 10 mV 后为 30,725 次，右眼 2,228 个 R1–R6 注入 10 mV 后为 33,014 次，23 个 LB3c 注入 30 mV 后为 37,561 次。右眼和糖输入还改变了真实下游 BCI 读出；左眼在这个短窗口未改变最终动作，记录保留此结果。各条件均冻结学权，重复基线完全一致。组合输入下 PPL 交付精确 200 ms，保存/恢复后输入状态、后续放电和学权完全一致。证据：`artifacts/connectome-direct-drive-20260913T100653Z-aa426d/evidence.json`；复现实验：`python -m connectome_adapter.validate_drive`。

全量接口验证也已通过：细群、宏群各覆盖全部 166,700 节点，47 宏群 dense 电流实际驱动每个节点；同初态与黑图下，完整左侧 R1–R6 细群的 1,108 个节点施加 +10 mV，100 ms 群放电从 0 增至 3,328。全部状态数组保存恢复精确一致，PPL 仍为 200 ms，导出的所有原始 ID、CSR 指针和目标节点与发布图逐项完全相等，没有裁剪拓扑。这里的注释细群包含尚未分配到旧视网膜映射端口的一个节点，因此群大小与旧左端口的 1,107 不同。证据：`artifacts/full-brain-validation-fc29109bf1cf/evidence.json`；复现：`python -m connectome_adapter.validate_full_io`。

## 来源与边界

### 逐神经元二进制电流直通

`neuron_currents_file` 接受 **166,700 个独立的连续电流值**。数组的每一项直接对应原始 `graph.ids` 顺序中的一个已存在神经元，不经宏群复制。它与旧三端口、细群、宏群、单 ID 接口共存。调用示例：

```python
import numpy as np

arrays, state = brain.read_arrays()
currents = np.linspace(-5, 5, len(arrays["ids"]), dtype=np.float32)
descriptor = brain.write_neuron_currents(currents, ids=arrays["ids"])
result = brain.step(
    rgb,
    duration_ms=28.6,
    visual_input_enabled=True,  # 原有视觉照常输入；模型是额外一路电流。
    general_stimulation=[{"neuron_currents_file": descriptor}],
)
assert result["general_stimulation"]["current_vector_sha256"] == descriptor["currents_sha256"]
```

无需启动脑的写入函数为 `connectome_adapter.neuron_currents.write_neuron_currents(currents_mv, ids, directory=None)`。输入必须已经是 `float32[166700]` 和 `int64[166700]`，每项电流有限且处于 `[-30,30]`，单位为模型中的 mV 等效附加驱动，不是经过生理标定的电极电流。返回的小描述符如下；实际数组不进入 JSON：

```json
{
  "schema": 1,
  "path": "/absolute/project/artifacts/neuron_currents/currents-UUID.npz",
  "sha256": "64位文件SHA256",
  "ids_sha256": "64位int64原始ID顺序SHA256",
  "currents_sha256": "64位float32电流数组SHA256",
  "neurons": 166700
}
```

NPZ 只能含 `currents_mv` 与 `ids` 两个数组。文件只能位于当前项目 `artifacts/neuron_currents/` 内，完整写入后原子更名；worker 检查普通文件、大小上限、ZIP 内容、禁用 pickle、整文件 SHA、数组 SHA、dtype、shape、有限值、范围，并将全部 ID **逐值**与已加载图核对。校验和读取使用同一份有限大小的内存字节。错误请求在推进神经时间前被拒绝。当前实际 ID 顺序 SHA 为 `6b6b40c3bddf84b1281ef0b18db06927c2bf61c5f4cb32a4219ec9b7839dc2e5`，调用方仍应从当前 `describe()` / `groups()` / snapshot 读取并验证身份。

同一时间步可按列表顺序相加多个逐节点文件与其他明确刺激，总规格仍最多 16 项，合计附加电流仍受每神经元范围限制。`general_stimulation.current_vector_sha256` 是本次实际通用电流总数组的 SHA；单独一个文件刺激时，它与描述符 `currents_sha256` 精确相等，包括被遮罩的负零。多个来源时，`neuron_current_components` 分别返回每个实际加载并校验的文件分量 SHA、原列表位置和全节点覆盖数。另返回完整 `coverage`、非零的 `driven_neurons`、`ids_sha256` 和最后积分分段的 `kernel_final_total_drive_sha256`；后者还包含原模型 tonic、固定 lamina 偏置、可选视觉、三端口及该分段的 PPL，不能与仅附加电流的 SHA 混用。

每个输入只用于当前区间，没有隐式持续保持。调用方等待 `step` 完成后可删除输入文件；脑存档保存真实神经状态和历史描述符作为来源记录，恢复时不会重新打开该文件，也不会重新施加上次刺激。保存/恢复继续使用所有真实神经元和全部保留边。

正常使用默认 `visual_input_enabled=True`：原有 RGB 视觉、三端口、原通用刺激都照常输入，MiniCPM 的电流作为额外一路并联加入。`shared_io.parallel_inputs.prepare_parallel_stimulation(...)` 根据真实 ID 和注释展开全部原刺激，按 worker 的 float32 顺序计算；若追加输入会超出每神经元附加电流预算，只限幅新的模型分量，保留原输入不变。原输入自身已越界会被明确拒绝。返回 `expected_general_currents_sha256` 用于核对实际总和，以及原输入、请求与实际模型电流的独立摘要和修改数量。原视觉驱动、tonic 与 PPL 不计入这项人工附加电流预算。

`visual_input_enabled=False` 仅作为可显式选择的隔离实验：绕开上游 R1–R6 与 R8 图像投影，并清零旧光适应状态以去除残留视觉驱动；原固定 lamina 偏置、tonic、LIF 动力学、完整连接传播和 PPL 时间规则保持。RGB 仍被读取和哈希。这种隔离对照不代表正常并联模式。

完整读接口的 NPZ 保留原有 `ids / v / counts / cumulative_counts / drive`，metadata 新增 `ids_sha256`、`duration_ms`（真实上一步时长，初态为 0）与 `visual_input_enabled`。电位是膜电位，counts 是本步真实放电数；不能把聚合数、推断时长或生成的数值冒充原始状态。

### 输入、输出注释的科学边界

当前真实注释中，`cb_sensory` 为 4,868 个、`ol_sensory` 为 6,098 个、`vnc_sensory` 为 6,370 个，共 **17,336** 个。`cb_motor` 为 107 个、`vnc_motor` 为 708 个、`descending_neuron` 为 1,314 个，共 **2,129** 个。这些类别可以用作明确的输入或读出遮罩，具体 ID 由 `macro_group_index` 与目录注释匹配得到；其他所有神经元和连接仍参与仿真。

这些是解剖注释，不提供“某一句话应该刺激哪一个细胞”或“某个放电值就是左转”的自然语义。将多模态模型特征映射到感觉细胞、将运动/下行细胞状态训练为身体动作，仍是待评估的工程映射。直接刺激运动神经元可能绕过期望的感觉至运动路径，因此应在实验记录中区分全节点干预与感觉输入/运动读出配置。接口支持全部节点，并不意味着当前语言模型投影具有 166,700 个完全独立的有效自由度。

逐节点真实验证入口：`python -m connectome_adapter.validate_neuron_currents`。它检查同初态因果对照、所有位置的实际驱动、身份/文件拒绝、关闭旧视觉后的图像无效性、删除输入文件后的精确脑恢复、200 ms PPL 以及全部原图 ID/CSR 拓扑。

并联保留输入的真实验证：`python -m shared_io.tests.validate_parallel_inputs`。`artifacts/parallel-input-check-5e6649a4f7c2/evidence.json` 记录了原视觉、三个感觉端口、旧逐细胞文件、宏群刺激、单 ID 刺激与新模型输入同时运行；原刺激不变，新分量按预算限幅，实际通用电流总 SHA 精确匹配，PPL 仍精确交付 200 ms。该验证证明输入按约定被实际整合，不表示行为效果已达到目标。

上游：[DOOMFLY](https://github.com/nftechie/doomfly)、[MaleCNS 数据](https://male-cns.janelia.org/download/)、[v6 协议](https://github.com/nftechie/doomfly/blob/main/docs/doom-live-training.md)。

真实连接拓扑不等于所有模型假设都经过生物学验证。视网膜投射、近似 LIF 动力学、人工刺激和 BCI 映射都有工程假设。v6 上游仍明确标记视觉、条件反射和生存表现尚未通过验证。学习参数发生变化，只证明可塑性机制运行，不能当作“学会了”的证据。

DOOMFLY 原创代码 MIT、MaleCNS 数据 CC BY 4.0；使用与分发时保留上游许可和归属。
