# 完整大脑计算与临时状态提速

2026-09-14。本次仍计算 166,700 个神经元和 25,582,938 条聚合有向连接，保留原感觉输入、突触权重、C++ 仿真核及事件步长。没有调用或推进正在运行的主脑。此文中的倍数属于指定局部操作，不能解释为整套系统已实时。

## 实际测量和已实现修改

| 操作 | 原实现中位数 | 新实现中位数 | 条件 |
|---|---:|---:|---|
| 47 宏群完整统计 | 25.54 ms | 1.90 ms | 同一真实脑状态，5 次交错，无 profiler |
| 23,163 细群完整统计 | 27.72 ms | 5.19 ms | 同上，全部字段 JSON 相同 |
| C 盘完整状态 NPZ 导出 | 161.40 ms | 152.19 ms | 同目录介质，完整文件字节相同；提升较小 |
| 临时预算状态写入、消费、SHA 复核与删除 | 254.01 ms | 43.80 ms | 同一份实际归档的 8.175 MB 全脑数组，C 与 Linux，5 次交错；这是 I/O 回放，没有新仿真 |

`full_io.py` 在不变的真实神经元分区上预建排序索引，使用 `reduceat` 计算电压极值。总和及平均值保持原 `bincount` 的顺序。`_init_group_reductions()` 只建索引缓存，既不清累计放电，也不重置刺激或学习记忆。

普通 NPZ 在不超过 32 MiB 时先在内存组包，随后写盘、原子更名，再从实际文件完整计算 SHA。包含全部边的较大导出继续原有流式写入。没有省略数组、放宽校验或复用旧校验结果。

独立真实仿真重放 3 × 28.6 ms，全部神经元放电计数、原生动作读出、完整状态数组与旧实现逐值一致；读操作前后的全部动态状态和权重 SHA 也一致。旧实现和新实现保存的 NPZ 文件整体 SHA 相同。

## 临时状态的精确范围

仅 `NativeOmniFlow` 在没有已固定的模型观察时，为当前电流预算创建的临时快照使用 Linux 本地目录：

```
/root/.cache/cyberfly/brain-state/<独占 owner UUID>/budget/<lease UUID>/state-<UUID>.npz
```

该单一根定义于 `connectome_adapter/hot_state.py`。目录要求当前进程用户所有、权限 0700、非符号链接；每个池/主会话 facade 的归档目录对应本进程独占 owner，每次消费对应新 lease。只有 `shared_io.parallel_inputs._snapshot` 的预算读取入口额外接受它。`NeuronStateCodec`、模型输入根、训练验证根及通用文件访问范围均未放宽。

文件仍由实际 `brain.snapshot()` 生成，描述符保留实际 Linux 路径及实际 SHA。消费者完整校验文件 SHA、全部真实 ID 和分区数组；消费结束后再次校验 SHA 和文件身份，才删除本进程明确登记的文件。普通消费者失败也进行同样清理；校验损坏时拒绝删除，保留诊断文件。没有目录扫描或递归清理，不收养另一进程的文件；进程退出只移除自身已空的 owner 容器。

永久回执明确写入：

- `budget_storage.kind: linux_hot_budget`、真实 owner/lease、创建进程和时间；
- `raw_snapshot_archived: false`，该临时数据不冒称永久可回放证据；
- `removed_after_budget_consumer_completed: true` 与 `removed_before_physical_step: true`。

之后的真实脑步仍独立核对实际电流总和。不能再使用旧的 `removed_after_verified_delivery` 表述：新生命周期是在预算消费者完成之后、物理执行之前释放数据。

已经固定的输入观察、执行后的完整反馈、训练样本、历史证据仍保存在原 C 盘路径。没有改写任何旧描述符或清理已归档证据。模型进程不需要为这项修改重启；主 Lab 和独立环境进程需正常加载新 Python 代码。

## 验证文件

- 初始隔离基线：`artifacts/brain_performance/baseline-20260914-01/evidence.json`。
- 分组统计与完整状态逐值验证：`artifacts/brain_performance/optimized-20260914-01/evidence.json`。
- 热目录 I/O、错误 SHA、越界、缺身份、符号链接、消费异常清理、变更后拒删：`artifacts/brain_performance/hot-budget-20260914-01/evidence.json`。
- 新代码源 SHA 与交付状态：同目录 `completion.json`。
- 真实独立 Lab、完整身体及全脑的 19 项验证：`artifacts/session_flow_checks/36d22fbabf30/evidence.json`。包含原感觉输入、真实 PPL、一次性电流、正确热状态清理、原持久观察保留、单条真实在线样本且 0 训练提交；自有身体/脑进程全部关闭。
- 首次验证新增断言变量错误的失败记录保留：`artifacts/session_flow_checks/590928073cc5/evidence.json`。没有回改为成功。

## 尚存瓶颈与未采用方案

当前神经仿真核是 CPU C++ 事件驱动实现。基线中实际 28.6 ms 神经区间的 kernel wall time 为约 53–83 ms；这次没有修改内核，也没有把运行迁移到 GPU。

C 盘完整快照读写仍分别约 60 ms，观测/反馈导出并未迁移。无损压缩 1 级虽将文件缩至约 2.46 MB，但压缩本身约 184 ms，完整 I/O 反而由约 136 ms 变为 232 ms，故未采用。证据为 `optimized-20260914-01/compression-trial/evidence.json`。

后续若迁移完整观察/反馈，必须同时处理模型 codec、持有租约的 retention、在线样本向 C 盘的独立持久复制以及模型进程所加载的源码版本；本轮明确未做这第二阶段。主会话的部署和完整模型闭环验证由主任务协调，隔离测试本身不等于主服务已更新。
