# CyberFly: 具身 MiniCPM 与完整果蝇连接组的语言驱动飞行

## 摘要（2026-10-01 修正版工作稿）

我们构建了一个可复现实验系统，将 MiniCPM-o 4.5、雄性果蝇全脑连接组（166,700 个神经元、25,582,938 条聚合连接）和 FlyBody 完整身体接入同一可审计接口。当前证据确认 78/78 执行器映射、103/103 物理关节及左右翅三轴联动；冻结完整演员与公开教师在同一组 20 个预先封存未见初态上均通过 2 秒悬停门。将最终演员的六个翅膀动作替换为冻结零更新锚点后，全部 20 例仍完成 2.5 秒且零触地，并呈现可重算的身体轨迹差异；这是有限初态的策略干预证据，不证明独立学习了每个关节。

统一四轴结果按协议报告：旧跨集合 fit 存在 hidden 重复，仅 2/4 方向通过保持门；新 25/25 独立提示分割没有跨分割 hidden 重叠，同一 fit/readout/完整身体初态在显式到达锁存下通过四个 3 mm/0.5 秒物理门。相同 −X 无锁存路径在 0.78 秒到达后只保持 0.46 秒，锁存路径保持 1.14 秒。70 条原始运动请求的模型文字答案目标准确率为 27/70，当前最近模板为 63/70，二者一致 28/70且没有完美静态映射；因此锁存物理通过不等于原生文字方向能力。

显式符号锁存的 native ±Z current/工程读出绑定在完整身体中实现 +4.0267/−3.9753 mm 垂直位移并维持姿态，仍不构成 learned Z：50 条原生 ±Z 请求都落入已有 ±Y 模板；四状态训练、第五状态记录留出的 native-template hidden→current 复验在同一留出状态的十条记录上全部预测为 zero，未注入身体。

独立脑实例的修正版 M4b 批次给出当前协议下的可复核阴性边界：惩罚组与未惩罚组各 10 个随机化实例的 MBON11 A−B 均值分别为 −0.9 和 −0.7，组间差 −0.20，精确标签置换 p=0.9326；三个独立 checkpoint 的 MBON11→DN 探针源放电增量为 408/439/438，而 DN 放电差为 −2/−2/−4。早期“9/10 对 2/10、p=0.005”的连续/非独立记录已撤回。本文把阳性结果限定为声明协议下的工程链路、姿态/传动和有限策略干预，并保留原生文字方向、无状态稳定性、learned Z、独立关节学习与更大群体泛化的开放边界；不宣称气味特异性飞行效应或整体目标完成。

## 1. 研究问题与贡献

本文只主张工程系统能力，不主张模型具有主观体验或生物等价性。研究问题是：

1. 语言模型能否在完整果蝇脑闭环中决定到达、转向、分段与规则执行？
2. 结果是否在置零电流、转向取反和规则缺失等预先声明的对照中消失？
3. 连接组中的突触变化能否被稳定读出并影响身体行为？

贡献：

- 一个固定版本、可审计的 MiniCPM-o 4.5—MaleCNS—FlyBody 接口，以及可复现的协议、权重和原始轨迹身份记录。
- 对完整身体的传动感知审计：78 个执行器、103 个物理关节、固定肌腱、接触依赖传动和左右翅三轴联动均从真实 50 微秒子步重算。
- 20 个预先封存未见初态的完整演员/教师同初态悬停复核，明确区分保留的预训练能力与本地更新贡献。
- 同一 fit/readout/初态下的独立提示四轴锁存物理通过，以及无锁存、原生文字方向和 native learned-Z 的独立负边界。
- 全部 20 个预先封存初态上的训练翅膀输出闭环干预，以及 M4b/MBON11→DN 修正版阴性结果的可复核记录。
- 对旁路、工程重标定、统计独立性和因果解释范围的明确限制。

## 2. 系统

完整脑状态以 28.6 ms 神经区间积分，身体以 10 ms 外层步长和 0.05 ms 物理子步长运行。文字代号与 hidden→current 输出分别归档；二者的世界轴标签可能不同，当前最近模板仅作为审计诊断，不改写模型请求。host 误差纠偏、到达锁存和符号锁存只在各自明确声明的协议中启用，原生对照排除 host 方向覆盖。感觉与运动端口是工程接口：逐细胞直连覆盖 17,336 个感觉入口和 2,129 个运动/下行出口；全脑计算仍保留 166,700 个神经元。

运行时版本与权重身份记录在 `sources.lock.json`、`model_training/weights-manifest.json` 及各批次 `protocol.json`。本轮验收结束后数值服务已停止，GPU 计算进程为空；历史运行所用 GPU 作为当时的资源记录保存，不代表当前服务状态。所有在线请求均使用本地回环地址，并保留加载、禁用和停止回执。

## 3. 预注册验收

| 里程碑 | 完整条件 | 对照 | 证据 |
|---|---:|---:|---|
| M1 单目标语言飞行 | 17/20 | zero 0/20；swapped 0/20 | `artifacts/sealed_acceptance/20260917-005311` |
| M2 两段语言计划 | 15/20 | zero 0/20 | `artifacts/sealed_acceptance/m2-20260918-203659` |
| M2b 任意航点路径 | 18/20；55/60 航点到达 | zero 0/20；swapped 0/20 | `artifacts/sealed_acceptance/m2b-20260919-001018` |
| M3 语音自报告 | 16/20 | 无记录 4/20 | `artifacts/sealed_acceptance/m3-20260919-043230` |
| M4 气味特异记忆 | 未通过 | — | `docs/BRAIN_PLASTICITY.md` |
| M4b 经验依赖飞行决定 | 当前修正版未支持正效应 | 20 个随机化脑实例的组间差 −0.20，p=0.9326；飞行绑定失败 | `artifacts/m4b_memory_flight/state_randomized_instances_20260925/` |
| M5 即时语言规则 | 10/10 | 规则缺失 0/10 | `artifacts/m5_rule_flight/sealed-20260919-065810` |
| M6 新控制器稳健性 | 19/20 | zero 0/20 | `artifacts/m6_combined/sealed-20260919-083323` |

M1 的早期开发批次 2/20 保留在 `artifacts/sealed_acceptance/20260916-204932`，不与最终批次合并报告。

## 4. 记忆实验

上游内核自身的塑性规则在真实气味输入下没有产生可检测的气味特异群体效应。触角叶抑制边 ×4 的工程重标定使 KC 活动进入稀疏范围，但任意两种气味在可塑边上的驱动仍有至少 54% 重叠。外部三因子 LTD 规则则将权重写入同一批 4,184 条 KC→MBON11 边；被惩罚脑 MBON11 平均 3.1 脉冲，未惩罚脑 10.5 脉冲。

M4b 的闭环读取以文字数值进入 MiniCPM，规则阈值 7 在飞行前声明。早期 20 行记录来自派生检查点的连续呈现，不能作为 20 个独立条件化学习实例；其“9/10 对 2/10、p=0.005”行为推断已撤回。当前主要复核为 corrected10 同初态配对（A−B 效应均值 −0.075，符号置换 p=0.8945）与状态随机化批次（组间差 −0.20，标签置换 p=0.9326），均未支持当前协议下的正效应。训练后 pair-00 两臂的真实飞行绑定都失败，故神经读数变化不能写成飞行记忆成功。

## 5. 统计与可复现性

逐飞原始轨迹已由只读审计脚本从 50 微秒数组重算；协议、源码/权重身份、留出分割和大型原始文件的 SHA256 已进入 `artifacts/reproducibility_20260925/manifest.json`。当前清单包含 22942 项，最终交付完整性审计通过且无缺失或哈希错误（`artifacts/final_delivery_audit_20260926.json`）。统一耦合器步数扫描也保留了 20/30/40/45/48/50/55/1200 步的独立 evidence；其历史报告值为 50 步 42/45，但 2026-09-30 hidden 字节复核发现该分割有 29/45 留出行与训练重复；去重后的不重叠子集为 13/16（1200 步为 12/16），因此不能把 42/45 写作独立泛化结果。对应 X 轴全身保持仍为 0.49/0.33 秒（`artifacts/unified_fit_sweep_audit_20260929.json`）。统计单位和阴性边界按脑实例或同初态配对报告，不把连续轨迹行当作独立样本。

## 6. 限制

### 连续双工链路补充验收

在本次接手后的复验中，真实 16 kHz 音频经过原生 APM/LLM/TTS 流，连续产生 5 个带脑与身体反馈的单元；同一 epoch 内序号连续，LLM cache 从 8 增至 464，单元均推进真实脑时间和身体时间，产生 2 个原生语音块，并在结束时关闭双工会话。证据位于 `artifacts/full_duplex_checks/f03223687576/evidence.json`。这证明连续缓存和数值脑/身体注入链路，不证明实时性能或任务能力。

- 感觉/运动端口、抑制增益和外部 LTD 是工程设计，不是完整生理重建。
- 连接组存在有损读出与旁路；zero-current 只能证明输入电流对任务必要，不能单独证明每条连接的生物必要性。
- M4 气味特异判据未通过；修正版 M4b 未支持当前协议下的正效应。记忆读数仍经文字接口进入模型，不能作为脑内直达行为因果。
- 身体、脑和模型的时间步长不同，未宣称严格生物时钟同步。
- 目前 vLLM-Omni 真实文本/服务链已验证；连续双工、原生语音和脑数值 token 的多单元验收仍需单独封存。

### 2026-09-30 ±Z 原生电流读出与身体绑定边界

### 2026-09-30 live-readout 四轴方向诊断

### 2026-09-30 unified learned-current 四轴短窗复验

将同一个跨集合统一 fit 的 `holdout-set0` learned-current 模板接到 live readout 和完整演员，四个同初态回放各运行 120 个 10 ms 外层步。+X、−X、+Y、−Y 的 signed majority 均正确，脑输入均 120/120 消费且无丢弃，沿轴位移为 4.216、3.759、2.633、3.809 mm，最大绝对 Z 漂移 0.109–0.118 mm。原始证据在 `artifacts/model_driven_unified_live_readout_20260930/`，审计为 `artifacts/model_driven_unified_live_readout_audit_20260930.json`。该结果支持统一 fit 的短窗四轴方向传递；由于没有航点锁存和 0.5 秒悬停阶段，仍不构成统一三毫米物理门。

为区分脑状态分布和 current 模板本身的影响，将真实闭环飞行状态训练的 live readout（留出准确率约 90.7%）接回原生 +X/−X/+Y/−Y current 模板。四个同初态回放各运行 120 个 10 ms 外层步，四个 majority readout 均与声明方向一致，脑输入 120/120 消费、队列丢弃为 0，最大绝对 Z 漂移 0.109–0.118 mm，最终沿轴位移为 +X 4.205、−X 3.759、+Y 2.942、−Y 3.888 mm。原始证据在 `artifacts/model_driven_flight_live_readout_20260930/`，审计为 `artifacts/model_driven_live_readout_direction_audit_20260930.json`。这改善了短窗方向传递诊断，但不构成 3 mm 到达/0.5 s 悬停门，也不替代统一 learned-current hidden→current 验收。

为分离原生语言接口和工程读出，`tests/z_axis_kernel_probe_20260930.py` 从原生 `/encode` 的真实 +Z/-Z current 文件取 10 个互不相同的变体，在同一完整脑初态上逐变体运行 16 个 28.6 ms C++ kernel 区间。160 个区间均保留完整 166700 神经元/25582938 条边和学习关闭校验，得到 50 个输出细胞测量行；没有模型请求或物理步。独立三输出头 `tests/fit_z_readout_20260930.py` 在 40 行训练集达到 40/40，在留出变体达到 7/10，证据和审计为 `artifacts/z_axis_readout_fit_20260930/3fedc285440d/` 与 `artifacts/z_axis_readout_audit_20260930.json`。

将同一读出头接入完整身体后，+Z 与 −Z 各运行 120 个 10 ms 外层步；两臂均 120/120 个真实脑输入被消费、无队列丢弃、持续在 flight 模式，每步均有显式 current 组件，高度范围为 7.916–8.132 mm 与 7.965–8.104 mm。原始证据为 `artifacts/z_axis_brain_flight_20260930/plusZ_capture/evidence.json` 和 `artifacts/z_axis_brain_flight_20260930/minusZ/evidence.json`。持续脑状态使读出符号在两臂内翻转，末端垂直位移为 −0.0687 mm 与 +0.0301 mm，因此这关闭了工程脑—身体三轴绑定的可复核空白，却不构成 learned closed-loop Z 任务通过；原生 +Z/-Z 文本仍在既有模板中退化为 −Y。

## 7. 待补实验

已完成的投稿前复核包括：状态随机化 M4b 20 实例、三个独立 MBON11→DN checkpoint、20 例封存未见初态完整演员配对、78/103 传动与翅膀审计、统一 fit 四轴回放、原生 ±Z 负边界和最终 manifest 哈希检查。仍开放的实验门是：

1. 将本轮独立 ±Z current 响应读出提升为稳定的同一 fit 三维身体门，并补齐原生语言模型到 ±Z learned-current 的稳定世界轴映射；当前 7/10 留出与符号翻转只算工程链路边界，工程参考旁路仍不计入该门。
2. 固定轨迹的双侧/单侧翅膀执行器消融已在四个封存初态中确认物理执行器贡献；训练演员的六个翅膀输出又在全部 20 个封存初态上完成了成对闭环替换审计。若要声称独立关节学习或闭环 learned wing control，仍需预注册单关节/单侧翅扰动、匹配 sham 和干预后的策略适应测试。
3. 若要声称 MBON11 的行为下游因果，需要去掉文字数值跳转的直接神经—下降神经元闭环和预注册行为读出。

这些开放门不会被历史 M1–M6 表格中的早期基准或不同 fit 的方向结果拼接替代。


### 当前候选模型的独立 HOLD 检查

HOLD v2 QLoRA 在未参与训练的 40 条合成文本测试上达到 38/40 严格匹配，8/8 个 HOLD 正例正确，但有 2 个错误 HOLD（应为 NY、PY）。v3 保留同一 40 条 test 并达到 40/40、HOLD 8/8、错误 HOLD 0；只读汇总审计为 `artifacts/hold_text_evidence_audit_20260929.json`。v3 随后已按 SHA 在线加载并完成物理负边界复验：30 次文字输出 +X，但实际 current 最近模板均为 −X，未通过到达/保持；40/40 仍只证明文本决策门，不宣称 HOLD 飞行能力。旧记录中的 19/20 属于 M6 物理飞行里程碑，与 HOLD 文本分母分开。

### 身体控制范围

FlyBody 物理模型保留 78 个执行器和完整关节动力学，当前统一神经接口为 9 维动作：2 个腿侧驱动、6 个翅膀残差和 1 个拍频修正。其余关节由步态/拍翼生成器、弹簧阻尼和物理接触联动。静态审计证据位于 `artifacts/body_control_audit_20260925/evidence.json`；因此本文不把“78 个执行器存在”表述为“脑直接学习了全部 78 个关节”。

### 未见初态完整演员配对（2026-09-25）

为避免把已查看轨迹当作泛化证据，使用预分配但未评分的 `seed=208020` 做原始教师与 198000 步微调完整演员的同初态配对。两条件均完成 0.75 s、15,001 个 50 μs 物理子步、无触地，并通过 0.5 s 短悬停门；最长有效窗口均为 0.68355 s。由于回合时长为 0.75 s，不能据此声称 2 s 长悬停通过；两条件汇总指标一致，因此不报告微调优势。证据保存在 `artifacts/unseen_initial_pair_208020_20260925/evidence.json`，属于开发性未见初态检查，不是封存最终测试。

### M4b / MBON11 路径探针修正（2026-09-25）

小批次外部三因子 LTD 复验（每臂 2 次配对训练、3 次读出重复）得到：paired 与 swapped 臂的 MBON11/树突读数发生变化，但置换检验均不显著；unpunished 臂权重和读数不变。该结果作为可复现的探索性阴性统计，不支持当前样本量下的气味特异性成功结论，证据为 `artifacts/m4b_memory_flight/pilot-20260925/evidence.json`。

重新读取完整图后发现 MBON11 有 1,649 条出边，其中目标类型包含 DNp52（2 条）和 DNp62（1 条）。因此“没有直接 MBON11→下降神经元边”的旧表述已修正为“存在直接 DN-like 解剖边，但尚无因果证据”；因果仍需 MBON11 扰动、下游读出和匹配 sham。

### MBON11 扰动探针

在同一完整脑检查点上，ORN_DA1 四个间隔的 MBON11 `-30 mV` 扰动使 MBON11 输出由 9 降至 0；匹配 MBON18 sham 仍为 9，说明源神经元扰动有效。但 DNp52/DNp62 在 baseline、MBON11 扰动和 sham 中均为 0 脉冲，当前读出对下游因果不敏感，故不报告 MBON11→DN 的功能因果结论。证据为 `artifacts/m4b_memory_flight/mbon11_causal_probe_20260925/evidence.json`。

同参数第二次独立进程复跑 `pilot-20260925-rep2` 逐项复现了上述数值（paired/swapped 的读数变化及不显著置换检验、unpunished 无权重变化），因此该阴性结果在本小批次设置下可复现；它仍不替代冻结协议要求的每臂 10 个独立脑实例。

### 独立脑实例正式批次的拒收

按修订后的实例级 CS/US 电流和空白间隔随机化，实际运行了每臂 10 个 worker。20 个进程最终只有 3 个唯一权重 SHA256，说明进程隔离和刺激元数据随机化仍未产生独立塑性脑状态。A−B 读数差为 -0.7，精确置换 p=0.713；该统计仅作为设计失败诊断，未纳入 M4b 正式推断。协议已标记 `run_rejected_design_duplicate`，下一版必须随机化/预置神经状态并在评分前强制检查唯一身份。

### 状态随机化独立脑实例批次

修订协议为每个 worker 先运行随机前置神经刺激并保存唯一派生 checkpoint，再连续完成 20 次配对和一次非 reset A/B 读出。每臂 10 个、共 20 个实例的初态 SHA256 全部唯一，该历史批次的唯一哈希只覆盖塑性权重且未同初态配对，不能作为独立身份门；其结果仅保留为探索性未检出，不能作为正式阴性结论。每实例航点飞行尚未执行。

### M4b 实例到航点飞行绑定

将 punished-00 的唯一派生脑 checkpoint 接入真实 `waypoint_flight` 后，模型两次输出 `-X`，但 100 步物理结果位移为 `[-0.428, 2.260, 0.014] mm`，未到达目标且未保持 0.5 s。该回合证明 M4 数值规则已进入完整脑—耦合器—身体链路，同时给出当前 M4→航点闭环失败结果；不能写成飞行成功。

正确绑定 punished-00 的 MBON11 实际读数 `4` 后重跑 M4 航点链路，规则选择 B 且模型输出 `-X`；物理结果仍为 `[-0.428, 2.260, 0.014] mm`，未到达/悬停。由此排除前次外部数值错配，当前失败归因于控制模板与身体世界轴/侧向响应，仍不能报告为行为成功。


## 2026-09-26 证据复核：撤回过强验收表述

以上历史记录中的“20/20 初态身份唯一即神经阶段通过”和“没有气味特异性效应”不成立。源码表明 memory_sha256 仅覆盖塑性突触权重；唯一哈希不能证明统计独立性，重复哈希也不能据此剔除实例。该批次使用各臂不同前置刺激/电流，未做同初态配对；实际为10次A与10次B呈现，惩罚配对仅10次。训练后完整检查点未保存，飞行加载的是训练前派生检查点。故现有结果仅为探索性未检出，G3仍未完成。三种飞行条件失败也不足以唯一定位世界轴映射错误；交换标签仅是待检验假设。

复核证据：`artifacts/evidence_reaudit_20260926/audit.json`。原始轨迹和历史记录保留，正式结论以本更正为准。

### HOLD v3 候选

在 v2 训练分割基础上增加近阈值方向负例，完全保留原 40 条 test。v3 QLoRA 实际完成 150 步并验证 adapter 重载。冻结独立 test：候选 40/40 exact、HOLD 8/8、错误 HOLD 0，达到文本门槛；运行时候选适配器到 PX/NX/PY/NY/HOLD 解析预检 5/5 通过。候选尚未激活到共享服务，也尚未完成物理身体到达/悬停验收。

### HOLD v3 候选身体回放

将通过文本门槛的 v3 候选 NX 决策回放到真实身体链路，并与相同派生脑初态 zero-current 对照。候选最终位移为 `[-2.203, 1.861, 0.017] mm`，zero-current 为 `[0.216, 2.017, 0.016] mm`；候选确实改变了 X 向运动，但两者均未到达 `[-1.6,0]` 目标，也未保持 0.5 s。候选回放不是在线 LoRA 部署，结果证明候选决策进入身体后仍有明显侧向漂移。

### 航点闭环提示修复

审计发现 steering prompt 原先使用沿当前方向合成的理想位移，遮蔽真实侧向漂移。`waypoint_flight.py` 已改为传入实际世界位移。200 步验证中模型从 `-X` 切换到 `-Y`，证明闭环反馈开始响应真实漂移；但切换发生过晚且耦合器模板仍有方向不一致，最终位移 `[-2.787, 2.528, 0.026] mm`，未通过到达/悬停门。

### 误差轴覆盖诊断

加入开发性最大误差轴 steering 后，第 100 步即由 `NX` 切换为 `NY`，但物理轨迹与普通闭环相同，最终仍为 `[-2.787, 2.528, 0.026] mm`。`NY` canonical prompt 的电流仍被 nearest-template 识别为 `-X`，因此问题收敛到 hidden→neural-current 的方向标定，而非 steering 触发时机。该模式仅作诊断，未进入默认控制。

### 四方向身体模板校准

在同一派生初态分别运行 PX/NX/PY/NY canonical prompt 各 50 步。最终位移分别为 PX `[-0.972,0.474]`、NX `[-0.688,0.810]`、PY `[-0.862,0.463]`、NY `[-0.372,0.832]` mm；四者都带共同漂移，不能支持简单的正负标签交换。随后在同一初态同步运行 50 步 zero-current，终点为 `[0.239,0.732,0.019]` mm。四个方向相对该基线的位移增量分别为 PX `[-1.211,-0.258,-0.003]`、NX `[-0.927,0.078,0.001]`、PY `[-1.101,-0.269,0.005]`、NY `[-0.611,0.100,-0.001]` mm，且全部未达到/保持目标。该结果把共同漂移与命令增量分离，但增量仍不构成可接受的世界轴映射；候选物理门保持未通过。证据 `artifacts/hold_candidate_waypoint_link_20260926/direction-calibration-with-zero-50.json`。

在这些同步试验的 0.5 s 外层步端点，四个命令相对 zero-current 的 78 维 actuator vector L2 差为 PX 1.468、NX 0.049、PY 0.456、NY 0.008，根角速度差的最大分量为 37.41、1.06、2.35、0.18 rad/s。说明命令已进入身体的多关节/姿态链路，但差异高度不对称，仍不足以建立世界轴控制；完整物理子步和关节名称归因仍需独立测试。证据 `artifacts/hold_candidate_waypoint_link_20260926/actuator-response-diagnostic-50.json`。

为核验外层摘要没有掩盖物理联动，对 PX 与 zero-current 进行了真实子步记录。每组 50 个外层步均产生 10,000 个 50 μs solver samples（每步 200 个），保存完整 78 维 ctrl、actuator force、qpos/qvel 和触地标记；两组均无触地，且都未达到/保持目标。PX 相对 zero-current 的末端 ctrl 差 L2 为 1.468，根角速度差为 `[6.41,37.41,3.43]` rad/s，位置差为 `[-1.211,-0.258,-0.003]` mm。该证据证明完整身体确实被驱动并产生条件相关姿态变化，但不构成飞行任务通过。证据 `artifacts/hold_candidate_waypoint_link_20260926/physical-substep-comparison-50.json`。

### 同初态 M4b 配对试运行

我们实现了同初态配对协议：每对由一个共享随机前置后的完整脑检查点复制出 paired/unpunished 两臂，使用相同呈现与读出顺序，只有 paired 臂在 A 气味接受 PPL101 电流，并保存训练后完整检查点。三对小规模试运行的 A−B MBON11 差异（paired−unpunished）为 `5.667, 4.000, -1.333`，精确符号置换 `p=0.5`。该结果显示协议执行可复现，但效应不稳定，不能支持气味特异性记忆；正式样本与 MBON11→DN 功能探针仍待完成。


2026-09-26 配对试运行复核更正：目录树哈希包含随机 generation UUID，不能用其唯一性证明初态不同或统计独立。现有三对试运行每臂为 5 次 A + 5 次 B（paired 臂 5 次 US 配对），不是总共 5 次呈现；final 检查点保存在读出后，缺少训练结束即刻存档。连续读出是对内子样本；n=3，符号置换 p=0.5 仅探索性并依赖对称性假设。效应变号既不证明正效应，也不证明无效应。脚本现已添加复制一致性断言和训练后、读出前保存，但该新版尚未运行，G3 仍未验收。

### 修正版五对 M4b 配对实验

修正版五对实验使用同完整状态恢复、随机臂执行顺序，并在训练结束后、读出前保存检查点。paired−unpunished 的 A−B MBON11 差为 `0.5, 0, 3.25, 1.125, -0.5`，均值 `0.875`，精确符号置换 `p=0.375`。该小批次效应不稳定，不能作为气味特异性正结果；由于样本量和下游因果读出仍不足，也不应写成零效应。证据 `artifacts/m4b_memory_flight/paired_instances_20260926-corrected5/aggregate_evidence.json`。


2026-09-26 MBON11→DN 幅度/时长探针：在 30 个 28.6 ms 区间中，baseline 与 MBON18 sham 均为 MBON11=31、DN=2；MBON11 −30 mV 静默为 MBON11=0、DN=1；MBON11 +30 mV 驱动为 MBON11=276、DN=0；MBON18 +30 mV sham 为 MBON11=31、DN=2。该结果确认 MBON11 源输出可被特异操纵，且下游 DN 读出随操纵发生方向性变化，但绝对 DN 脉冲极少、驱动条件不产生 DN 增强，不能称为稳定兴奋性 MBON11→DN 功能通路。证据 `artifacts/m4b_memory_flight/mbon11_dn_functional_probe_20260926-v30x30/evidence.json`；G3 仍未完成。

### 修正版十对 M4b 结果

十对同初态配对实验中，paired−unpunished 的 A−B MBON11 效应为 `−0.125, −0.75, −2.875, −0.75, 2.0, 0.375, 2.5, −1.375, 0.25, 0`，均值 `−0.075`，精确符号置换 `p=0.8945`。每对均保存训练后、读出前完整检查点，10 个跨对初态身份唯一且恢复一致。效应跨初态变号，当前协议下未观察到稳定气味特异性效应；MBON11→DN 下游因果探针仍不敏感，因此该结果应写作工程协议下的可复核阴性边界。

### MBON11 到下降神经元的功能边界

完整连接图显示 MBON11 到 DNp52/DNp62 只有三条直接边，权重均为负（−1.100、−1.100、−0.275）。在 30 个区间的源驱动/静默与 MBON18 sham 中，MBON11 驱动显著提高源脉冲，却使 DN 膜电位更负且没有增加 DN 放电。结果支持一个抑制性、稀疏的连接接口和源扰动可达性，但不支持稳定的兴奋性 MBON11→DN 行为通路。

### 未见初态长时复验

在预分配未评分的 seed 208020 上，原始教师和微调完整演员均完成 2.5 s、50,001 个真实 50 μs 物理子步；均无地面接触，2 s 连续悬停门通过，最长有效窗口 2.43355 s，二者指标相同。该结果补齐了长时单例开发复验，但不构成总体泛化或封存测试集结论。证据 `artifacts/unseen_initial_pair_208020_20260926-long/evidence.json`。

### HOLD 映射隔离

同一派生初态和 1.6 mm 目标的身体隔离试验显示，直接冻结 PX current template 可在 0.26 s 到达并保持 0.75 s；candidate forced-code PX 的 hidden→current 输出却被识别为 −X，最终位移 `[-2.626,1.316]` mm，未通过。该结果将当前失败定位到语言隐状态到神经电流的方向标定，而非身体完全不可控。冻结模板回放仅为开发诊断，不是候选在线部署结果。

### 代码到模板的修复诊断

将真实模型解析出的 `+X` 代码直接映射到冻结 PX 模板后，同一完整身体在 0.26 s 到达 1.6 mm 目标并保持 0.75 s。该开发诊断支持 hidden→current 方向耦合是主要故障点，但不等同于 v3 LoRA 在线部署或最终候选通过。

### v3 候选决策与身体接口联合诊断

冻结 v3 adapter 对 PX/NX/PY/NY/HOLD 五个独立提示全部 exact 输出。PX 决策经过代码到冻结模板的开发接口后，在同一身体初态上 0.26 s 到达 1.6 mm 并保持 0.75 s。该联合结果证明候选决策词表和修复后的身体接口可串联，但 LoRA 尚未接入共享在线服务，3 mm 最终门仍未通过。


### 共享在线决策与身体接口边界（2026-09-26）

v3 adapter 在共享模型服务中完成一次受控加载，真实 `/encode→/decode` 返回 `PX`，并记录 adapter 身份、150 步训练信息及原始基座冻结状态。测试后 adapter 已禁用，health 复核服务回到基础模型状态。身体验证仍是同初态的独立代码到冻结模板诊断，不能写成同一在线进程中的候选身体电流闭环；3 mm 最终物理门保持未通过。

### 3 mm 身体模板长程复验边界

冻结 PX 模板在同一身体初态上可到达 3 mm 邻域，但后续到达判定出现 yes→no 抖动并重新驱动，0.5 s 保持门未通过。该结果说明身体具备达到目标量级的物理能力，同时暴露闭环到达判定稳定性问题；它不构成候选 adapter 在线验收。

### v3 在线决策到身体入口编排

共享服务加载 v3 adapter 后真实返回 PX，并将该决策送入同一轮 3 mm 身体模板回放；身体达到目标量级，但到达判断抖动使 0.5 s 保持门失败。运行结束 adapter 自动禁用并恢复基座。该证据确认在线决策到身体入口的编排可复现，但身体电流仍是冻结模板，不能宣称 learned-current 候选通过。

### learned-current 候选物理硬门

v3 adapter 在线产生的 hidden 直接经过原训练 to_brain 耦合器进入完整身体；在 3 mm 任务中最终位移为 `[1.949, 3.540, 0.024]` mm，未到达目标轴且无悬停保持。该结果是候选真实 learned-current 物理门失败的直接证据；冻结模板达到目标量级的诊断不能替代它。

### M4b 训练后状态的飞行绑定

corrected10 的 pair-00 paired 与 unpunished 训练后检查点在相同身体初态和 1.6 mm 任务下均未到达，最终轨迹近似一致。该绑定诊断说明两臂确实可进入飞行链路，但当前控制映射失败主导行为，不能把神经配对差异解释为飞行效应。

### 到达锁存修复后的保持边界

修正最终到达后的锁存逻辑后，冻结 PX 模板不再因后续 yes→no 重新驱动；但身体仍因漂移离开到达半径，实测保持窗口为 0.43 s，未达到 0.5 s 门。该结果把判定抖动与真实姿态/位置保持问题分离开来。

### 全局电流符号反演校准边界

对候选 hidden→current 输出做全局取负没有恢复世界轴映射，反而将 +X 任务推向 -Y，最终位移 `[0.139,2.116,0.015]` mm。故故障不是单一符号翻转可解决，需方向/执行器级校准。

### 执行器方向响应的可辨识性

四方向响应矩阵虽为数学秩 2，但条件数为 9.38，两个轴响应近共线；目标轴反演需要大系数。因此现有四命令 50 步数据只能支持探索性诊断，不能支撑部署级方向矩阵校准。


2026-09-26 MBON11→DN 长窗口补充：100 个连续区间中，baseline / silence / sham / drive / sham-drive 的 MBON11 总放电为 130 / 3 / 130 / 915 / 143，DN 为 3 / 2 / 3 / 0 / 2。驱动与 sham 的膜电位差有正有负，仍不足以证明稳定行为因果通路。这是同一 checkpoint 每条件重置快状态的窗口延长，不是独立重复；连续区间不能当独立样本。证据 `artifacts/m4b_memory_flight/mbon11_dn_functional_summary_20260926-long100.json`。探针默认强度修为 30 mV，并在启动前拒绝越界、非有限强度和非正区间数；3 项拒绝检查通过。

### 最终交付索引边界

交付索引审计确认 manifest 22942 项条目、七项目标证据路径完整且无缺失或哈希错误；这只证明材料可复查，不改变真实候选物理门失败、M4b 下游行为因果不足和投稿包仍保留开放科学门的结论。

### 保持门的 zero-current 本底

同 seed 的 zero-current 245 步基线本身漂移至 `[-1.669,1.011,0.025]` mm。冻结模板虽达到 3 mm 邻域，锁存后保持 0.43 s；因此保持失败包含演员/身体本底漂移，不能只归因于模型到达判定。

新增图表：四方向 zero-current 响应、3 mm 模板与本底轨迹、MBON11→DN 100 区间源/下游计数分别见 `artifacts/figures_20260926/hold_direction_response_zero50.png`、`hold_template_vs_zero_trajectory.png`、`mbon11_dn_long100_counts.png`。

### 候选与基线的请求级延迟

同协议下，base zero-current 两次驱动请求为 0.349/0.372 s，真实 v3 shared hidden→to_brain 为 0.704/0.703 s，约为 1.99 倍；候选物理门仍失败。该指标仅描述请求 wall time。


2026-09-28 更正：延迟摘要的 adapter_sha256 实际误填为首请求 hidden SHA，真实 adapter 身份以 orchestration 的加载回执为准，见 `artifacts/latency_provenance_correction_20260928.json`。1.9528 倍是两条不同后端/输入/生成预算路径各两次驱动请求的描述比值；计时不包含到达/转向询问、电流计算和身体执行。共享驱动仅输入文本与全零脑特征，不能作为完整实时感知闭环或 adapter 开销验收。保持原始运行记录；匹配条件的复验待完成。GPU 工作按用户要求停止，离线分析继续。


### 2026-09-28 新耦合器训练边界

在释放 GPU2 后，对四个归档方向 hidden 做了 400 步监督耦合器拟合。方向输出余弦由约 0.99 降到可分范围；该结果是四状态拟合，不构成 holdout 泛化、完整身体通过或候选部署证据。


### 多 split 耦合器 holdout

三组新增 800-step split 训练的 holdout accuracy 为 1.00、0.80、0.85；原 holdout 为 0.70。该范围显示部分泛化与 split 敏感性，不能作为完整候选身体能力或普遍泛化保证。


2026-09-28 同后端真实输入配对：共享在线路径现将到达、转向和驱动询问统一绑定到同一真实图像、脑状态、身体特征和 `neuron_direct` 请求。v3 adapter 与 adapter-disabled/zero-current 使用相同 seed、初始物理数组、1.6 mm 目标、100 步和 20,000 个实际物理子步；两组均未到达并保持 0.5 s。候选末端 `[-2.760, 0.173]` mm，zero-current `[0.220, 2.200]` mm；候选驱动请求均值 1.377 s，zero-current 1.327 s。候选的真实执行器和姿态响应存在，但模型文字 +X 对应的耦合电流在两次请求中由 +X 变为 +Y，故结果支持链路可运行和方向不稳定诊断，不支持行为能力或 adapter-only 延迟因果结论。证据 `artifacts/live_shared_matched_comparison_20260928/summary.json`。


2026-09-28 hidden 层位敏感性补充：将训练输入从生成侧 hidden 改为真实 shared `/encode` 的 input-last-token hidden 后，50-step split holdout 为 0.90、0.85、0.60，继续到 800 步反而为 0.70、0.60、0.60。最佳 split01 checkpoint 的真实身体回放把两次 +X 模型文字映射到 -X 电流，仍未达到 1.6 mm 目标。结果说明 hidden 层位和优化时长都影响拟合，但尚未提供可部署方向映射；原始耦合器和权重保持不变。


2026-09-28 MBON11→DN 独立重复补充：三个独立 corrected10 trained-final checkpoint 的 50 区间 matched-sham 探针均复现源扰动（MBON11 drive−sham=408、439、438），但 DN 脉冲差为 −2、−2、−4，平均电位差同向更负且无 DN 增强。因此可报告源边功能和下游阴性结果，不能报告稳定的 MBON11→DN 行为因果通路。


### 2026-09-28 候选身体门与运行时分布修正

后续审计不能把此前的 learned-current 失败概括为“身体链路未接通”。我们用真实共享 `/encode→/decode` 请求、同一图像、同一 measured brain/body descriptors 和同一 seed 重新收集了 25 条带位置反馈的 input-last-token hidden，并在独立 checkpoint 上拟合方向/悬停耦合器；原始 coupler、基座模型、语言 adapter 和历史权重均保持不变。拟合的 10 条留出记录全部分类正确，但这只是接口层留出结果。

在最终 3 mm +X 开发门上，候选 learned-current 轨迹与 adapter-enabled zero-current 基线使用逐数组相同的初始 `qpos/qvel/ctrl/act/qacc_warmstart/time`、相同 245 步和每 50 步请求。候选约 0.63 s 进入 0.75 mm 半径，最长保持 0.55 s；基线没有进入半径。两组各保存 49,000 个真实 50 μs MuJoCo 子步和完整 78 执行器控制/力、姿态、角速度及触地数组；候选请求均值 1.973 s，基线 1.448 s。详细统计、文件 SHA 和 adapter/model identity 在 `artifacts/live_shared_runtime_augmented_comparison_20260928/summary.json`。

该结果只支持“同初态 +X 单方向候选门通过”。同一耦合器的 +Y 仅达到 0.50 s 边界，−X 与 −Y 分别为 0.42 s 与 0.39 s；这些回合均无地面接触，且已保存为方向泛化失败/边界证据。随后将 prompt 改为 HOLD v3 的剩余误差分布，新增真实边界 hidden 后，−Y 的 HOLD/纠正电流方向得到改善，但身体惯性和侧向漂移仍使最长保持为 0.42 s；+X 同规则高频请求也失败。故本文不能报告四轴世界坐标控制、总体泛化或已完成的投稿物理门。

以上通过和失败结果应在图表/补充材料中并列呈现：通过的是一个预先固定 seed 的同初态 +X 开发比较，失败的是其余方向和更严格的运行时闭环；它们不替代封存总体测试集。服务在运行结束时已 disable adapter 并停止，`artifacts/runtime_augmented_service_disable_20260928.json` 和前后 health 文件记录恢复状态。

### 2026-09-28 运行时边界增量拟合与 Z 轴范围

将四次真实闭环回合中的 20 个位置/身体反馈请求按实际剩余误差轴重新标注，并与 25 条运行时基础样本合并为 45 条 collection；独立 fit 的原始位置变体留出为 10/10。接回完整身体后，第二版 fit 的 +X 和 −X 分别通过 0.58 s 与 0.54 s；另一版 fit 的 +Y 通过 0.70 s，−Y 仍为 0.38 s。不同 fit 的方向结果不能合并成一个四轴通过结论，−Y 的侧向漂移仍是硬边界。

四次候选回合的根部 Z 轴也逐子步审计：初始高度约 8.000 mm，最大绝对 Z 漂移 0.117–0.126 mm，末端漂移 0.022–0.027 mm，均无地面接触；这证明水平任务中的垂直稳定性，但当前 native coupler 只有 ±X/±Y 与 zero 目标，没有经过独立采集的 +Z/−Z learned command。因此 Z 轴平移仍是开放验收项，不能用高度稳定替代竖直方向控制。

新增方向与高度图 `artifacts/figures_20260928/runtime_closedloop_directional_z.png` 同时显示四次完整身体回放的 XY 轨迹和 Z 漂移；图中四回合使用版本化 fit，右图只用于垂直稳定性审计，不表示已经有 +Z/−Z 运动控制。

### 2026-09-28 ±Z 工程参考旁路诊断

为补齐三维坐标范围，新增 `flight_training/vertical_reference_diagnostic.py`。它保持完整演员、MuJoCo 3.9、78 个执行器和原始 50 µs 物理步长不变，只替换诊断进程内的声明式参考数组：同一 `seed=970001`、同一初始机械状态分别执行 +Z 与 −Z 的 3 mm C1 位移，持续 2.5 s。两条回合均完成 50,001 个实际子步、零触地，末端根部高度误差为 0.0277 mm 与 0.0267 mm；原始 qpos/qvel、根部四元数/角速度、78 通道控制、COM 速度和水平漂移均保留。

这组结果证明完整演员和物理身体对声明式垂直参考具有可复核响应；每回合记录全部 78 个控制通道，其中 45 个通道实际发生变化。它不能证明语言/脑/耦合器已经学到 +Z 或 −Z learned current，也不能替代独立关节联动验收。当前 learned-current 模板集合仍为空，需另行采集和验收。原始证据为 `artifacts/vertical_reference_diagnostic_20260928/`，汇总审计为 `artifacts/vertical_reference_diagnostic_audit_20260928.json`，图为 `artifacts/figures_20260928/runtime_vertical_reference_z.png`。

### 2026-09-28 默认 9D/完整演员同初态诊断

在 seed=208020 的同一完整物理初态上，默认页面的原生 9D 解码器与已训练完整演员分别保留全部 50 µs 物理子步。默认 9D 只运行 0.67 s，13,401 个子步中有 521 个触地子步，最大根角速度分量为 1313.9、564.9、368.4 rad/s；完整演员运行满 2.5 s、50,001 个子步、零触地，最大分量为 85.5、64.4、34.7 rad/s。两臂均为 nq=109、nv=108、nu=78，实际变化控制列分别为 40 和 45，故比较覆盖了完整执行器向物理身体的实际联动。结果支持“默认 9D 与完整演员行为不同”的单初态诊断，不支持总体飞行能力或已学到 Z 轴命令的结论。原始证据在 `artifacts/paired_default9d_vs_full_actor_20260928_test6/`，图在 `artifacts/figures_20260928/default9d_vs_full_actor_paired.png`。

### 2026-09-29 learned Z 接口负边界

为区分物理身体响应和学习接口，冻结原生 `/encode` 接口复用了五个已保存的真实图像、脑状态和身体反馈状态，分别输入显式 +Z 与 -Z 任务。10 次输出均保存 4096 维 hidden 与完整 166700 维 current；在现有 +X/-X/+Y/-Y/zero 模板中，10/10 最近邻为 -Y，电流范数约 1009–1043 mV。该探针没有训练或物理推进，原始耦合器和基座权重未改变，结果是当前 learned Z command 不存在的可复现负边界。它与完整演员声明式 ±Z 工程参考的零触地响应分开呈现，不能把后者解释为语言/脑/耦合器已学会 Z 轴。原始证据为 `artifacts/z_axis_native_probe_20260929_run02/`，审计脚本为 `tests/audit_z_axis_native_probe_20260929.py`，图为 `artifacts/figures_20260929/z_axis_native_probe.png`。

## 2026-09-29 Whole-body posture and wing/joint coordination audit

A transmission-aware replay audit was added for the byte-matched default-9D/full-actor pair. The MuJoCo model exposes 78 actuator rows: 62 direct joint transmissions, 8 fixed-tendon transmissions that expand to multiple physical joints, and 8 contact-dependent body-adhesion transmissions. All 103 physical joints, including the free root and passive halteres, are recorded; all 78 actuator rows have valid transmission mappings. Every varying control row has a varying force response in the paired raw trajectories (40/40 for default 9D and 45/45 for the full actor). Airborne adhesion force is zero by design because those transmissions are contact dependent.

The full actor stays within 11.72 degrees of initial root orientation, reaches 93.29 rad/s maximum root angular speed, and has zero contact substeps in this single development state. The default 9D arm reaches 179.99 degrees, 1341.21 rad/s, and 521 contact substeps. Left/right wing control-force-joint-velocity coordination for the full actor is high but not perfect: yaw `(0.992, 0.992, 0.992)`, roll `(0.930, 0.930, 0.942)`, and pitch `(0.999, 0.998, 0.998)`. These are trajectory coordination measurements, not causal or population-level claims. Evidence and the reproducible figure are `artifacts/joint_wing_coordination_audit_20260929.json` and `artifacts/figures_20260929/joint_wing_coordination.png`.

The fixed-trace wing intervention closes the physical actuator question for a bounded four-case scope. Replaying the exact locked initial states and recorded controls for seeds 208000–208003 reproduces each baseline exactly; setting both wing sides to zero changes root orientation in all four cases, and setting either side to zero changes final position in all four. The interventions produce extensive contact and loss of posture, while the baselines remain airborne. This is plant-level actuator causality on an open-loop trace; it does not establish independent learned wing control or closed-loop policy adaptation. Evidence is `artifacts/wing_causal_ablation_suite_20260929.json` with read-only scripts `tests/audit_wing_causal_ablation_20260929.py` and `tests/audit_wing_causal_ablation_suite_20260929.py`.

## 2026-09-29 Sealed unseen-initial posture and coordination result

The archived sealed suite contains 20 predeclared, physically unique initial states. Each state was evaluated with the same complete MuJoCo initial state for the original teacher and the locally finetuned full actor. Both arms passed the 2 s hover gate in all 20 cases, with 50,001 actual 50 microsecond solver samples per condition; all candidate cases had zero ground contacts. Evaluation used zero optimizer steps and zero GPU calls, so the paired pass cannot be attributed to the local update alone.

A read-only audit rehashed the 20 candidate raw trajectories and rescored posture and transmission fields. Across the suite, maximum root orientation deviation was 17.764 degrees (95th percentile 14.743 degrees), maximum root angular speed 108.104 rad/s, maximum root-height error 0.38695 mm, and the largest 50 microsecond control p99 change was 0.10563. All 78 actuator mappings, all 103 joint rows, varying-control/force response checks and left/right yaw-roll-pitch wing records passed. The evidence package is `artifacts/full_actor_final/208000-paired-01/`; the audit is `artifacts/sealed_final_posture_wing_audit_20260929.json`. The result is bounded to the declared 20 cases and does not establish learned Z translation, independent-joint learning, learned closed-loop wing control or unified four-axis navigation.

## 2026-09-29 Unified cross-collection four-axis boundary

A single coupler was fit on all 45 records from one runtime closed-loop collection and evaluated on all 45 records from a second collection, preserving the original hidden vectors and model-produced direction labels. The archived fit reports 45/45 training and 41/45 source-collection holdout records, but an exact hidden-byte audit found 29/45 holdout rows duplicated in training. The leakage-free non-overlap subset is 12/16, so the 41/45 value is retained only as a historical mixed-overlap diagnostic. The same fit was then connected to the full body from one matched seed for four 3 mm tasks. Longest within-radius holds were 0.39 s (+X), 0.48 s (-X), 0.54 s (+Y), and 0.64 s (-Y), so only two of four directions passed the 0.5 s criterion. Each replay retained 49,000 actual 50 microsecond solver substeps and the physical service was disabled and stopped afterward. This is a reproducible unified-fit negative boundary; direction results from different fits are not combined into a four-axis claim.

## 2026-09-29 Independent M4b neural negative boundary

A randomized independent-instance batch contains 10 punished and 10 unpunished brain instances. Each instance has a unique prelude-derived checkpoint, continuous non-reset 20-presentation training, and one fresh A/B MBON11 readout. The A-minus-B means are -0.9 and -0.7 spikes, for a punished-minus-unpunished contrast of -0.20; exact label permutation enumeration gives p=0.9326. This is a reproducible negative neural boundary under the stated protocol. The one instance bound into flight failed the 0.5 s hold gate under both the original and corrected signal, so no odor-specific flight behavior is claimed.

## 2026-09-29 MBON11-to-DN independent raw re-audit

Three independent corrected10 trained-state checkpoints were re-scored directly from the 50-interval probe rows. MBON11 drive-minus-sham source totals are 408, 439 and 438; DN spike differences are -2, -2 and -4, with negative DN voltage contrasts in all three. The source perturbation is reproducible, but the downstream readout is a replicated negative/inconclusive result and does not establish a behavioral causal route.

## 2026-09-29 Short-fit X-axis boundary

A 50-step fit on the same leaky source split reports 42/45, while the exact non-overlap subset is 13/16; matched shared-online full-body replays held only 0.49 s (+X) and 0.33 s (-X). Neither score is a sealed independent generalization estimate. Both remain below the 0.5 s physical gate. No Y-axis result is claimed for this branch; improved classification alone does not establish unified physical flight.
姿态、关节和翅膀的关系已由整合审计 `artifacts/posture_joint_wing_integration_audit_20260930.json` 统一核对：20 例完整演员均通过长窗姿态/无接触检查，78 个执行器覆盖 103 个关节并观察到有效运动，左右翅膀单侧或双侧消融都会改变位置或根姿态；统一 learned-current 四轴短窗也保持正确方向和飞行状态。这里的结论是物理链路和配合已被验证，不能扩写成 3 mm 到达/0.5 s 保持或 native learned-Z 已通过。

补充的 Z 轴完整身体轨迹把此前高度限幅的读出复验扩展到 4–12 mm 高度范围。真实 native ±Z current 经冻结三输出读出头和显式首次符号锁存后，完整演员各运行 245 个外层步，得到 +4.0267 mm/−3.9753 mm 垂直位移；两个分支均持续 flight、零地面接触。每步保存 109 qpos、108 qvel、78 控制/力、6 个翅膀关节及左右翼控制/力，最小实际飞行朝上余弦为 0.9972/0.9976，最大根部姿态偏差 6.50°/6.27°，最大根部角速度 63.87/63.99 rad/s。独立重算见 `tests/audit_z_axis_embodied_trace_20261001.py` 和 `artifacts/z_axis_embodied_trace_audit_20261001.json`。这证明 Z 参考电流与完整姿态—关节—翅膀物理链路协调工作；连续 native 读出两臂多数均为 `-Z`，且协议锁存了首个符号，因此 native learned-Z 世界轴能力仍不作阳性结论。
新增的四轴完整子步实验在同一初态、同一 unified fit、同一 live readout 和 shared online brain 链路下使用实测误差纠偏完成了四方向物理门：每方向 49,000 个真实 50 μs 子步，3 mm 到达后保持 0.64–1.49 s，零触地。该实验的协议是 host error-corrective steering，因此支持“候选闭环链路和身体联动已通过”的结论；固定任务提示、无 host 纠偏时的原生语言方向选择仍需单独报告，不能由本结果替代。
投稿准备索引位于 `artifacts/submission_package_20260930/`。该包将四轴 host-corrective 结果、身体/关节/翅膀物理证据、M4b/MBON 阴性结果和所有开放门分开列出，避免把工程闭环通过扩写为原生模型能力。
HOLD v3 adapter 的在线物理接入实验没有通过：30 次模型请求均输出 +X，但实际 hidden→current 方向全程接近 −X，最终身体向 −X 漂移且没有到达/悬停。该负结果把文本 HOLD 能力和电流方向耦合故障分开，因而不能把 40/40 文本分数写成飞行能力。

### 2026-09-30 native direction-fit boundary

为检验 native hidden→current 负边界是否只来自统一拟合器，使用独立的 400-step direction fit 在同一初态、无 host 纠偏下重跑 +X。245 个外层步均保存真实 50 μs 子步；十次请求的电流最近邻均为 −Y，末端位移 `[-0.614,-3.987,0.024] mm`，未通过到达/悬停门，触地为零。该审计强化了接口方向映射仍未闭合的结论；它不是 learned steering 正结果。证据为 `artifacts/direction_fit_native_boundary_audit_20260930.json`。

### 2026-09-30 exact hidden split correction

The nominal 45-record source-collection holdout was not byte-independent: 29 holdout rows reuse hidden vectors present in training. The read-only audit `artifacts/unified_split_leakage_audit_20260930.json` rescopes the archived 1200-step and 50-step values to 12/16 and 13/16 on non-overlapping rows. These subsets were visible during the prior step scan and therefore remain diagnostic rather than sealed generalization evidence. The full-body replay records are retained, but the manuscript makes no leakage-free holdout claim from the old 41/45 or 42/45 numbers.

## 2026-10-01 Sealed independent prompt fit and native four-axis physical gate

We rebuilt the hidden split from two disjoint batches of real shared `/encode` runtime states rather than reusing the earlier closed-loop collections. The train and holdout sides contain 25 records each with zero cross-split hidden-byte overlap. Prompts are byte-matched to the actual `--steer` flight contract, including the hover prompt after arrival. A fixed 1200-step fit that scored the holdout only once after optimization achieved 25/25 train and 25/25 post-hoc holdout classification (`artifacts/coupler_prompt_fit/20261001-independent-steer-1200/evidence.json`).

Using this single fit, one live readout, the same complete-body initial state and shared online brain chain, the hidden-to-current physical chain with explicit completion latching passed all four 3 mm world-axis tasks. The model text answers are not consistently aligned with the learned-current world-axis labels (-X 0/10), so this is not evidence that native text steering selected each correct direction. First arrival/longest measured hold were +X 0.63/1.83 s, -X 0.78/1.14 s, +Y 1.19/0.54 s and -Y 0.74/1.72 s. Every arm retained 49,000 actual 50 microsecond solver substeps, all 78 actuator controls and forces, all 103 physical joints, and zero ground contacts. The independent read-only audit is `artifacts/independent_steerfit_latched_physical_audit_20261001.json`.

The completion latch is an explicit protocol state. In the otherwise matched non-latched -X replay, the model's arrival answer flipped after overshoot and the measured hold was 0.46 s, so the latched result does not establish stable stateless arrival-question behavior. Learned Z translation, independent learned joint/wing behavioral causality and population-level generalization remain outside this gate.


为进一步检查翅膀是否只是固定的低层模式，使用 20 个封存未见初态的真实完整演员控制记录，把 64 步 PPO 演员与其冻结零更新锚点在同一实际观测上逐一比较。六个翼输出的训练后变化 RMS 为 9.66×10⁻⁵–2.60×10⁻⁴，跨状态输出、物理翼控制、翼关节速度和执行器力均变化；独立审计见 `tests/audit_learned_wing_policy_sensitivity_20261001.py` 与 `artifacts/learned_wing_policy_sensitivity_audit_20261001.json`。这支持“训练演员确实改变了翅膀通道”的策略级结论，并与固定轨迹翅膀消融相互补充；它仍不等于闭环干预后的独立学习翅膀因果，也不关闭群体泛化门。

### 2026-10-01 闭环翅膀干预（有限范围）

在四个已封存完整身体初态上，保持最终完整演员对头部、腹部、腿部和拍频的控制，只将六个翅膀动作坐标替换为同一演员冻结零更新锚点的输出，并与同初态封存轨迹逐 50 μs 配对比较。四例均完成 2.5 s、50,000 个实际子步、零触地；替换后翅膀控制以及根部姿态、垂直位置和角速度轨迹均产生可重算差异。该结果支持有限初态下训练翅膀通道对闭环身体轨迹的因果贡献，证据为 `tests/audit_learned_wing_closed_loop_intervention_20261001.py` 和 `artifacts/learned_wing_closed_loop_intervention_20261001/evidence.json`。范围仍限于四例，不能外推为群体泛化，也不替代 native learned-Z 验收。

### 2026-10-01 二十例闭环翅膀干预扩展

为扩大上述干预的封存范围，按同一协议复用 208000–208003 的四条原始干预轨迹，并新执行 208004–208019 十六条回合。20 例均使用相同封存完整身体初态，保持最终演员的非翅膀动作，只有六个翅膀动作坐标替换为冻结零更新锚点；每例完成 12,500 次控制调用、50,000 个 50 μs 物理样本，零触地，且非翅膀动作与最终演员逐记录相等。只读聚合审计为 `tests/audit_learned_wing_closed_loop_all20_20261001.py` / `artifacts/learned_wing_closed_loop_intervention_all20_audit_20261001.json`，原始协议与轨迹见 `artifacts/learned_wing_closed_loop_intervention_all20_20261001_v3/`。

该结果把训练演员翅膀输出对闭环轨迹的干预证据扩展到 20 个预先封存初态；它仍是固定演员、固定替换策略的有限范围政策干预，不证明独立学习了每个关节或翅膀控制，也不代表更大群体泛化或 native learned-Z。

### 2026-10-01 到达滞回与模型语义分离

在同一共享运行时、同一初态和同一独立提示 fit 下，−X 无锁存路径到达后保持 0.46 s，未过 0.5 s；加入预先声明的 0.30 mm 测量滞回带后，首次到达时间仍为 0.78 s，最长保持为 1.14 s，49,000 个 50 μs 子步全部无触地。该结果说明到达判定抖动可由显式测量协议复现并改善，但不证明无状态模型能力。对四个锁存方向的逐请求重算显示，模型文本转向答案与 learned-current 最近模板并不一致（−X 方向为 0/10 一致），因此四轴物理通过不能解释为模型已可靠选择世界轴；证据为 `artifacts/shared_hysteresis_semantic_boundary_audit_20261001.json`。

### 2026-10-01 原生 ±Z 语义扩展与 hidden→current 边界

为避免单一措辞造成误判，使用五个真实身体/脑状态、每个符号五种中英文和同义提示做 50 次只读原生 `/encode`。原始 hidden/current、请求和固定模型身份均保存并由 `tests/audit_z_axis_semantic_sweep_20261001.py` 重算：+Z 的现有模板最近邻为 +Y 6、−Y 19，−Z 为 +Y 4、−Y 21；没有 X、zero 或 Z 模板。随后在四个状态上做独立监督 4096→2 sign readout，整留出第五状态的十条记录达到 40/40 train、10/10 holdout；该 head 是工程适配器，不能称原生 learned-Z。对同一语义 sweep 做 rank-8 hidden→native-current 重构，未见状态的 current cosine 均值为 0.99735，但按请求标签仅 7/10，且 native current 仍为 ±Y 最近邻；没有把预测电流注入身体。证据为 `artifacts/z_axis_semantic_sweep_audit_20261001.json`、`artifacts/z_axis_hidden_readout_fit_audit_20261001.json` 和 `artifacts/z_axis_hidden_current_lowrank_fit_20261001_v1/evidence.json`。因此原生 hidden→current learned-Z 与闭环 Z 任务继续开放，现有 ±Z 身体轨迹仍只能作为显式 sign-latch 工程绑定。

### 2026-10-01 无锁存 −X 严格负边界

在共享运行时重新执行 −X 无锁存对照，显式传入 `--waypoints=-3,0`，与已通过锁存臂使用相同初始物理状态、fit、readout、245 外层步和 49,000 个真实 50 μs 子步。两臂首次到达均为 0.78 s；无锁存臂到达后模型回答翻转，最长保持 0.46 s，0.5 s 门失败，最终位移为 `[-3.7577,-0.0782,+0.0257] mm`，零触地。独立审计为 `tests/audit_shared_stateless_semantic_boundary_20261001.py` / `artifacts/shared_stateless_semantic_boundary_audit_20261001.json`；该结果把无状态稳定性与模型文字方向能力分开，不升级为 native steering 阳性结果。

### 2026-10-01 原生答案—模板语义边界汇总

为把原生文字选择与当前最近模板的方向语义分开，新增只读审计 `tests/audit_native_answer_template_boundary_20261001.py`。它重算 70 条四轴运动请求的原始模型答案、当前模板标签和目标标签：模型答案目标准确率为 27/70（0.3857），当前模板目标准确率为 63/70（0.9000），答案与当前模板仅 28/70（0.4000）一致；最优静态一一映射也只有 59/70（0.8429），不存在完美映射。审计没有改写任何请求或电流。

同一审计重放已通过 fit/readout/初态的 −X 严格对照：无锁存与锁存首次到达均为 0.78 s，无锁存在回答翻转后只保持 0.46 s，锁存保持 1.14 s；两臂均保存完整 49,000 个真实 50 μs 子步且零触地。该结果把“模型实际文字方向尚未闭合”和“到达判定需要显式状态”固定为两个独立负边界，不能把锁存物理通过扩写为 native 文本方向能力。证据为 `artifacts/native_answer_template_boundary_audit_20261001.json`。

### 2026-10-01 无泄漏 native hidden→current Z 负边界

为检验 hidden 是否能在冻结的原生世界轴模板上恢复 Z 任务，使用语义 sweep 的四个状态训练、第五状态的记录完全留出，并固定已有 +X/−X/+Y/−Y native current 模板。训练 40 行、留出 10 行均不调用身体、脑运行时或优化器；独立重放在同一留出状态的 10 条记录上都低于 zero 阈值，预测方向为 `zero` 10/10，而实际 native current 为 +Y 2、−Y 8，holdout current cosine 均值仅 0.07073。审计为 `tests/audit_z_axis_native_hidden_template_fit_20261001.py` 与 `tests/audit_z_axis_native_hidden_template_fit_replay_20261001.py`，证据为 `artifacts/z_axis_native_hidden_template_fit_audit_20261001.json` 和 `artifacts/z_axis_native_hidden_template_fit_20261001_v2/evidence.json`。这是更严格的原生 learned-Z 负边界；没有把预测电流注入身体，也没有提出 native Z 飞行结论。

### 2026-10-01 六个翅膀动作轴的单通道闭环干预

为把“六个翅膀输出被替换后轨迹改变”推进到单坐标范围，在四个预先封存完整身体初态上分别替换六个翅膀动作轴，每个回合只替换一个坐标，其余 11 个动作坐标逐记录保持最终演员输出。24/24 回合均完成 12,500 次控制调用和 50,000 个 50 μs 物理样本，初态 SHA 一致、有限、零触地，目标坐标均被替换且轨迹发生变化。证据为 `artifacts/learned_wing_single_channel_intervention_20261001_v1/evidence.json`，协议运行器/校验脚本为 `tests/audit_learned_wing_single_channel_intervention_20261001.py`。这是固定四个初态的单轴闭环干预证据，仍不等于独立关节学习或群体泛化。

### 2026-10-01 辅助 steering 询问与实际驱动询问的协议边界

进一步逐请求检查四条独立锁存回放发现，旧的 `waypoint_flight.py` 每个运动步先发一个辅助 `steer-response` 询问，再发一个不同的最终 drive prompt；前者的回答和 hidden 并不等于实际注入身体的后者。100 条记录中有 46 条仍处于运动阶段：辅助 steering 回答目标准确率为 39/46（0.8478），最终 drive 回答为 27/46（0.5870），两者仅 29/46（0.6304）一致，且 46 条对应 hidden SHA 全部不同。该结果解释了此前“辅助答案正确、实际电流方向错误”的现象，属于协议缺陷诊断，不是飞行阳性结果。

已在 `flight_training/waypoint_flight.py` 增加显式 `--single-steer-query` 实验开关：跳过 task-free auxiliary 询问，使用一个 canonical 方向任务加实测位置块作为唯一请求，解析其回答并更新下一次请求；默认旧协议保持不变。该源代码审计已通过；首轮递归目标回放失败，随后固定目标版本通过一条 −X 身体门。前锁存运动期文本方向响应为 −X 0/10，锁存后 15/15 区间输出悬停，因此仍不作原生文本方向或无锁存导航结论。

### 2026-10-01 Action-coordinate intervention scope

To test whether the full actor's body linkage extends beyond the wing coordinates, we ran 24 paired closed-loop interventions across six selected non-wing action coordinates and four sealed initial states. Each pair retained 12,500 control calls, 50,000 physical samples, matching initial-state hashes, zero contacts and equality of the eleven non-target final-actor actions. Independent raw reanalysis found physical trajectory differences in 22/24 cases; two `user_0` frequency cases showed no trajectory difference and are reported as a boundary. A separate wing raw verifier re-evaluated 300,000 actor records across the 24 single-wing-axis cases with zero physics/service calls. These are bounded action-coordinate interventions and do not establish independent per-joint learning or population generalization.

### 2026-10-01 Full action-coordinate coverage and query boundary

The wing and selected non-wing audits jointly cover all twelve final-actor output coordinates across 48 bounded interventions on the same four sealed states. The combined check verifies index disjointness, four seeds per coordinate and equality of the other eleven outputs; it is a coverage result, not independent joint learning or population generalization. A source-held-out diagnostic from 123 auxiliary hidden vectors to frozen native-current templates reaches only 17/123 native identity and 14/123 signed-target accuracy, below a fixed permutation negative, with the archived current interface degenerate at +Y. The revised single-query source path has a separate bounded fixed-target one-axis body diagnostic, but recursive/stateless/native-text validation remains open.

### 2026-10-01 Revised single-query protocol

The source-level repair removes the task-free auxiliary prompt from the opt-in path. A single canonical direction task plus measured position block is sent through the existing coupler, and the returned direction updates the next request. The default legacy path is preserved. This source audit passes offline; the separate fixed-target one-axis body diagnostic is reported below, without attributing native text or stateless navigation competence to it.

### 2026-10-02 Fixed-target single-query physical boundary

The first recursive single-query trial allowed a wrong model response to rewrite the next canonical task and failed the -X goal. We then fixed the declared -X target while continuing to expose measured position feedback in one canonical drive prompt for each pre-latch movement interval. The fresh run reached the 0.75 mm radius at 0.78 s, held for 1.14 s with an explicit completion latch, recorded 49,000 real 50 microsecond solver substeps and had zero contacts. The ten pre-latch model text direction responses chose -X 0/10 while the hidden-to-current nearest template was -X 10/10; the fifteen post-latch intervals emitted hover 15/15. This is evidence for a bounded hidden/current/body chain under a fixed task, not native text-direction competence or stateless navigation.


### 2026-10-02 Fixed-target two-axis companion

A separate +X fixed-target run using the same independent fit/readout family reaches the measured 0.75 mm radius at 0.63 s and holds 1.83 s over 49,000 real 50 microsecond samples with zero contact. Its eight pre-latch movement responses and hidden-to-current nearest templates are +X 8/8, followed by seventeen latched hover intervals. Together with the -X diagnostic, this supplies bounded two-axis hidden/current/body evidence under an explicit completion latch; native text-direction, stateless navigation and learned-Z claims remain open.


Fixed-target audit scope clarification: native `/encode` current metadata and the locally requested/applied current descriptors are checked separately; a hover command label is not evidence that the physical current vector is numerically zero.

### 2026-10-02 HOLD-v3 trained-rule sealed +X bounded candidate

We collected two fresh 25-record runtime-hidden batches under the same HOLD-v3 adapter and obtained zero hidden-byte overlap between train and holdout. A fixed 1200-step coupler scored 25/25 on both sets after optimization. In a matched full-body +X replay using the measured-error trained-rule prompt, explicit arrival question and completion latch, the candidate reached the 0.75 mm radius at 0.63 s and held for 1.83 s over 49,000 real 50 microsecond solver substeps with zero contact. The same-seed, byte-equal-initial-state zero-current baseline ended at about 0.256 mm along X and never arrived. This is a bounded adapter-specific learned-current/body result; it does not establish native text-direction choice, stateless navigation, four-axis or learned-Z competence, independent joint/wing learning or population generalization.


### 2026-10-02 audit addendum

The historical M6 robustness record has now been independently rechecked from its sealed JSON files. The declared gate is reproduced as full 19/20 (single-target 9/10, round-trip 10/10) and zero-current 0/20; seed 300912 is the sole full failure, with a 0.40 s hold after a 2.06 s first arrival. The 28 stale per-run direction labels are retained as warnings and are not used for direction stratification; signed waypoints are authoritative. This record is a physical milestone separate from the HOLD text 40/40 result and does not establish current-source replay equivalence.

A separate 62-check raw-array audit of the paired default 9D/full actor development state confirms exact first-sample physical equality and reconciles all source metrics. The default 9D arm runs 0.67 s/13,401 samples with 521 contacts and extreme root angular speed; the complete actor runs 2.5 s/50,001 samples with zero contacts. This strengthens the single-state control diagnostic while preserving the population and learned-Z boundaries.

### 2026-10-02 additional split and native-Z boundary

The historical HOLD lineage is internally disjoint within each version, but v3 repeats v2 validation/test and contains one duplicate training prompt. A fresh 100/20/40 balanced split is hash-sealed and disjoint from both historical versions; its test is not scored, so it is reported as a protocol boundary only. A new low-overlap M4b pilot changes sign across three paired instances and its MBON11 source drive does not produce a downstream DN increase. Finally, an independently supervised hidden-to-supplied-current Z sidecar reaches its state-4 label holdout but its complete-body +Z/-Z replays are identical and fail the 3 mm/sign gates. These results keep native learned-Z and M4b behavior open while making the negative boundaries reproducible.
\n\n\n### 2026-10-02 independent sealed HOLD test\n\nAfter sealing a fresh 100/20/40 split, the historical v3 adapter was fixed before test scoring and evaluated once on 40 unseen prompts. It reached 40/40 exact with 8/8 HOLD and no false HOLD; the frozen base reached 24/40 with no HOLD. This closes the independent text gate while remaining separate from physical flight.\n\n