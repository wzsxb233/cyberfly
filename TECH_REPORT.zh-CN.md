# CyberFly-01 Technical Report / 技术报告

## Abstract / 摘要

CyberFly-01 is an auditable embodied-AI interface connecting MiniCPM-o 4.5, the MaleCNS connectome simulation and a MuJoCo fly body.

CyberFly-01 是一个可审计的具身智能接口，将 MiniCPM-o 4.5、MaleCNS 连接组仿真和 MuJoCo 果蝇身体连接起来。

The system keeps 166,700 neurons, 25,582,938 aggregate directed edges, 17,336 sensory entries and 2,129 motor or descending outputs. The body exposes 78 actuators, 103 physical joints and six wing axes.

系统保留 166,700 个神经元、25,582,938 条聚合有向边、17,336 个感觉入口和 2,129 个运动或下行出口。身体包含 78 个执行器、103 个物理关节和 6 个翼轴。

## Method / 方法

Text, image or audio enters MiniCPM-o as a measurable hidden state. A declared engineering projection routes the signal into sensory IDs, through the connectome simulation, and into an explicit body readout.

文字、图像或音频进入 MiniCPM-o，形成可测量的隐藏状态。一个明确声明的工程投影将信号送入感觉 ID，经过连接组仿真，再进入身体读出。

This interface is observable, intervenable and replayable. A numeric influence through a port is not evidence of subjective thought, consciousness or biological equivalence.

这条接口可观察、可干预、可重放。端口中的数值影响不能证明主观思想、意识或生物等价性。

## Baked release / Baked 发布

CyberFly v3 used a short language-only update: 150 supervised steps, 420 training examples, 20 training-evaluation examples and 1,916,928 trainable parameters. The language LoRA is merged into the MiniCPM language backbone. No standalone adapter is published.

CyberFly v3 使用了短程语言更新：150 个监督步骤、420 个训练样本、20 个训练评估样本和 1,916,928 个可训练参数。语言 LoRA 已合并进 MiniCPM 语言主干，公开发布不包含独立 adapter。

## Evidence boundaries / 证据边界

The connectome graph and declared ports can be loaded, stepped and audited. The body exposes condition-dependent joint, wing and attitude trajectories under declared protocols.

连接组图和声明的端口可以加载、运行和审计。身体可以在声明协议下产生条件相关的关节、翅膀和姿态轨迹。

Text holdout scores are text checks, not flight scores. Native ±Z text requests currently collapse into existing templates. Historical M5/M6 numbers remain protocol-scoped. A combined demo does not prove that one native LLM call drove every displayed body clip.

文本 holdout 分数只是文本检查，不是飞行分数。原生 ±Z 文字请求目前会落入已有模板。历史 M5/M6 数字只能按其协议解释。合成演示不能证明一次原生 LLM 调用直接驱动了视频中的每个身体片段。

## Meaning and limits / 意义与限制

CyberFly-01 is a research instrument for inspecting where behavior changes across model, neural simulation and body physics. It is not a consciousness claim, a wet-brain reconstruction or unrestricted native language flight.

CyberFly-01 是一个研究工具，用来检查行为如何跨越模型、神经仿真和身体物理发生变化。它不是意识证明、活体湿脑重建，也不是不受限制的原生语言飞行系统。

See [TECH_REPORT.md](TECH_REPORT.md) for the full English report and [PAPER_DRAFT.md](PAPER_DRAFT.md) for the paper draft.

完整英文报告见 [TECH_REPORT.md](TECH_REPORT.md)，论文草稿见 [PAPER_DRAFT.md](PAPER_DRAFT.md)。
