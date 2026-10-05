# CyberFly-01 发布文案

## 标题

CyberFly-01：第一支赛博果蝇

## 正文

我们发布 CyberFly-01。

它把 MiniCPM-o 4.5 的多模态模型、成年雄性果蝇的真实连接组数据、可复现的神经动力学仿真和 MuJoCo 身体物理，放进同一条研究链路。

输入一段文字、图像或音频。模型产生隐藏状态，隐藏状态进入 17,336 个感觉入口，经过 166,700 个神经元和 25,582,938 条聚合有向边，再从 2,129 个运动与下行出口进入身体读出。

最终我们看到的不是一张静态图片，而是关节、翅膀、接触、速度和身体姿态的变化。

这不是把活体大脑搬进电脑，也不是意识证明。它是一个可以记录、干预、重放和对照的工程系统：模型信号、神经状态和身体运动被放在同一条可检查路径上。

CyberFly-01 的公开模型是 baked 版本。LoRA 已经合并进 MiniCPM-o 4.5，发布仓库不包含独立 adapter。普通运行代码、模型卡和技术报告同步公开。

我们希望它开启的不是“机器是否有意识”的猜谜，而是一个更具体的问题：

当语言、神经计算和身体物理真正连接起来时，我们能否逐层看见一个行为是如何产生的？

这就是第一支赛博果蝇。

## 发布链接

- 模型：<https://huggingface.co/GizzAI/CyberFly-01>
- 模型：<https://modelscope.cn/models/GizzAI/CyberFly-01>
- 运行代码：<https://github.com/wzsxb233/cyberfly>
- 技术报告：<https://github.com/wzsxb233/cyberfly/blob/main/TECH_REPORT.md>
- 模型卡：<https://github.com/wzsxb233/cyberfly/blob/main/MODEL_CARD.md>
- 论文 draft：<https://github.com/wzsxb233/cyberfly/blob/main/FRUITFLY_RESEARCH.md>

## 视频

- 主宣传片：`artifacts/video_demo/cyberfly_system_master_promo.mp4`
- 完整能力演示：`artifacts/video_demo/cyberfly_full_capability_demo.mp4`
- 内部运算演示：`artifacts/video_demo/cyberfly_internal_computation_release.mp4`

## 说明

模型仓只发布 baked 权重、配置、tokenizer、模型卡、技术报告和校验文件；不发布独立 LoRA、训练代码、训练数据或私有实验资产。
