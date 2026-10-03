# 赛博果蝇：开源项目、真实强化学习与可实施路线

核验日期：2026-09-13。以下区分上游已实现的事实、源码审读结论和本项目的工程建议。下载源码、跑通仿真、发生参数更新、评估显示学会任务，是四个不同的里程碑。

## 结论

可以用公开身体模型、真正的强化学习训练器和 MiniCPM-o 组合成可训练的赛博果蝇。推荐先把 **FlyGym / MuJoCo 身体、明确的任务奖励、PPO 参数更新、训练存档和独立评估**做完整，再把真实连接图接成另一种可训练控制器。MiniCPM-o 负责多模态交互和将命令转换为任务；不能用它的回答或脚本动作冒充强化学习结果。

这条路线的第一版训练的是控制策略。除非真实脑图进入了该策略的计算、参数更新与评估路径，否则不能称为“在训练完整果蝇大脑”。此处是工程建议，不是上游已经完成的产品。

## 网上爆火的项目分别是什么

| 项目 | 已核实事实 | 可获取状态与适合用途 |
| --- | --- | --- |
| MaleCNS v1.0 | 成年雄性果蝇的脑、视叶和腹神经索连接图，包含连接强度、细胞注释、神经递质预测及形态数据 | 数据开放，CC BY 4.0；真实连接图来源，不是训练好的智能体 |
| DOOMFLY | 本轮 Doom 演示的公开源码；连接图经近似神经动力学和人工输入输出接口控制游戏 | 可拉，原创代码 MIT；适合连接图实验参考，当前未验证学会生存 |
| Fly64 | Jessica Paquette 的 MaleCNS 接 Mario64 实验，包含视觉输入、模型和观测台 | 可拉；作者仅测试一台 M2、16 GB Mac，游戏资产需自备 |
| Eon fly-brain | 基于 FlyWire 雌性脑的 LIF 仿真、多后端运行与性能比较 | 可拉，GPL-2.0；公开库不是现成完整宠物 |
| FlyBody | Google DeepMind / HHMI Janelia 的 MuJoCo 果蝇身体与行走、飞行 RL 环境 | 可拉，Apache-2.0；身体与运动训练基础 |
| NeuroMechFly / FlyGym | EPFL 的身体、感知、地形和控制框架 | 可拉，Apache-2.0；建议作为交互身体与训练环境底座 |
| FlyGM | 把果蝇连接拓扑构造成图神经网络，研究模仿学习与强化学习身体控制 | 官方仍标注 Code (Coming soon)；目前不能作为已开源可复现依赖 |

来源：[MaleCNS 下载](https://male-cns.janelia.org/download/)、[DOOMFLY](https://github.com/nftechie/doomfly)、[Fly64](https://github.com/ornata/fly)、[Eon fly-brain](https://github.com/eonsystemspbc/fly-brain)、[FlyBody](https://github.com/TuragaLab/flybody)、[FlyGym](https://github.com/NeLy-EPFL/flygym)、[FlyGM 官方页](https://lnsgroup.cc/research/FlyGM/)。

## Doom 是如何接起来的，哪些地方真的会更新

DOOMFLY 保留 166,700 个神经元、25,582,938 条聚合有向边和 124,177,617 个突触接触。神经元数、聚合边数和突触接触数不能混称。

画面被映射到模型的视觉输入，神经活动经过固定的神经元到按钮接口控制移动和射击。损伤会刺激两个 PPL101 多巴胺细胞；实验版本在 4,184 条既有 KC→MBON11 连接上应用可塑性规则。感受器映射、动力学、按钮解码和强化刺激都包含人工建模选择。

截至核验日，其 README 明确记录 v6 未通过视觉、条件反射和生存验证。因此“有神经活动”“有权重变化”“某一局活得更久”都不足以证明学会玩 Doom。[当前实现与结论](https://github.com/nftechie/doomfly)、[实验协议](https://github.com/nftechie/doomfly/blob/main/docs/doom-live-training.md)。

## Eon 演示与完整可训练大脑的差别

Eon 在 2026 年 3 月展示了 FlyWire 连接图加 LIF 神经模型驱动仿真身体。官方说明运动输出是工程连接，身体运动神经元没有随该 FlyWire 脑数据一起扫描；当时使用的模型无突触可塑性规则，也不能形成长期记忆。[Eon 原文](https://eon.systems/updates/weve-uploaded-a-fruit-fly)。

其提及的高预测准确率不应扩大解释为“整只数字果蝇与活果蝇有同样准确率”。原始 Shiu 论文验证的是摄食启动、触角梳理等具体感觉运动线路。[Nature 原始论文](https://www.nature.com/articles/s41586-024-07763-9)。

## FlyGym 2.1 确实提供真正的 PPO 例子

已审读本地上游版本 `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`。

- 官方教程：`vendor/flygym/docs/tutorials/6_muscle_imitation.md`。
- Gymnasium 环境：`vendor/flygym/src/flygym_demo/muscle_imitation/env.py`。
- PPO 训练入口：`vendor/flygym/src/flygym_demo/muscle_imitation/train.py`。
- 命令行入口：`vendor/flygym/src/flygym_demo/muscle_imitation/__main__.py`。

此例通过 Stable-Baselines3 的 PPO 创建 MLP 策略，调用 `model.learn` 实际更新策略，输出每回合奖励、周期检查点及 `final_model.zip`。它提供训练闭环，不只是播放动画。对应官方文件：[教程](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/docs/tutorials/6_muscle_imitation.md)、[训练器](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym_demo/muscle_imitation/train.py)。

**范围限制：**这个示例训练左前腿的 15 个肌肉激活去跟踪动作捕捉片段，胸部固定，其他腿被动或锁定。它不能直接算作“整只果蝇学会自由寻食”。全身寻食任务需要另外定义全身身体环境、观测、动作和奖励。

示例复现命令，需在独立 Python 环境执行：

```bash
python -m pip install 'flygym[rl]==2.1.0'
python -c 'import stable_baselines3; import torch'
python -m flygym_demo.muscle_imitation \
  --clip 0002 --total-timesteps 200000 --seed 42 \
  --log-dir runs/muscle-0002 --video-path runs/muscle-0002/rollout.mp4
```

20 万步仅适合功能验证，不能据此承诺收敛。上游示例命令涉及千万级训练步数；实际速度应本机测量。

**源码审读发现：**官方命令行在缺少 `stable_baselines3` 时会提示跳过训练并退回随机策略。因此成功退出或生成视频不等于完成训练。本项目自己的训练入口应该缺依赖即报错，并核验策略参数变化、检查点和评估结果。[命令行源码](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym_demo/muscle_imitation/__main__.py)。

依赖要求：FlyGym 2.1.0 为 Python `>=3.12,<3.15`、MuJoCo `>=3.9,<3.10`；`rl` 可选依赖安装 Gymnasium、Stable-Baselines3 `>=2.7,<3.0` 和 TensorBoard。这个肌肉训练例子使用 CPU MuJoCo 世界，不是 Warp 并行仿真。[依赖定义](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/pyproject.toml)、[安装文档](https://neuromechfly.org/installation/)。

2.x 已重写 API，旧 Gymnasium 版迁移为 `flygym-gymnasium`，旧教程和新包不能混用。[官方迁移说明](https://neuromechfly.org/migration/)。

## 为什么 snedea/flybrain 不作为本项目 RL 核心

审读本地版本 `9191824d17871b7851645782d53d23f213ddb938`，结论仅针对这份代码。

1. `js/sim-worker.js` 的连接权重 `values` 在载入时赋值、归一化，并在索引重排时复制。运行中的 `tick()` 读取这些权重传播信号，更新电压、放电和不应期；没有基于奖励的权重更新、优化器或训练策略参数。
2. `js/education.js:186` 本身明确说明没有突触可塑性，权重固定。这与神经模拟实现一致。
3. `js/brain-worker-bridge.js:273` 的 `synthesizeMotorOutputs()` 用手设系数把神经分组活动合成为走路、飞行、梳理和进食驱动；它明确说明这在代替未包含的 VNC 输出。
4. `js/fly-logic.js:54` 用优先级与阈值选择行为；`js/fly-logic.js:106` 的 `computeFoodSeekDir()` 直接利用食物坐标计算目标朝向。食物趋近本身不是通过 PPO 学出来的。

因此它是适合参考界面、脑活动显示和互动方式的固定权重连接图演示；不能把给它喂食或其多巴胺节点亮起称为“已经强化训练”。这不是否认使用真实连接数据，而是区分数据来源与学习机制。

固定版本证据：[神经 worker](https://github.com/snedea/flybrain/blob/9191824d17871b7851645782d53d23f213ddb938/js/sim-worker.js)、[自述限制](https://github.com/snedea/flybrain/blob/9191824d17871b7851645782d53d23f213ddb938/js/education.js#L186)、[人工运动合成](https://github.com/snedea/flybrain/blob/9191824d17871b7851645782d53d23f213ddb938/js/brain-worker-bridge.js#L273)、[行为逻辑](https://github.com/snedea/flybrain/blob/9191824d17871b7851645782d53d23f213ddb938/js/fly-logic.js#L106)。

## 本项目训练路线与验收建议

以下是工程建议，不能作为已完成的功能清单。

### 第一阶段：身体会训练

在 MuJoCo 身体上封装 Gymnasium 任务，例如寻找食物、到指定地点或避开障碍。PPO 的动作可先控制左右步态驱动、速度及转向，底层步态使用明确披露的已有控制器。这样训练的是任务控制策略，避免一开始就必须同时学会六条腿平衡和理解任务。

独立记录环境状态、策略动作、奖励分量、回合结果及随机种子。评估须使用训练未见过的种子或场景，对比初始策略、随机策略和训练后的策略。展示成功率、到达时间和碰撞等任务指标，不能只看训练总奖励。保留最佳与最近检查点，并支持恢复训练。

### 第二阶段：加入真实连接拓扑

将真实脑图实现为可训练策略结构或可训练读出前的固定神经模块，清楚说明哪些边固定、哪些参数更新，以及传感器和动作如何映射。与同规模普通 MLP、重连图对照，评估真实连接拓扑是否有作用。FlyGM 提供这一方向的论文依据，但没有可拉的官方训练代码。[FlyGM 论文](https://arxiv.org/abs/2602.17997)、[官方代码状态](https://lnsgroup.cc/research/FlyGM/)。

### MiniCPM-o 接入

让 MiniCPM-o 输出结构化任务，如目标类别、目的地、难度和允许动作；经过校验后交给环境与策略。它可以解说训练状态、与人对话和处理相机/语音输入。环境负责计算客观奖励，训练器负责更新参数，仿真负责身体运动。不要让语言模型虚构分数或训练成绩，也不要把语言模型直接生成动作混入“纯 RL 评估”而不标记。

## 下载、3D 资产和许可

MaleCNS 的 DOOMFLY 最小输入是约 13 MB 的注释、42 MB 的神经递质预测、1.1 GB 的聚合连接图。无需为该路径下载完整电镜体积或 12.7 GB 原始突触点。官方下载页面还提供神经 skeleton；这类神经形态与能走路的带关节身体模型是不同资产。[官方下载](https://male-cns.janelia.org/download/)。

FlyGym 提供真实身体模型与 [官方浏览器交互 viewer](https://neuromechfly.org/wasm/viewer/viewer.html)，身体 mesh 会按需获取。它既可以用于物理仿真，也可作为视觉参考；应保留上游来源和许可。[FlyGym 文档](https://neuromechfly.org/)。

DOOMFLY 原创代码采用 MIT，但数据、引擎和游戏资产有各自许可；它使用 ViZDoom 所带 Freedoom 资产，不意味着商业 Doom 资产随源码开放。[第三方许可说明](https://github.com/nftechie/doomfly/blob/main/THIRD_PARTY.md)。FlyBody 和 FlyGym 为 Apache-2.0；Eon fly-brain 为 GPL-2.0。若分发整合产品，应分别保留相应许可和署名。

本报告未宣称本机已经完成长期训练或取得训练效果。具体安装、运行、训练步数和实测结果以项目运行记录为准。
