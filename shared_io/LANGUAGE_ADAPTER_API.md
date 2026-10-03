# 真实脑与身体数值驱动的语言 QLoRA

原始 MiniCPM-o 4.5 LLM、VPM、APM、TTS 参数保持冻结。在现有 NF4 语言模块内部给 `q_proj`、`v_proj` 加 PEFT LoRA，保留同一个原生语言模型对象与 `generate` / `model` / `lm_head` 接口。rank 4 新增 1,916,928 个可训练参数；不合并或覆盖原权重。原生语音仍消费同次语言生成的隐藏状态。

这是新增的小型语言适配器训练，不能称作全脑突触与大模型联合端到端微分。此次只优化 LoRA，脑连接层、身体投影、感觉/运动读出均冻结，避免将多个模块同时变化混为一种能力提升。

## 服务与资源

- 正式服务：`http://127.0.0.1:18647`，GPU 3。
- 隔离候选：`http://127.0.0.1:18648`，GPU 1。只读已验证的连接层和身体投影；独立语言 checkpoint 根 `/root/.cache/cyberfly/candidate-language-gpu1`。`/save` 不覆盖正式连接层指针。
- 同一 GPU 使用 `model_training.cli.training_lock_path(uuid)` 互斥锁。`CYBERFLY_SHARED_GPU_UUID` 指定实际 GPU UUID；候选 `CYBERFLY_SHARED_READONLY_HEADS=1`；`--port` 与 `--runtime-file` 独立指定服务地址及运行配置。
- 全部端点只绑定本机。训练、载入、禁用持有服务锁，并拒绝活动的原生双工会话。载入/禁用后清除旧事件缓存。编码、解码、原生流 epoch 及语音张量携带语言适配器 SHA 和 generation，不能跨版本复用同一事件。

## 训练与评估协议

`POST /language_adapter/train`：

```json
{
  "manifest_path": "/absolute/project/datasets/latent_grounding/sixseeds_v2/manifest.json",
  "output": "/absolute/project/artifacts/model-training/new-unique-run",
  "steps": 64,
  "rank": 4,
  "learning_rate": 0.0001,
  "seed": 42,
  "max_seq_tokens": 768,
  "activate_after_training": false
}
```

请求在服务内同步执行，UI 由后台任务代理。输出目录必须尚不存在，实际原始材料位于项目 `artifacts/`，登记 manifest 也可位于 `datasets/`。进度为 `progress.json`，追加事件为 `events.jsonl`，最终记录 `training-evidence.json`。component 固定为 `grounded_language_qlora`，含 `status`、`step`、`total_steps`、真实 loss / gradient / 显存峰值。`output/STOP`、`output/stop.request` 或 `output.parent/STOP` 会在下一个优化步骤前停止，保留真实已完成参数并记录 `interrupted`。

每条记录必须包含：

```text
event_id / source_path / source_sha256 / group_id / split
input.text                          所有样本相同、不含真值的指令
input.image_path / audio_path       原始媒体，音频可缺失
input.brain_state_before/after      真实全节点 NPZ 描述符
input.body_features_file            真实前身体数值描述符
input.body_feedback_file            真实后身体数值描述符
input.connection_mode=neuron_direct
input.io_scope=sensory_motor
input.fusion_mode=latent_only
target_text / numeric_gold          仅用于标签、独立打分
```

输入实际计算为原 VPM/APM 感知 token，加真实前身体 1 个 soft token、后脑 4 个 soft tokens、后身体 1 个 soft token。全部数值直接进入 `inputs_embeds`，不转为 measurement facts 文本。监督答案只放在目标 token 区，前缀 labels 全部为 -100。优化器仅包含新增 LoRA；原模型参数的 storage、shape、dtype、版本与冻结状态逐步核验。

训练前固定 train / validation / test 和完整日程。前后分别对相同留出样本执行：真实数值、同媒体但六个数值 token 置零、同媒体但换成另一个真实样本的数值。保存每次原始生成、各字段正确/错误/未知、目标 NLL；换入样本固定选择，不根据模型结果挑选。checkpoint 保存后在同一真实模型中重新载入，再做后测。禁用适配器后要求原模型固定输入的隐藏状态逐值不变。

`POST /language_adapter/evaluate` 接受 `manifest_path`、新 `output`、`split: validation|test`、`max_seq_tokens`、`ablation`。它不训练；不要反复看 test 调参后再宣称独立测试。

在线 replay 的 manifest 使用 `online_learning: true`，保留封存的 validation/test 原记录，train 只接受独立的真实新回放。请求必须显式 `evaluation_splits: ["validation"]`：测试记录不会被编码、生成或优化；调用独立 evaluate 的 test 也会拒绝。默认静态训练仍前后评估 validation/test。在线新回放的近重复与 episode 隔离由独立 CPU 数据登记器检查，服务继续核验原始文件 SHA 和标签不进入输入。8 步在线更新仍是后台真实优化；不能承诺每一个身体仿真帧都完成大模型训练。

普通 `/decode` 可使用 `max_new_tokens: 10..24` 与 `generate_audio: false` 缩短后台文字评论。`max_new_tokens: 0` 只计算真实隐藏反馈，返回 `say: ""`、`generation_skipped: true`、`status: actual_neuron_conditioned_hidden`，不伪造语言或语音；它与 `generate_audio: true` 不兼容。`measure_brain_ablation: false` 可跳过每轮额外反事实前向，响应明确 `brain_ablation.performed: false`，不产生虚假因果数值。两个选项默认分别是原来的完整生成和实际对照。语音短预算若截断在自然结束前，原生 hidden 对齐保护仍会报错，不能将残缺输出包装成已完成语音。

## 载入、禁用与存档

`POST /language_adapter/load`：

```json
{"checkpoint":{"path":"/absolute/artifacts/model-training/run/adapter.safetensors","sha256":"actual file SHA256"}}
```

文件内嵌模型 revision、原权重 manifest SHA、LoRA 结构、实际张量内容 SHA、代次和优化步数。先检查文件 SHA、来源、键/形状/FP32/finite，再原位恢复并检查实际内容 SHA。载入成功才更新当前指针。`POST /language_adapter/disable {}` 立即恢复原始基础模型输出并保存禁用状态；不删除训练结果。`POST /save {}` 同时保存语言适配器，候选不会保存正式脑/身体头。

`GET /health` 与普通编码/解码响应附加：

```text
language_adapter_enabled
language_adapter_sha256             禁用时 none
language_adapter_generation
language_adapter_training_steps
language_adapter                    结构、checkpoint、冻结与可训参数统计
```

默认训练完成后恢复运行前状态，不自动部署候选。模型基础身份与 LoRA 身份分别保留，音频不冒称来自未改动的语言分布。

## 固定 64 步实跑记录

目录：`artifacts/model-training/latent-language-run64-v2-01`。历史提交地址 `artifacts/grounded-language-qlora/run64-v2-01` 是该目录的符号链接；只有一次日程、一次前测、一次训练和一次后测。

冻结数据 manifest SHA 为 `d23119d7f35ee22a6379f07482be863c16f27e4fee3ab320affb1b441b2b4432`：48 条真实记录，6 个独立模拟 episode，32 train / 8 validation / 8 test。固定 64 步、rank 4、学习率 1e-4、seed 42，32 个训练样本各使用两遍。答案只要求可以读取的 2,129 个 output 神经元放电，不把不可见的全脑总放电作为必答标签。

此批数据全为移动实例，且 test 没有接触不变例子，没有原始音频样本。不能用它证明静止判断、一般声音理解或泛化到未知场景。实际结果以 `training-evidence.json` 和 `before/`、`after/` 的逐条证据为准；数值影响了 logits 或 NLL 降低，都不等于已经可靠读懂脑状态或主观思想。

实跑已完成：64 步，adapter 参数变化 L2=3.34292，checkpoint 实际重载通过，禁用后原模型固定 hidden 逐值相同，原始参数可训数 0；峰值分配 8.70 GiB、保留 8.91 GiB。validation/test 精确字段正确均为 1/24，数值置零对照均为 2/24；完整三字段全对均为 0。NLL 从 1.610/1.706 降到 0.597/0.635，但没有建立可靠数值理解。`assessment.json` 明确建议原模型继续作为默认，候选只作可切换实验。
