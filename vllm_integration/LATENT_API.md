# 原生 vLLM-Omni 数值连接

插件已经安装，CPU 档案、既有连接权重、占位位置与身份校验已通过。真实 vLLM 三阶段推理尚待主运行器验收；不能把这些 CPU 检查写成模型已经迁移成功。

固定官方源码为 `vendor/vllm-omni` 的 `e284d907b5554038bef6ab861bdb78b0b99dd5ea`。官方模型仍是缓存中的 MiniCPM-o-4.5 revision `503e754207c94da6bb26850b4469f367c9ea3582`。不修改官方 vendor 或原始权重文件。

## 配置

独立环境 `/root/.cache/cyberfly/vllm-omni-env` 已安装 `cyberfly-vllm-latent==0.1.0`。同一个幂等 `cyberfly_latent` 同时注册在 `vllm_omni.general_plugins` 和 `vllm.general_plugins`：后者是实际 engine 子进程的加载入口，只有前者会导致主进程已识别架构、子进程却退回 Transformers。stage 0 的 `engine_args.model_arch` 设为 `CyberFlyMiniCPMO45ForConditionalGeneration`。stage 1、2 保留原官方 Talker 和 Token2Wav。三阶段配置序列化往返与实际子进程入口解析已通过，见 `worker-registry-evidence.json`。

首版限定 `enforce_eager=true`、`enable_prefix_caching=false`、`max_num_seqs=1`。多个环境可以排队共享，尚未证明多请求合批后的张量映射。原官方 LoRA wrapper 不支持直接加载旧适配器，本插件也尚未开放此功能。

当前 vLLM 0.29 已移除 AR bitsandbytes loader；安装 bitsandbytes 不会恢复它。源码支持 `quantization=fp8_per_tensor`、`load_format=auto`，从原 BF16 逐层量化，并可在 SM86 上选择 Marlin FP8 权重运算。这里的权重格式是 8 bit，不能称为之前的 NF4；实际加载、内存和计算结果仍需 GPU 验收。`int8_per_channel_weight_only` 当前只量化 MoE，不适用于此 dense Qwen3。

## 准备数值输入

```python
from vllm_integration.latent_contract import LatentCoupler, build_latent_content_prefix

coupler = LatentCoupler(brain_state_descriptor, io_scope='sensory_motor')
condition = coupler.condition(
    event_id,
    actual_brain_state_descriptor,
    body_features_file=actual_body_before,
    body_feedback_file=actual_body_after,
)
prefix = build_latent_content_prefix(condition, speak=True)
```

`LatentCoupler` 在 CPU 上读取完整实际神经元档案，使用原有训练过的逐 ID 低秩头。默认读 2,129 个已注释运动/下行节点，写 17,336 个感觉节点；`all_neurons` 使用全部 166,700 个节点的独立读写参数。身体投影只加载已经存在且身份正确的权重，没有随机回退。所有这些头在推理中冻结。

数值文件为 float32 `[K,4096]`，K 为 4–6。四个脑 soft token，加可选前后身体 soft token。它们进入真实 LLM attention，不转成状态报告文字。

HTTP 请求必须只有一条 user 消息，并把 `prefix['content_part']` 放在该消息内容第一项，之后才是原图像、音频和用户文本。合并 `prefix['additional_information']`。使用原模板 `enable_thinking=false`、`use_tts_template=speak`。保留的 `<|fim_pad|>` ID 为 151662，首槽位置 3；媒体在其后展开，因此不会移动这些位置。worker 按真实 scheduler offset 和实际 token IDs 验证，然后只替换对应 embedding，保持 token 数、位置与 KV 记账不变。

## 读取持续变化的模型活动

```python
from vllm_integration.latent_contract import read_latest_hidden

hidden, source = read_latest_hidden(event_id, request_id=None, max_age_s=30)
result = coupler.currents(hidden, event_id=event_id, source=source)
# result['currents_file'] 可交给真实 brain.step 的一次性文件电流入口。
```

`hidden` 是 float32 `[4096]`。`source` 含 event_id、request_id、sequence、phase、token_offset、input_token_ids、tokens_processed_this_step、prefill_complete、hidden SHA、时间、soft token SHA、原权重和小头身份。

导出来自每次实际 thinker forward 的最后一个已处理 token。prefill、后续逐 token 生成都会更新；该次 forward 之后才采样下一个 token，不把尚未处理的采样 token 冒称已有 hidden。后台仅一个 latest 队列，旧快照可被新快照替换，明确记录丢弃次数。每请求只保留最新 16KB hidden 与元数据，不为每个 token 写 166,700 个电流。

`read_latest_hidden` 返回的源文件会被后续更新覆盖。调用方在选择实际施加的脑步时，应先归档这份 hidden 与元数据，再投影电流。实际脑输入时间、RGB、神经响应及下一次反馈文件分别保存；模型处理的旧观测不能标成新鲜身体帧。相同 sequence 不应重复施加，超时快照应拒绝。

读取发布窗口修复（2026-09-14）：事件父 `latest.json` 仅用于首次定位真实 worker request_id，随后立即使用该请求子目录的权威指针。调用方一旦已有 `source['request_id']`，后续传 `request_id=...`，不要使用 HTTP 客户端生成的无 worker 后缀ID。每次最多读取两遍，不在身体循环里 sleep；短时 NPZ/JSON 尚未配对时抛 `LatestHiddenPending`（继承 `FileNotFoundError`），保持身体步进。此时尚未判定原因，不能报为有效模型输出。重复轮询时，同一错误配对持续0.35秒后明确报稳定SHA错误；即使文件不断变化，持续2秒不一致仍失败。元数据、路径、原模型身份或已提交张量错误直接报错。缓存限64请求。离线真实原子更新测试及原失败flow归档只读核验见 `artifacts/vllm_omni_migration/hidden-publication-20260914/evidence.json`，不替代修复后的新模型闭环验收。

读取性能修复（2026-09-14）：`checked_artifacts.read_checked_artifact(descriptor, root=..., limit=...)` 在本机采用 Linux `openat2` 的 BENEATH/NO_SYMLINKS 路径限制，缓存最多8个受信根目录句柄，每次仍检查实际文件类型、大小与完整 SHA。符号链接在此模式下明确拒绝；不支持的内核回退原完整 resolve 检查。`checked_file` 与 latest hidden 读取已接入；无文件内容/校验结果缓存。原子发布的短时 pending、稳定损坏拒绝及全部模型/事件身份检查保留。实际历史文件20次交错基准，完整 latest hidden 校验中位23.35→6.75毫秒；不能据此声称整套系统已实时。证据 `artifacts/vllm_omni_migration/performance/cpu-profile-20260914/read-benchmark.json`。

worker 返回原始 `OmniOutput` 对象，且核验 `multimodal_outputs['latent']` 与原 `text_hidden_states` 为同一张量。输出给原生 Talker/Token2Wav 的真实 hidden 不被替换。这个对象约束已经实现，原生语音仍需首轮 GPU 端到端验收。

## 当前边界

此版普通请求支持生成期间持续导出 hidden，后续请求可接受最新实际数值反馈；尚未把同一请求中途的脑反馈追加进原生 duplex KV。带 duplex 信息的数值请求会明确拒绝，不能把普通请求循环称作官方全双工。后续多环境批处理、原生 duplex 单元和 LoRA 热载需要独立验证。

## 视觉、音频的独立脑通道

原 HF 完整流程另外使用两条既有工程映射：原 resampler 的 4096 维视觉 tokens 经逐 ID attention 到 6,098 个 `ol_sensory` 节点；原 APM/projection/pool 的 tokens 经另一头到 6,370 个 `vnc_sensory` 节点。后者是实验端口选择，不能称为已经还原果蝇自然听觉线路。当前保存的两头都做过两步真实优化，监督标签为集成检查电流，不能据此宣称学会感知任务。

`LatentCoupler.modal_currents(descriptor, event_id=...)` 已严格复用这两份保存权重，不创建新头。`read_latest_modal_embeddings(...)` 将读取每请求的一份不可变 `vision/audio/language` NPZ，缺音频时应为零行，而不是造出音频特征。

独立 `modal_export.py` 已完成 CPU 检查，但尚未挂入当前 pilot worker。准备的 hook 将先保留官方 `embed_input_ids` 所用的真实 `is_multimodal` mask，再按原图像/slice/音频边界读取实际合并输入中的那些 token，保持原 embedding 与语音 latent 不变。单凭提示词中的 `<unk>` 文本会被拒绝。旧实际 HF 模态档案与旧小头兼容已验证；这不能替代新的 vLLM 模态 GPU 推理证据。

## LoRA 后续边界

当前官方外层和 thinker wrapper 都没有 `SupportsLoRA`；其内部 `Qwen3ForCausalLM` 已支持。可行的最小扩展是限定原 LLM，外层显式支持并传下 `qkv_proj: [q_proj,k_proj,v_proj]`、`gate_up_proj: [gate_proj,up_proj]` 的 packed 映射。现有适配器的 `model.layers.*` 权重须对应规范运行时路径 `thinker.llm.model.layers.*`，不改原模型加载映射来凑适配器名称。

新版 LoRA manager 已用对象 ID 处理 `thinker` 与 `model` 同一模块别名，避免同一线性层重复注入；仍需实际加载证明 canonical 路径和启停结果。运行时 FP8、原生多阶段 hidden 与 LoRA 同时启用尚未测试，因此当前保持关闭，不把“有接口”写成“已经兼容”。

离线转换入口为 `lora_export.export_adapter(checkpoint, training_evidence, output, lora_alpha=8)`，两个来源都要求显式 path/SHA，且训练记录必须为已完成的真实 latent-grounded QLoRA、原参数训练数 0、无文字测量输入。历史 checkpoint 未内嵌 alpha，因此必须显式给出原训练所用的 8，不能套用 PEFT 默认值。`validate_export(output)` 会重新核验来源、144 个张量、模型身份与全部导出文件。

实际导出位于 `artifacts/vllm_lora_exports/grounded-run64-v2-01`，包含标准 `adapter_model.safetensors`、`adapter_config.json`、`packing-plan.json` 与来源证据。q/v 两个独立 rank-4 分支分别保留；QKV 由官方逻辑按 `[q,None,v]` 组合，K 没有新增适配器。官方 CPU reader 已读入全部 72 个模块，36 组 packed delta 与原训练矩阵计算一致；证据在 `artifacts/vllm_omni_migration/lora-cpu-20260914/official-reader-evidence.json`。这些检查没有加载大模型或启用 GPU adapter。

`lora_capability.OriginalLanguageLoRAMixin` 是未注册的候选能力文件，当前 worker 没有继承它。它限制 thinker 阶段、语言 q/v 模块，并把原视觉/音频 tower 与 connector 排除在 LoRA 外，检查 36 个规范 QKV 路径与原别名指向同一对象。后续仍须先验实际 FP8 runtime 的加载、启停、hidden 和原生语音，再决定是否接入主环境。

官方扩展点：

- [MiniCPM-o-4.5 官方 recipe](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/recipes/OpenBMB/MiniCPM-o-4_5.md)
- [原生三阶段 wrapper](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py)
- [实际调度输入与 preprocess](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/vllm_omni/worker/gpu_model_runner.py)
