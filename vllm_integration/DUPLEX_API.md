# 原生双工数值接线：协议已验，服务扩展待合并

当前仅独立 CPU 协议与官方客户端假传输测试通过。没有重启、修改正在加载的 worker，没有跑 vLLM 双工 GPU 推理；当前应用仍使用 simplex。此文件中的“待接”能力不能标为已完成。

固定官方源码 commit `e284d907b5554038bef6ab861bdb78b0b99dd5ea`。公开原生入口是 `/v1/realtime?duplex=1&autostart=0`（也有 `/v1/duplex`），不是 `/v1/chat/completions` 的文本流。官方 [客户端](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/vllm_omni/clients/duplex.py) 保留独立输入/输出、listen/speak、原音频 chunk、播放 ACK、取消和恢复。

## 已有本地接口

`vllm_integration.duplex_client.NativeLatentDuplexClient(url, *, model, ref_audio, session_id, initial_user_text='', connect=None)` 是异步 context manager。限定 loopback；官方原生音频输出需要真实 `ref_audio`。`initial_user_text` 是用户真实指令，不包含脑/身体测量文字。数值来自已有 `LatentCoupler.condition` 的真实脑/身体 soft token 文件。

方法：

- `await append_unit(event_id=..., pcm16=bytes, condition=descriptor, image_path=None)`：严格一秒、16 kHz、单声道 PCM16；可附原 PNG/JPEG。返回 `sent_not_yet_worker_verified` 审计，不能把发出成功当作入模成功。
- `events()` / `responses()`：保留官方原始事件与增量语音 handle；正常 listen 允许空文字、无音频。
- `await ack_playback(played_ms)`、`commit()`、`cancel_response(response_id=None)`。取消后本包装器要求重新开启会话，防止旧 epoch 数值误投新上下文。自动网络重放暂禁用；未来有服务端严格 operation ACK 后才开启。

客户端检查真实 `session.capabilities.implementation_level == model_native_duplex`、实际 runner KV / resumable request / 独立 IO 等标志。还必须收到自定义能力：

```json
{"cyberfly_duplex_latent":{"schema":1,"placement":"before_sensory","append_only":true,"facts_in_model_input":false,"generation_hidden_export":true,"unit_samples":16000}}
```

**当前 stock 服务没有该标志，客户端会拒绝并关闭会话。** `require_native_capabilities(session, require_latent=False)` 可单独核实原生协议，但不能据此声称数值融合已接通。

音频-only 可以；当前官方 native append 拒绝纯文字/video-only 输入。没有麦克风时需明确提供“合成静音时间输入”，不得声称采集到了音频。首个官方 APM 窗口约 1035 ms，原实现会在真实首包前面补零；客户端仍传真实一秒样本，审计不能把内部 padding 当作录音。应用负责尾段切分/补零且记录实际样本数；本包装器不暗中补齐。

## 需要合并的最小扩展点

1. **Serving 透传与校验，不能只改 worker。** `entrypoints/duplex/realtime_input.py:401` 和 `session_runner.py:1582` 都重新构造音频 payload，stock 会丢弃任意 `additional_information`。在项目持有的受限 serving adapter/兼容层中保留并校验 `cyberfly_duplex_latent`；它包含 session/epoch、独立零起始 `unit_index`、event_id、文件 SHA 与原始媒体 SHA。文件只能由本地已授权 artifact 根解析。官方缓冲器会把小 PCM 包拼成 unit；首版只接一包一秒，禁止把最后一包的 condition 误绑定到多个拼接单元。
2. **先预留新的 scheduler slots。** `MiniCPMO45DuplexRuntimeExtension.plan_append` 使用 [runtime.py](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/vllm_omni/model_executor/models/minicpmo_4_5/duplex/runtime.py) 的 `build_duplex_data_plane_prompt`。项目扩展可调用 `reserve_append_slots(prompt, soft_token_count=K)`，在提交 scheduler 前增加本次 append 的 K=4..6 个位置。`duplex_runtime_extension` 是 pipeline 配置上的正式类路径扩展点。不得在已经分配的旧 KV 上插入/移动 token。
3. **在原 helper 输出后、左 padding/offset 切片前插入。** [stage0.py](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/vllm_omni/model_executor/models/minicpmo_4_5/duplex/stage0.py) `_stage_prefill_embeddings_only` 产生原 VPM/APM token。使用当前 `<unit>` 后、原视觉/音频前的位置插入 K 数值行。保留原 `<unit>`、上次生成终止符、所有感官行和最后感官 embedding，不强制 speak/listen，不屏蔽 EOS。`fuse_current_unit` 已用真实归档 soft token + 明确人工索引数据验证。首版只允许实际一个单元；多单元 flush、尚在 buffering、自动静音 continuation 必须有独立映射，不能套用一个新脑快照并冒称多份新反馈。
4. **尊重两个原生 offset。** 官方 [wrapper preprocess](https://github.com/vllm-project/vllm-omni/blob/e284d907b5554038bef6ab861bdb78b0b99dd5ea/vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni.py#L289) 用 `duplex_prompt_len` 与 `duplex_token_offset` 将当前 append 放在可续请求尾部；runner 同时传 `_omni_num_computed_tokens`。左 padding 用来对齐已存在前缀，不能把它误计为新感官。`token_offset >= prompt_len` 时必须使用原生成 token 的 embedding lookup，绝不能复用 prompt 数值切片。任何长度不足必须报错，不继续用 padding 假输入。
5. **原 hidden 持续导出，TTS 对象原样返回。** 沿用现 bounded latest exporter，每次真实 forward 的最后已处理 token hidden 可产生新序号。增加 `session_id/incarnation/epoch/runtime_seq/unit_index/event_id` 与对应 condition/body/LoRA 身份。`runtime_seq` 来自引擎（当前 planner 首 append 为 1）；客户端 `unit_index` 为 0，禁止简单视为同值。生成 sampled token 后下一次 forward 才有它的 hidden；不能把 next-token sampling 的前一 hidden 标成已经处理了该 token。原 `OmniOutput.multimodal_outputs['latent']` 与 `text_hidden_states` 保持同一原对象交给原 Talker/Token2wav。
6. **逐新单元脑反馈。** 身体/全脑继续真实推进；消费者按每请求最新 hidden 限频投影，只归档实际施加的逐 ID 电流。在后续音频单元读取最新真实脑/身体状态，经原小头生成新的 soft tokens，只影响此刻之后的注意力/生成。没有新观测的 continuation 标为沿用先前 KV，不能虚构新的共享输入。当前头是工程映射，因果影响不等于正确读懂动物主观想法。

顺便合并现 worker 的 `_cyberfly_sequences` 生命周期清理：跟请求结束/condition 淘汰使用有界存储，不能在仍在计算同一 request 时重置序号。当前 worker 文件保持冻结，尚未改动。

## 必须通过的上线验收

原 raw 音频 + 图像连续输入；每单元核对模型/脑/身体时间及身份；真实生成非空语义 token hidden、原 native 音频可播放；正常 listen 被保留。同一个 frozen checkpoint 下做数值扰动对照，验证后续 hidden/logits 改变；同一输出张量继续进入原 TTS。取消→新 epoch→新输入不串 KV/声音。网络失败不能自动重复施加电流。所有测试需保留原始事件；纯聊天流式或假 socket 测试不算这一验收。
