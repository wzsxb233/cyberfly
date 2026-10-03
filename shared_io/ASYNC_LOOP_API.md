# 身体持续运行、模型后台反馈

`AsyncSharedLoop` 是独立的延迟输入协议，只支持 `neuron_direct + simplex + latent_only`。原同步 `SharedIOBus` 的同画面断言不变，原生 duplex 也没有被快照循环替代。

线程边界：身体 actor 读取一次真实全脑快照和身体感官；后台仅接触这些文件、复制的 RGB 和只读 `SnapshotBrain`，执行原模型编码。身体照常步进。编码完成后，actor 在新鲜图像、当前脑状态和现有输入上重新计算新增电流余量，只消费一次模型电流。完成这个实际神经/物理步后保存后状态，交后台生成文字或原生语音。

```python
loop = AsyncSharedLoop(brain, io_scope="sensory_motor",
    max_delay_s=30, max_new_tokens=16, speak=False)

# 以上及以下入口必须由同一个身体 actor 调用。
loop.poll()                         # 只检查已完成 Future，不等 HTTP
if loop.phase == "idle":
    loop.observe(text, rgb, body_packet=actual_body_packet,
        legacy_general=actual_existing_general,
        legacy_neural_drive=actual_normalized_ports,
        visual_input_enabled=True)

application = loop.take_stimulation(fresh_rgb,
    body_packet=fresh_body_packet,
    legacy_general=actual_existing_general,
    legacy_neural_drive=actual_normalized_ports,
    visual_input_enabled=True)       # 尚未 ready 时返回 None

if application:
    env.set_general_stimulation([
        *original_general, *application["additional_stimulation"]])
try:
    env.step(original_passthrough_action)  # 等待模型时也照常执行
finally:
    # 场景已有约定：文件刺激是一次性队列，step 在执行前即消费。
    # 不把已经消费的文件重新排队；持续的 group/ID 输入继续保留。
    env.set_general_stimulation([
        item for item in original_general if "neuron_currents_file" not in item])
if application:
    loop.feedback(actual_neural_result, body_state=actual_state,
        body_packet=actual_body_after)
```

`observe` 返回 `{accepted,phase,observation_id,...}`，busy 时不排第二份任务；其 actor 时间仅包含只读快照与入队。`take_stimulation` 返回 `None` 或 `{event_id,additional_stimulation,model_observation,actual_application}`，第二次取同一份结果返回 None。`feedback` 立即返回，下一帧继续推进身体。`poll()` 收到完成事件后写入 `last`；`status()` 为轻量状态。`close()` 不等网络，会阻止该循环再施加任何晚到电流，后台只有文件访问。超过 `max_delay_s` 的结果过期且不施加。

每个 `artifacts/shared_io/<event_id>/event.json` 明确记录：

- `schema:3, transport:async_simplex, asynchronous:true`。
- 原始 `image_path/image_rgb_sha256`、`brain_state_before`、`body_features_file` 是模型观察时刻的数据。
- `model_observation` 保存观察墙钟、脑时间和身体时间。
- `actual_application.image_path/image_rgb_sha256` 是真正送入脑的当前画面；`brain_state_at_application` 是此时的全量真实脑快照。
- `actual_application.model_delay_s`、`image_matches_model_observation` 明示延迟与图像是否不同。
- `current_consumption_count:1`，真实 worker 的总电流 SHA、原端口、最新图像 SHA 逐项核验。
- `brain_state_after` 的时间、counts SHA、spikes 总数必须与真正执行的那一步相同。
- `actual_input_verified:true`，同时 `shared_input_verified:false`；不会冒称同一画面同时到达模型与脑。
- `body_feedback` 的位移区间是从模型观察到执行后反馈，包含等待期间真实身体步；完整时间均保存。

新增 body 电流使用实际执行时的身体感官；视觉/APM/语言新增电流来自模型较早观察。三端口、原视觉、原有持续刺激仍按场景已有规则执行。语言解码继续使用原观察的图/音/text 与真实后脑/后身体数值，模型与连接层身份跨事件两端必须一致。

默认短文字可以在 16 tokens 处截断；它是有限后台评论，不能当成完整自然句。`max_new_tokens:0` 明确跳过文字；`speak:true` 应给足自然结束预算，如128。全量同次 hidden→TTS 路径保持，未返回真实原生音频就不会提供播放文件。默认在线跳过每次额外脑消融，响应明确 `brain_ablation.performed:false`。

实际证据：`artifacts/async-shared-validation/6cae34b2e58d/evidence.json`。独立真实 166,700 节点脑、3D 身体和 GPU1 原 MiniCPM，编码等待期间13步、解码等待期间3步；观察快照和入队0.332秒；新旧图像SHA不同、一次电流施加、原输入和脑记忆保持。没有访问或重置主 lab。该次初始文件刺激队列为空，队列语义另行验证。CPU 所有权/取消测试在 `shared_io/tests/test_async_protocol.py`，不能代替真实闭环证据。

这项改动移除了模型网络请求对身体 actor 的阻塞；全脑计算、物理积分和快照仍有实际成本，不宣称已经实现固定60帧或硬实时。

一次性队列已独立实际验证：`artifacts/async-file-queue-validation/492b123fd5fc/evidence.json`。非零逐神经元文件仅在首个真实区间加载，持续group输入保留；第二个区间没有重放，错误SHA文件的失败区间也会消费队列，随后真实步可继续，脑记忆不变。该测试不调用LLM。另一次含队列的完整后台测试遇到GPU1训练排队94秒，超过90秒过期门限，正确地不施加晚到电流；原失败记录 `artifacts/async-shared-validation/3e6b2aa90cfd` 保留，不扩大门限冒称成功。
