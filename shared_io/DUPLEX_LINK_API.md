# 原模型连续数值双工 API

新增源 `shared_io/duplex_link.py`，复用现有模型。**最新真实主工作台闭环已通过**：`artifacts/full_duplex_checks/f75510651b8c/evidence.json` 与 `latent-freeze-memory-audit.json`。7个实际脑/身体单元，原感觉输入全部保留；先监听，随后自然输出3个实际语音块并结束回合。全程纯latent、无facts、greedy；每单元真实KV计数一致，原模型全部冻结，已有脑记忆 e5a40eaf…/2206条改变边保持不变。身体确实移动约0.65mm，模型却说“身体没有变化”，因此这次验证通过的是数值连接、真实闭环和语音，不是可靠状态理解、一般轮次表现或主观思想解读。原已验证轮次语音仍为 `generate_audio:true` 的 `/decode`。

2026-09-13 的隔离诊断确认：原始流式输入在 greedy 下先 `<listen>`，随后连续实际说出“你好，果蝇身体呈棕黄色，头部有红色复眼，翅膀透明”。证据 `artifacts/duplex_diagnostics/baseline_av-a3684606cddf/`，这是原模型对历史真实感官/脑/身体记录的模型侧诊断，没有重新执行脑闭环。原先将工程软token末位预测直接作为听说决策的接法失败：原 `<listen>` 被改成 `<chunk_eos>`，没有语义hidden。重复末感知embedding的替代实验同样失败，均保留原证据。

修复 `decision_anchor:'native_boundary'` 保留原 `streaming_prefill` 的实际分布，只决定首个原生听/说控制token；后验脑和身体soft仍实际写入原KV，正常喂入已选控制token后，后续语言hidden通过attention读取这些数值。它不强迫说话、不屏蔽EOS、不依赖改变采样参数。当前单元的后验脑不改变首控制token，前序单元脑数值可以影响后续单元的原感知分布；原生重复惩罚和未结束回合的继续说话逻辑保持原实现。不能把这条工程连接称为官方训练过的神经接口。

纯latent模型侧验证已经通过：`artifacts/duplex_diagnostics/boundary_av-3ef5802ea04f/verified-evidence.json`。8单元中先监听，随后5个非空文字单元与4个实际24kHz音块；没有facts，使用原greedy。固定实际第4单元的SPEAK、同KV/身体，仅将4个后验脑token置零，实际语言logits L2差110.67996、hidden L2差6.86470。所有原LLM和原model参数 `requires_grad=False`，实际训练参数数为0；未改变任何checkpoint。包含额外消融KV拷贝的全程GPU峰allocated 13,941,766,144 bytes、reserved 14,740,881,408 bytes。模型侧诊断读取历史真实脑/身记录，最终实时脑→身体→模型闭环另看主工作台验收，不混为同一测试。

默认不需要 `facts`，脑/身可以只通过数值soft token共享隐空间。若显式传入facts，则是额外文本辅助信息，必须如实标识。

主工作台显式使用 `numeric_placement:'before_sensory'`：把前验脑/身体数值放在原 `<unit>` 前，使原VPM/APM感知计算直接读取它们；动作后的脑/身体数值继续在该单元输出前写回。后放前验soft的早期方案在新的实时脑响应下出现持续监听，记录 `artifacts/full_duplex_checks/dcc8ecc9db6d/progress.json`，不能把先前历史记录上的成功推广为该闭环成功。改为前放后，同一失败记录的模型侧对照自然产生“你好，果蝇身体在缓慢晃动。”与3个实际音块，记录 `artifacts/duplex_diagnostics/boundary_av-before_sensory-a1c102ff058a/`。语言的运动描述仍是模型解释，不当成测量真值。

前放数值token也纳入同一原生unit的KV长度，从其真实前位置登记滑窗起点。`native_window_accounting_verified`检查原系统prefix+所有保留unit总长度是否等于实际KV长度；最新7个主闭环单元全部通过。这项检查验证真实记账一致，不等同于已经跑过数小时对话或所有滑窗边界。

- `POST /duplex/open`：`{session_id,connection_mode:'neuron_direct',io_scope:'sensory_motor'|'all_neurons',brain_state}`。回 `epoch`、完整模型/codec/读写ID/耦合器身份。可选`system_prompt`≤2000字。实际 `model.as_duplex()` 复用原模型与已核验Token2wav，临时打开原LLM cache；不重复载入LLM，不替换权重。
- `POST /duplex/prefill`：`{session_id,epoch,sequence:0,event_id,text?,image_path?,audio_path?,brain_state,body_features_file?}`。序号从0严格递增，同一会话只能有一个待完成单元。若有音频，必须恰好1秒、16kHz mono PCM16 WAV；由上层分块，原first-window padding另记。无音频时可使用原VISION或TEXT模式。至少一个真实输入必须存在。回实际4096hidden、逐ID `currents_file`、原vision/audio/language `modal_embeddings_file`、真实脑读取证据、`body_input`和KV长度。
- 上层把该电流并联追加到原感觉输入，实际执行脑与身体，再 `POST /duplex/generate`：`{session_id,epoch,sequence,event_id,brain_state,body_feedback_file?,facts?,generate_audio:true}`。必须有实际向前推进的脑快照，不能重复prefill原状态。后验数值再次feed原KV；facts加入该原单元的文本token。原 `streaming_generate()` 回 `say`、`is_listen`、`end_of_turn`和实际audio chunk；say空/监听是正常状态。监听的原零静音只是时序信号，不导出为模型语音。
- 发声时 `audio_file` 为24kHz mono PCM16，含path/sha256/event_id/session_id/epoch/sequence/sample_rate/samples/duration_seconds和`same_generation_hidden:true`。位置在 `artifacts/native_duplex/<epoch>/`。原vocoder前瞻可能使用该会话前序单元的代码，`source_units`明确追溯，不能把缓存前瞻说成每个音频采样只源于当前事件。尾块可短于0.1秒，任何非空真实PCM帧都应保留。`generate_audio:false`静音交付，但保留原内部语音状态推进。
- 只有同一真实语音turn中存在非空文字和语义hidden来源，才交付音频。上游从空hidden/bare audio_bos生成的波形会被拦截，回 `nonsemantic_audio_suppressed:true`，不设置可播放的audio_file；回合结束后清除旧语义来源，避免误用上回合缓存身份。
- `POST /duplex/cancel`、`POST /duplex/close`：`{session_id,epoch}`。同一个已关闭epoch可幂等关闭，旧epoch绝不能关闭新session。恢复原LLM cache/checkpointing配置，清理原APM/processor/decoder/TTS stream状态；不重置外部果蝇脑。当前GPU计算只能在原单元边界结束，前端可以立即停止播放并丢弃晚到旧epoch块。

同session的耦合器/模型/codec/读写ID身份固定；身体头按layout固定权重SHA。reset脑时间必须新epoch。普通 `/encode`、`/decode`、训练和LoRA验证在活动duplex session期间明确拒绝；`/save`仍可安全保存权重供管理器退出。网络可继续收输入，GPU单元顺序执行，不同时改原可变KV。

已有逐ID转换器训练也补齐 `body_feedback_file`：训练阶段把真实动作后身体软token加入fused输入，再构造语言labels；原body输入与后验body各自可见。原视觉/APM/TTS冻结，当前流方法no_grad；不把流式推理宣称全模块训练或全脑可微反传。

诊断项（不作为产品语义保证）：open可选 `diagnostic_baseline:true` 完全绕过数值/事实注入，`decision_anchor:'none'|'last_sensory'|'native_boundary'`，`numeric_placement:'before_sensory'|'after_sensory'`，`decode_mode:'greedy'|'sampling'`。实际原始控制token及各阶段top10记录在 `decision_trace`。generate的 `diagnostic_brain_ablation:true` 使用同一个复制KV、同body与同控制token做两次实际冻结LLM前向，仅把4个后验脑token替换为0，保存两支实际logits/hidden与差异，不修改运行会话KV。它证明数值的因果影响，不证明已学会自然神经语义。

上层可用 `shared_io.audio_chunks.split_actual_audio` 把原录音按1秒分块（仅末尾零padding有记录），用 `join_native_audio` 按同session/epoch且递增sequence拼接实际发声块。它不插入监听静音、不重采样。保持浏览器输入接收与输出播放同时进行，并保留原会话KV，才符合此原生流式路线。
