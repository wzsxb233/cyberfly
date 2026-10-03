# 原生视频 / 双工的最短接线审查

2026-09-13，只读审查已下载并核验的官方 MiniCPM-o4.5 revision `503e754207c94da6bb26850b4469f367c9ea3582`。**下述 HF 双工数值链路尚未运行验证。** 当前已运行的是同次隐藏状态轮次语音，证据 `artifacts/native-speech-validation/e6ac3ba31dd8/summary.json`；已有GPU2原生WS双工是独立旧运行时，不能替代这条脑/身体数值双工链路的验收。

最短路径复用 GPU3 当前完整原模型，不再加载另一个LLM：

1. 使用 `model.as_duplex()` / `MiniCPMODuplex.from_existing_model` 创建原双工包装器，再 `prepare(prefix_system_prompt=..., prompt_wav_path=已验证默认音色)`。赋原processor实例避免重复处理器加载；同会话独占原APM缓存、LLM缓存、TTS缓存和vocoder缓存。
2. 收音仍连续输入，按原处理器约1秒一单元送 `streaming_prefill(audio_waveform, frame_list, text_list)`；首单元默认1035ms，后续1000ms，16kHz。同一单元可附真实3D帧（例如1fps）与文字，不为每帧重置原模型会话。
3. 原prefill成功后，真实逐ID脑状态与身体向量经过已有可训练头产生 `soft[K,4096]`，调用 `duplex.decoder.feed(soft, return_logits=True)`。这是原LLM对原多模态KV的实际数值追加；返回真实hidden可给现逐ID电流头。把返回logits写回 `duplex.pending_logits`。schema附 `('brain',4)` / `('body',1)`等标记供审计，真正的向量保存在artifact。
4. root执行对应神经/身体步，读取该步真实数值，再追加后验brain/body soft tokens并刷新pending_logits。随后原 `streaming_generate()` 会保留说话轮次，返回 `is_listen/text/audio_waveform/end_of_turn`，原TTS `generate_chunk` 与Token2wav流缓存实际参与；音频为24kHz块。接收器可同时继续接新麦克风包，模型GPU工作仍按单元串行。
5. 每个单元记录event_id/sequence、原frame/audio bytes SHA、前后神经/身体packet identity、原模型KV长度、coupler SHA、实际soft token形状/内容SHA、输出音频SHA。浏览器按epoch和sequence播放，取消时立即停止旧播放并丢弃晚到旧epoch块。

这条路径有几个必须显式处理的细节：

- `model_training/omni_loader.py` 当前设置 `config.use_cache=False`，原 `utils.StreamDecoder.feed` 没传显式 `use_cache=True`。双工独占会话必须临时启用原LLM cache并停用gradient checkpointing，退出恢复；否则名为stream的调用会没有真实连续KV。
- 原 `from_existing_model` 无条件再次 `model.init_tts()`，会重建Token2wav。适配层应复用现已加载且核验过的同一个原tokenizer实例；不要重复占显存，也不要指向未固定版本的下载。原基础LLM/视觉/APM/TTS参数都复用。
- 混合audio+text时，原prefill在audio后先保存pending_logits，随后文字仅feed但不刷新；最终数值soft feed并取得新logits能确保生成决策对应含最新文字与脑/身体向量的完整KV。
- 向量必须在 `register_unit_start` 后、`register_unit_end` 前注入，才能被原滑窗单元正确计数。默认先关闭滑窗做短验证；长会话使用其 `basic`/`context`窗口，并记录丢弃范围，不能暗示无限记忆。
- 原HF的break/session_stop仅在prefill/generate入口检查，没有在每个文字token和vocoder算子中即时检查。第一版取消只能保证下一单元边界停止计算；浏览器停止播放可以更快。若新会话要求完全清理，显式清除原APM past KV、processor流状态、decoder KV、TTS past KV与flow/HiFT cache。保持外部果蝇大脑和其学习参数，不因取消语音重置它。
- 双工和当前轮次入口不能同时使用同一processor/codec状态。网络I/O可以并行，单模型会话所有权必须串行，训练也需要先保存并结束该会话。

验收必须实际连续至少两个音频/视频单元：新音频在旧音频播放期已进入接收队列，原APM/LLM/TTS缓存跨单元保留；同一当前帧、不同先前帧的输入应有可核验上下文影响；输出真实音频块；取消旧epoch并接新输入不得播放旧块；脑/身体数值单神经元消融仍要影响原logits。循环调用当前快照 `encode/decode` 不满足此定义。

训练边界：原 streaming_prefill/streaming_generate/StreamDecoder.feed 都带 no_grad；不能直接宣称在线流会话全图反传。当前可训练的是独立逐ID双向头、模态输入头、身体数值头和语言模块LoRA SFT/DPO。流式记录可离线重放成显式监督训练样本；若要训练原视觉/APM/TTS或跨长KV反传，需要各模块目标、可反传重放、显存与真实参数变化验收。外部脉冲脑和MuJoCo仍不在该autograd图中。

精确本地入口：`model_training/models/modeling_minicpmo.py` 的 `as_duplex`约2429、`from_existing_model`约2473、`prepare`约2692、`streaming_prefill`约2777、`streaming_generate`约3151；`model_training/models/utils.py` 的 `StreamDecoder.feed`约2079。原文件未为这次审查修改。

官方源：[固定原模型实现](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/modeling_minicpmo.py)、[固定原StreamDecoder](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/utils.py)。
