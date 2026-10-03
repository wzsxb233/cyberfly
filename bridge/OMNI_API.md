# 显式 vLLM-Omni 应用桥

入口：

```python
from bridge.omni import OmniBridge, OmniBridgeConfig

bridge = OmniBridge(OmniBridgeConfig(
    base_url="http://127.0.0.1:18649/v1",
    model="cyberfly-minicpm-o-4.5",
))
result = bridge.chat("看一下现在的画面", state={}, image_path=actual_png)
text = bridge.transcribe(actual_wav_bytes)["text"]
wav_bytes = bridge.speak("你好，赛博果蝇。")
```

配置状态始终是 `configured_unverified`，不会仅因填了地址宣称连接成功。不修改原 `BridgeConfig` 支持列表、默认模型配置或正在运行的服务。根应用应显式选择本构造器；`provider='vllm_omni'` 可以区分旧后端。

`plan`、`plan_for_scenario`、`plan_neural_drive`、`plan_brain_stimulation`、`plan_direct_action`、`chat` 原签名和原严格解析全部复用。JSON 规划属于语言交互模式；该桥不会把它标为数值共享，也不会向主干暗中添加脑/身体测量文字作为 latent 替代。真实逐 ID 数值融合仍由独立共享总线负责。

传输统一复用 `vllm_integration.client.OmniClient`。新增 `generate(messages=[...])` 保留真实 system/user/assistant 角色与实际图片内容。自定义 messages 不能同时传 text/image_path/audio_path/condition/additional_information；会在网络请求前拒绝，防止媒体展开或模板变化悄悄移动已有 numerical slots。原 `generate(text, image_path=..., condition=...)` 调用行为保持。

语音使用已存在的 `/v1/chat/completions` 多模态协议：转写将真实单声道 PCM16 WAV 作为 `input_audio` 交给原 APM/LLM；朗读请求 `modalities=['text','audio']`，返回原 Thinker/Talker/Code2Wav 音频。没有假定服务实现 `/audio/transcriptions` 或 `/audio/speech`。`transcribe` 限30秒；`speak_result` 另返原请求ID、音频描述符、文件SHA和证据路径。`speak` 返回兼容原应用的 WAV 字节。

朗读时核对模型返回文字与指定原文（允许标点/空白差异）；若模型改写、漏音频、文件SHA不符则拒绝。该核对不是外部语音识别对音频内容的确认；真实端到端验收仍须听检/音频转写校核。不调用浏览器TTS、云端或旧GGUF作为失败替代。

证据边界：`bridge/tests/test_omni.py` 的6项 mock HTTP 测试通过，包括真实请求体角色/媒体保留、原JSON非法数值拒绝、原生音频字段、缺音频和改写拒绝、混合messages/condition拒绝。测试PCM/回复均为人工fixture，临时文件自动清理；不是模型推理成功证据。当前模块尚未进行 GPU 推理验收，等待主三阶段真实服务测试完成。
