# 已实测的 MiniCPM 原生流式语音

连接 `ws://127.0.0.1:18645/v1/realtime?mode=audio`。底层是 C++ 的 **full_duplex** 模式，不是把轮次式转写/朗读改名。已实测真实语音流输入、声音分块输出、取消当前输出、再次输入后新声音输出，证据在 `artifacts/lab-integration/minicpm-duplex-verification.json`。

本通道用于自然语音对话，`brain_involved=false`，不会执行身体或神经动作。一个会话独占单个模型 Worker，期间 HTTP 规划/转写请求排队。主动关闭语音会话后其它请求恢复；最多300秒需重新连接。

1. 等待 `session.queue_done`；可能先收到 `session.queued`。
2. 发送 `{"type":"session.init","payload":{}}`。
3. 等待 `session.created`：包含真实 `session_id`、`mode="full_duplex"`、`epoch`、输入16000/输出24000采样率及 `max_inflight=3`。
4. 发送音频块：

```json
{"type":"input.append","sequence":0,"input":{"audio":"base64 float32 PCM"}}
```

输入为小端 float32 PCM、16kHz单声道、每块0.1至1秒、幅度[-1,1]；不能直接发送WAV/Opus容器。`sequence`严格递增，取消后也不归零。可附 `force_listen=true` 只听该块。`input.accepted` 表示接收，`input.processed`表示该块处理完成，最多3个未完成块；超限明确报错。

`response.output.delta` 有三种 `kind`：`listen`（监听）、`text`（增量字幕）、`audio`（base64、24kHz单声道float32 PCM）。播放音频队列，不把模型字幕当作神经控制。`response.done`是原生分块的完成边界；语音播放与文字可能异步，不应假定一一对应。

发送 `{"type":"input.cancel"}` 可取消自己当前原生会话。立即停止浏览器待播放声音，等 `response.cancelled` 再发送新音频。网关会关闭当前native session、释放其推理线程、清空尚未处理的输入；下一音频重建native session并产生新的 `epoch`。

**取消会重置本段模型对话上下文，不是无缝保留上下文的打断。** 同一WebSocket在取消后可以继续。其它连接不能指定或取消这段session；网关关闭自己session后才释放Worker。发送 `{"type":"session.close"}` 最终关闭并释放Worker。断连和错误同样清理。

`duplex_client.js` 提供可复用浏览器ES模块：

```js
import {MiniCPMDuplexClient} from '/path/to/duplex_client.js';
const voice = new MiniCPMDuplexClient({onEvent: event => console.log(event)});
await voice.open();
await voice.startMicrophone(); // 应由用户点击触发浏览器麦克风权限
// voice.interrupt();          // 打断并清空当前上下文
// voice.close();              // 关闭麦克风和会话
```

也可用 `appendPcm(Float32Array)` 接外部录音处理。helper有播放队列、停止播放和输入积压提示；浏览器麦克风/AEC体验尚不等于自动化语音回放验证。首次切换轮次与双工模式会重建原生上下文，需要等待加载。默认音色来自官方配套参考音频，无额外音色模型下载。
