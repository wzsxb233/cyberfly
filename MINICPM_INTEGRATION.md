# MiniCPM-o 4.5 接入说明

核实日期：2026-09-13。**已下载并校验官方 MiniCPM-o 4.5 GGUF 全套权重，在独立 GPU 2 上运行真实文本、图像和轮次式 WebSocket 推理。** 本地服务为 `http://127.0.0.1:18645/v1`，模型名 `MiniCPM-o-4.5-Q4_K_M`；配置与实际推理证据见 `model_runtime/runtime.json` 和 `model_runtime/README.md`。不需要云 API key，没有访问凭据或云服务。

当前支持三类不同用途的连接：

- `plan_brain_stimulation(text, state, group_catalog, image_path=None)` 使用真实全图目录输出最多 8 群 signed 电流，支持全部 47 宏群及 23,163 细群，严格校验真实 group_id。47 宏群统计覆盖全部 166,700 神经元，完整细群目录不直接塞进有限上下文。实际模型 signed 请求见 `artifacts/lab-integration/minicpm-fullbrain-plan.json`。
- 更深入的 `shared_io/model_link.py` 使用原始图像/音频编码器与量化语言模块，做真实 `hidden4096→电流47` 和 `脑240→4×4096 soft tokens→inputs_embeds` 双向数值连接。协议见 `shared_io/MODEL_LINK_API.md`，实际验证以 `artifacts/lab-integration/shared-hidden-01` 为准，不能把初始数值影响当成已经读懂思想。

- `neuron_direct` 已实际验证原4096隐藏状态→逐ID独立电流，以及原166,700电压/放电数据→模型内部软向量。默认感觉输入17,336个ID、motor/descending反馈2,129个ID；也可选全部166,700。原视觉和已有感觉刺激照常保留，新模型输入并联追加。证据见 `artifacts/neuron-link-validation/live-ef1fd1b784ac/summary.json`。

原生同次隐藏状态语音、原视觉/音频token导出、身体数值软向量的新增协议与现场验证入口见 `shared_io/MODEL_LINK_API.md`。完整原模块轮次链路已真实通过：`artifacts/native-speech-validation/e6ac3ba31dd8/summary.json`，同次7个生成token的真实hidden→原TTS→54音频codes→2.16秒24kHz语音；含原视觉64/音频30 tokens与身体758维软向量。原HF视频/双工数值链路也完成实际主闭环：`artifacts/full_duplex_checks/f75510651b8c/`，7个真实脑/身体单元、自然监听后3个实际音块、无facts、原模型冻结与脑记忆保持。先前存在持续监听失败；最终仍出现错误运动描述，不能宣传可靠状态理解或一般连续对话性能。完整成功/失败边界见 `shared_io/DUPLEX_LINK_API.md`。

按用户最新授权，MiniCPM LLM/VPM/APM/TTS 原始权重继续全部冻结，可训练新增语言 LoRA/QLoRA、隐空间连接、感觉/运动读出、外部RL及果蝇脑可塑性。`configs/training_policy.json` 允许适配器更新并禁止全量主干微调。GPU 1 上的独立候选服务将真实图像、脑状态和身体数值作为模型输入训练语言适配器，GPU 3 正式服务保持可用；协议、固定留出评估和证据见 `shared_io/LANGUAGE_ADAPTER_API.md`。此前纯文本 SFT/DPO 的历史实验见 `model_training/README.md`。

## 1. 兼容的三端口连接与自然对话

以下是仍兼容的早期三端口链路：MiniCPM 读取语言、神经状态和身体反馈 → 输出三个有界输入 → 真实连接组全图积分 → 原生动作读出 → 场景身体 → 新观测。当前默认 `neuron_direct` 使用上述逐ID数值隐空间连接；这段JSON桥示例不能代表其内部计算方式。

```python
import json
from pathlib import Path
from bridge import BridgeConfig, MiniCPMBridge

runtime = json.loads(Path("model_runtime/runtime.json").read_text())
bridge = MiniCPMBridge(BridgeConfig(
    transport=runtime["transport"], base_url=runtime["base_url"], model=runtime["model"]))

# state 来自当前真实仿真：含 neural、body、lastdrive；brain 是连接组客户端。
plan = bridge.plan_neural_drive(
    "提高左侧视网膜工程输入，观察神经和身体反馈。",
    state=state, brain_ports=brain.describe()["input_ports"],
    # image_path="/absolute/path/to/current_frame.png",
)
# 调度器将 plan.neural_drive 交给 brain.step(..., neural_drive=...)。
# plan.say 是供显示的模型解释，不能当作已执行结果。
```

返回 `NeuralDrivePlan(neural_drive, say, provider, model, status)`，可调用 `.to_dict()`；模型 JSON 必须恰好包含 `neural_drive` 与 `say`，调制对象必须完整包含：

| 输入端口 | 幅值范围 | 实际神经映射 |
|---|---|---|
| `retina_left` | 0–1 | 左 R1-R6，额外 0–10 mV-equivalent |
| `retina_right` | 0–1 | 右 R1-R6，额外 0–10 mV-equivalent |
| `sugar` | 0–1 | LB3c，额外 0–30 mV-equivalent |

负值、超范围、NaN、Infinity、布尔值、字符串数字、缺失/额外/重复字段、代码围栏和尾随文本全部拒绝，不截断、不伪造输出。接口会移除冗长的 `neuron_ids` 列表，但保留端口数量、细胞类型、侧别、量程、来源及限制。

**这些是工程电流调制，并非经过校准的自然刺激或语言语义编码。左视网膜输入不保证左转，LB3c 的糖感受身份来自同源注释，糖端口不应被宣称已证明是正强化。** 真实控制只使用校验后的数字；模型解释可能出错，不能覆盖神经适配器的实际量程和测量。

自然对话调用 `bridge.chat(user_text, state=None, image_path=None)`，返回 `BridgeResult(task="explain", goal_mm=None, say=真实自然回复, ...)`，不修改神经输入。HTTP/WS 都沿用真实模型传输，没有模板回复或失败回退。

兼容接口仍保留：`plan(...)` 返回有限导航任务，`plan_for_scenario(..., scenario_spec)` 只设置场景允许的高层参数。这两个用于独立场景配置，不是直接神经连接模式的控制通道。

单独运行 `python3 -m bridge status` 默认 `disabled`，不会联网；这是客户端配置检查，不会自动读取 runtime.json。实验室服务会读取本地 runtime.json，显式 `MINICPM_*` 环境配置优先。关闭模型用 `model_runtime/stop.sh`，再启用 `model_runtime/start.sh`。

## 2. 官方云端文本与截图接口

官方 API 使用 Chat Completions。主动选择 `modelbest` 后默认指向 `https://api.modelbest.cn/v1`，模型 `MiniCPM-O-4.5-9B`；必须显式提供 API key。[官方 API 文档](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md)

```bash
export MINICPM_TRANSPORT=modelbest
export MINICPM_API_KEY='在本机填入实际 API key，不要提交进仓库'
python3 -m bridge plan '去世界坐标 [3,0] 毫米处' \
  --state-json '{"position_mm":[0,0],"food_goal_mm":[3,0]}'
```

该模式只向 HTTPS 的 `api.modelbest.cn` 发送请求，不会将密钥跟随 HTTP 重定向发送。鉴权失败、超时或模型输出不符合 JSON 合同会明确报错。

客户端使用 Python 标准库；不依赖 OpenAI SDK。截图以 `image_url` 的 Base64 data URL 提交。没有设置官方尚未明确保证的 JSON Schema / function calling 参数；通过提示词要求结构化结果，再做本地严格校验。

## 3. 自行部署的兼容 HTTP 服务

如果已经运行兼容服务，必须显式填写其 base URL 和模型名：

```bash
export MINICPM_TRANSPORT=openai
export MINICPM_BASE_URL=http://127.0.0.1:8000/v1
export MINICPM_MODEL='填入本地服务实际公布的模型名'
unset MINICPM_API_KEY  # 如服务需要鉴权，则显式设置对应 key
python3 -m bridge plan '解释当前果蝇状态' --state-json '{"position_mm":[0,0]}'
```

本项目已提供独立的本地启动与安装工具，见 `model_runtime/README.md`。`MINICPM_BASE_URL` 不包含 `/chat/completions`；客户端自动追加。不同框架的模型支持范围和多模态输入支持可能不同，不能把 HTTP 可连接等同于全双工语音可用。

## 4. 官方本地 Demo 的轮次式 WebSocket 接口

旧仓库 `OpenBMB/minicpm-o-4_5-pytorch-simple-demo` 已重定向至 [OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)。当前维护的部署入口是 Docker Compose，可选 PyTorch 或 C++ 后端；模型目录由主机挂载，镜像本身不包含权重。

本项目支持该 Demo 最新统一协议的 **mode=chat**：

```bash
python3 -m pip install -r bridge/requirements-realtime.txt
export MINICPM_TRANSPORT=realtime
export MINICPM_REALTIME_URL=wss://localhost:8006/v1/realtime?mode=chat
export MINICPM_MODEL=MiniCPM-O-4.5-9B
unset MINICPM_API_KEY  # 如网关有鉴权，再显式设置
python3 -m bridge plan '去世界坐标 [3,0] 毫米处'
```

Demo 使用自签名 HTTPS 证书时，需要在本机信任部署证书或配置受信任证书。客户端保留 TLS 校验，不默认跳过证书验证。若自有本地网关明确开放未加密 WS，URL 可以用 `ws://127.0.0.1:PORT/v1/realtime?mode=chat`。

最小交互顺序：

```text
连接 wss://host/v1/realtime?mode=chat
服务端 -> session.queued / session.queue_update（可选）
服务端 -> session.queue_done
客户端 -> {"type":"session.init","payload":{}}
服务端 -> session.created
客户端 -> {"type":"input.append","input":{
  "messages":[{"role":"user","content":"任务文本"}],
  "streaming":false,
  "generation":{"max_new_tokens":512,"length_penalty":1.1},
  "tts":{"enabled":false},
  "use_tts_template":false,
  "enable_thinking":false
}}
服务端 -> {"type":"response.done","text":"完整模型输出"}
客户端 -> {"type":"session.close","reason":"turn_done"}
```

等待 `session.queue_done` 后才能初始化，等待 `session.created` 后才能发送输入。截图使用 `{"type":"image","data":"Base64图片"}`，不同于云端 HTTP 的 `image_url`。本客户端校验结束事件，再解析结果，不把中间文本直接用于控制。

协议来源：[统一协议](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/overview.md)、[当前 Chat 协议](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/chat.md)。仓库旧 `docs/en/api/chat.md` 仍能看到 `/ws/chat`，不要混用。

## 5. 已接入的语音与全双工边界

**当前已实测并接入 MiniCPM 自身的轮次式语音输入和语音生成，不依赖浏览器 TTS。原生持续音频输入、音频分块输出、取消后新输入已实际验证。** 官方基础模型具备中文和英文语音交互能力。[官方模型卡](https://huggingface.co/openbmb/MiniCPM-o-4_5)

本地 HTTP 提供 `POST /v1/audio/transcriptions`（multipart 字段 `model`、`file`）和 `POST /v1/audio/speech`（JSON `model`、`input`、`voice="default"`、`response_format="wav"`）。Python 标准库客户端：

```python
transcript = bridge.transcribe(Path("recording.wav").read_bytes())  # dict: text/provider/model/status
spoken_wav = bridge.speak("你好，我是赛博果蝇。")                  # bytes: 标准 WAV
Path("reply.wav").write_bytes(spoken_wav)
```

输入接受 60 秒内 PCM16 WAV、单/双声道、16/24/32/44.1/48kHz，最大 8MiB；网关转换为原生 16kHz 单声道 float32 PCM。输出为可播放的 24kHz 单声道 PCM16 WAV。默认音色来自官方权重配套语音缓存。语音和文字/图像共用 GPU 2 的单个串行 Worker；请求会排队。

语音输出是模型根据朗读请求生成的文字与音频，不能保证所有输入逐字一致。原生实际文字通过 HTTP `X-MiniCPM-Text-Base64` 响应头暴露；工作台应将转写显示给用户。一次真实转写将“你好”识别成“您好”，记录保留这一字差异，不声称无误差 ASR。模型错误、无音频、格式错误均报错，没有备用合成声音。

我们发现固定版本 C++ 原生后端会在已有音频或图片嵌入时丢掉随附用户文字，导致一个音频请求被当作对话回答。已在隔离源码应用 `model_runtime/multimodal-text.patch` 并重新编译：多模态用户文字在媒体嵌入前进入模型。修复后同一图片与两条不同文字分别给出了指定的不同端口值，见 `artifacts/lab-integration/minicpm-multimodal-regression.json`。旧 `vision_verification.json` / `additional_inference_verification.json` 的看图结果只证明模型看到图片，不能证明按问题完成控制。失败语音记录也保留在 `model_runtime/audio_verification/bridge_roundtrip_initial_failure.json`。

本地已接入 `ws://127.0.0.1:18645/v1/realtime?mode=audio`，完整客户端协议见 `bridge/DUPLEX_API.md`，浏览器 helper 为 `bridge/duplex_client.js`。实际验证见 `artifacts/lab-integration/minicpm-duplex-verification.json` 与两个原生音频输出 WAV。取消会清空原生上下文并重建会话；不宣称保留上下文的无缝抢话。当前 helper 仍需真实浏览器麦克风与回声条件下验证。

以下是官方上游视频/音频扩展字段示例，本地安全包装只接受文档中公布的音频字段，不能把它直接当本地接口合同：

```json
{
  "type": "input.append",
  "input": {
    "audio": "16kHz 单声道 float32 PCM 的 Base64",
    "video_frames": ["JPEG 的 Base64"],
    "force_listen": false,
    "max_slice_nums": 1
  }
}
```

音频通常每秒一块，不能直接把 WAV 或 Opus 文件字节当 PCM。音频输入是双工协议的主要时间轴；视频帧是可选附加项。输出 `response.output.delta.kind` 为 `listen / text / audio`，声音为 24kHz 单声道 float32 PCM。文字和声音分开到达，不保证一一对应；应分别管理字幕和播放队列。双工没有每轮 `response.done`，需处理 `listen`。官方默认音频会话 600 秒、视频会话 300 秒，需要恢复会话的设计。

来源：[Audio 协议](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/audio.md)、[Video 协议](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/video.md)、[官方最小 probe 客户端](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime)。

下一步可以接仿真第一视角的视频双工。长时间双工会占用后端 Worker，不能假定同一 GPU 可以无成本同时做轮次规划、实时语音与 PPO 训练。

## 6. 本地模型资源与版本

模型 ID 为 `openbmb/MiniCPM-o-4_5`，总规模约 9B。旧 `OpenBMB/MiniCPM-o` GitHub 地址如今指向 [MiniCPM-V 主仓](https://github.com/OpenBMB/MiniCPM-V)。

| 部署方式 | 官方数据 / 要求 |
|---|---|
| 模型卡的单次推理表 | BF16 约 19GB；INT4 约 11GB |
| 完整 PyTorch 实时 Demo | Linux，超过 28GB NVIDIA 显存；初始化约 21.5GB |
| GGUF 的 NVIDIA 全双工 | 最低 12GB，推荐 16GB+；3060 12GB 属边缘情况 |
| GGUF 的 NVIDIA 半双工 | 最低 10GB，推荐 12GB+ |
| GGUF 全套文件 | 约 8.3GB，不能只下载约 5GB 的 LLM 文件 |

前两项来源：[模型卡](https://huggingface.co/openbmb/MiniCPM-o-4_5)、[Demo README](https://github.com/OpenBMB/MiniCPM-o-Demo)。后面三项来源：[Cookbook WebRTC 部署文档](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/demo/web_demo/WebRTC_Demo/README.md)。这些是不同场景的数字，不应把单次推理显存当完整实时服务最低需求。

GGUF 需要视觉、音频、TTS、projector、Token2Wav 与 `prompt_cache.gguf` 等配套文件。Cookbook 特别提示更新过的缓存和其它组件需要匹配，否则可能初始化失败或影响声音质量。[官方 GGUF 权重页](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)

Mac 要求在官方材料中存在差异：模型卡写全双工 M4 Max/24GB；较新 Cookbook 写最低 M4 Pro/36GB、推荐 M4 Max/64GB，并指出吞吐和内存带宽也是瓶颈。实际选机或采购前，应以目标模式的实际跑分为准。

Transformers 路径官方测试环境是 Python 3.10、`transformers==4.51.0`、Torch 2.3 至 2.8；TTS/流式使用 `minicpmo-utils[all]>=1.0.5`。不要误用同一个主仓里 MiniCPM-V 4.6 对应的 Transformers 5.x 安装说明。视频工具另需 FFmpeg。[MiniCPM-o 4.5 初始化说明](https://huggingface.co/openbmb/MiniCPM-o-4_5#offline-inference-examples-with-transformers)

## 7. 许可边界

当前模型卡声明 MiniCPM-o/V 的模型权重与主仓代码采用 Apache-2.0。不要沿用旧模型版本的商用登记要求；再分发应保留许可证、版权等适用声明，并标明修改。[模型声明](https://huggingface.co/openbmb/MiniCPM-o-4_5#license)、[主仓 LICENSE](https://github.com/OpenBMB/MiniCPM-V/blob/main/LICENSE)

[Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) 为 Apache-2.0，[llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) 为 MIT。核实时独立 `MiniCPM-o-Demo` 仓库的 GitHub license 元数据为 null，目录树未见 LICENSE；不能把主仓许可自动套到整个 Demo 的商业再分发。此处 bridge 为独立实现的协议客户端，没有复制整套 Demo。商业打包 Demo 前需确认该仓库具体许可。

## 8. 验证结果及限制

```bash
cd /mnt/c/Users/dengy/Desktop/gizz/cyberfly
python3 -m unittest discover -s bridge/tests -v
```

已通过 25 项桥接协议测试（其中包括神经调制、动态场景合同和自然对话）：默认不联网、官方云缺少 key 时失败、配置校验、严格任务 JSON、真实 localhost HTTP 请求的路径/模型/鉴权/截图、HTTP 错误、拒绝重定向、拒绝伪成功文本、真实 localhost WebSocket 的队列与会话生命周期。WebSocket 测试需安装可选依赖，未安装时会明确显示跳过。

协议测试使用人工构造的服务响应，不加载模型、不训练果蝇。独立真实推理记录包括 `model_runtime/inference_verification.json`、`vision_verification.json`、`realtime_verification.json`、`scenario_planning_verification.json`；`bridge/neural_drive_verification.json` 保存两种实际模型端口输出与一次真实自然对话，附真实观测来源。该桥接验证读取独立保存的神经/身体快照，没有声称已经完成同步闭环。模型第一条解释曾将 0.7 左输入误算成 10mV；证据保留此错误，实际数值 .7 通过校验且由适配器映射为 7mV。提示词现要求解释不自行换算电流。
