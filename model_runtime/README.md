# 已部署的本地 MiniCPM-o 4.5

2026-09-13 已在本机 RTX 3090 的 **GPU 2** 完成真实模型部署与推理。当前本地网关为 `http://127.0.0.1:18645/v1`，模型名为 `MiniCPM-o-4.5-Q4_K_M`。网关和 C++ 后端都只监听本机地址，不需要云 API key。

## 实际完成了什么

- 下载官方 Q4_K_M 全套 10 个 GGUF 文件，共 8,872,297,088 字节；包括 LLM、视觉、音频、TTS 与 Token2Wav。每个文件都按固定 Hugging Face revision 的 LFS SHA256 校验通过。
- 使用 CUDA 12.9，为 3090 的计算能力 8.6 编译官方 `llama.cpp-omni`；源码 revision 固定在 `64d092c60db4b4ee45768476bd752f03fdcc98ea`。
- 真实中文请求通过原有 `MiniCPMBridge` HTTP 接口返回合法的导航任务。首次加载加响应约 5 秒，后续请求更快；这些是本机测量，不是普适速度保证。
- 真实视觉请求读取项目的三维果蝇截图，识别了红色复眼、绿色小球和棋盘格反光地面。
- 统一 WebSocket `mode=chat` 也完成真实推理验证。

文本/图像初始常驻显存约 7.6GiB；加入原生语音模型后约 11GiB，仅使用 GPU 2；GPU 0/1 上已有服务没有被停止或修改。权重与隔离环境位于 `/root/.cache/cyberfly/minicpm-runtime`，没有把大模型文件塞进 C 盘项目目录。

当前公开网关已支持文字、图片、PCM16 WAV 语音输入及 MiniCPM 自身的语音输出。`bridge.transcribe(audio_bytes)` 返回带 `text` 的字典，`bridge.speak(text)` 返回可播放的 24kHz 单声道 PCM16 WAV 字节。HTTP 为 `/v1/audio/transcriptions` 和 `/v1/audio/speech`，仅默认音色。**轮次式语音已经实测，原生持续语音输入→分块语音输出→取消→新输入已实际验证，协议见 `bridge/DUPLEX_API.md`；取消重置上下文，不宣称无缝保留历史。** ASR 和朗读可能有字词误差；模型验证不代表果蝇策略已训练或完成任务。

## 启动、检查、停止

```bash
cd /mnt/c/Users/dengy/Desktop/gizz/cyberfly
bash model_runtime/start.sh
python3 model_runtime/manage.py status
python3 model_runtime/manage.py smoke
bash model_runtime/stop.sh
```

启动脚本不碰已有服务；端口已被其它进程占用，或 GPU 2 空闲显存不足时会拒绝启动。用 GPU UUID 锁定 `nvidia-smi` 的 GPU 2，避免 CUDA 枚举变化选错卡。停止脚本只结束当前运行时记录且进程身份匹配的进程。

`start` 的状态先是 `running_unverified`，因为健康接口只说明服务活着。`smoke` 真正请求模型并严格检查目标坐标，成功后才更新 `runtime.json` 为 `verified`。`stop` 会改为 `stopped`。这是手动启动服务，未安装系统开机自启。

如缓存被移走或系统重装，可重新构建和下载：

```bash
bash model_runtime/build.sh
python3 model_runtime/download_models.py
bash model_runtime/start.sh
python3 model_runtime/manage.py smoke
```

重建脚本针对当前 Linux / CUDA 12.9 / 3090 环境；依赖 `git`、`uv`、C++ 编译器和已有 CUDA Toolkit。它只操作独立缓存目录，固定源代码版本，不覆盖不同 revision 的现有源码。下载可断点续传，默认走 OpenBMB 的 ModelScope 公共镜像，最终始终按固定 Hugging Face 文件哈希校验；可设置 `CYBERFLY_MODEL_SOURCE=huggingface` 直接从 Hugging Face 下载。

## 工作台或其他程序调用

`runtime.json` 是无密钥的可机读配置。现有 bridge 的显式环境配置仍可优先选择其它模型服务：

```bash
export MINICPM_TRANSPORT=openai
export MINICPM_BASE_URL=http://127.0.0.1:18645/v1
export MINICPM_MODEL=MiniCPM-o-4.5-Q4_K_M
unset MINICPM_API_KEY
python3 -m bridge plan '把目标设为世界坐标 [-2.5,4.2] 毫米'
```

直接神经模式使用 `bridge.plan_neural_drive(user_text, state, brain_ports, image_path=None)`，返回三端口有界调制；自然对话使用 `bridge.chat(...)`，返回 `.say`。完整协议与工程电流边界见 `../MINICPM_INTEGRATION.md`。

兼容跨场景程序入口：

```python
from bridge import BridgeConfig, MiniCPMBridge

bridge = MiniCPMBridge(BridgeConfig(
    transport="openai",
    base_url="http://127.0.0.1:18645/v1",
    model="MiniCPM-o-4.5-Q4_K_M",
))
plan = bridge.plan_for_scenario(
    "把当前回合步数上限设为30",
    state=env.task_state(),
    scenario_spec=env.scenario_spec,
)
# 只有 configure 才能申请场景参数更新；环境负责最终类型与范围验证。
if plan.operation == "configure":
    env.apply_task(plan.parameters)
```

跨场景输出包含 `operation`、`parameters`、`say`、`provider`、`model`、`status`。操作只有 `configure / pause / reset / explain`；参数键必须在场景的 `task_updates` 白名单内。模型不直接操纵关节、替代游戏动作或更新策略权重；环境动作仍由独立策略负责。

JSON 结构合格不代表每个参数的领域类型都正确。例如模型可能把坐标写成字符串，或误解场景限制；`env.apply_task` 必须保留严格检查。失败应显示真实错误，不能转成伪成功回复。

## 证据与日志

| 文件 | 内容 |
|---|---|
| `runtime.json` | 当前服务配置和验证状态，不含密钥 |
| `inference_verification.json` | 实际模型加载与导航 JSON 验证 |
| `additional_inference_verification.json` | 新坐标与通过 bridge 的截图请求 |
| `vision_verification.json` | 实际果蝇截图与真实视觉回复 |
| `realtime_verification.json` | 实际 WebSocket chat 推理 |
| `scenario_planning_verification.json` | 果蝇导航与 Doom 参数的实际跨场景规划 |
| `provenance.json` / `weights-manifest.json` | 源码、模型版本、哈希、硬件与来源 |
| `localhost-bind.patch` | 上游服务尊重 `--host` 参数的一行隔离修复 |
| `multimodal-text.patch` | 修复原生分支丢弃音频/图片附带用户文字的问题 |
| `../artifacts/lab-integration/minicpm-multimodal-regression.json` | 同图不同文字的真实神经端口输出回归 |
| `../artifacts/lab-integration/minicpm-voice-to-neural.json` | 实际 TTS→WAV→转写→神经输入规划 |

旧 `vision_verification.json` 和 `additional_inference_verification.json` 在文字保留修复前生成，只证明看图能力；不能用作按指定问题进行多模态控制的证据。对应失败转写保留在 `audio_verification/bridge_roundtrip_initial_failure.json`。修复后的真实测试见上表。

服务日志为 `/root/.cache/cyberfly/minicpm-runtime/backend.log` 和 `gateway.log`；构建/下载日志为同目录下 `build.log`、`configure.log`、`download.log`。

## 官方来源和许可

[模型卡](https://huggingface.co/openbmb/MiniCPM-o-4_5)、[官方 GGUF 权重](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)、[OpenBMB 的 ModelScope 镜像](https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf)、[llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)、[官方 Demo 当前协议](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs-app/content/docs/en/realtime-api/overview.md)。模型采用 Apache-2.0，C++ 引擎采用 MIT。本网关独立实现，没有复制整个独立 Demo 仓库。
