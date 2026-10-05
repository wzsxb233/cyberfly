# CyberFly-01 Baked Model Card / 模型卡

CyberFly-01 is a merged/baked MiniCPM-o 4.5 checkpoint for an auditable model-to-connectome-to-body interface.

CyberFly-01 是一个合并后的 MiniCPM-o 4.5 baked checkpoint，用于可审计的模型—连接组—身体接口。

- Base / 基础模型: `openbmb/MiniCPM-o-4_5`
- Revision / 固定版本: `503e754207c94da6bb26850b4469f367c9ea3582`
- Update / 更新: 150 steps, 420 train examples, 20 training-evaluation examples
- Release / 发布形式: merged weights only; no standalone LoRA adapter / 仅合并权重，不发布独立 LoRA

Use it for controlled embodied-AI, connectome-interface and replayable simulation research. Do not present it as consciousness, a biological fly brain, clinical software or proof of native language flight.

可用于受控具身智能、连接组接口和可重放仿真研究。不要将其表述为意识、生物果蝇大脑、临床软件或原生语言飞行证明。

The baked-only artifact does not include upstream MiniCPM `assets/token2wav/` files. Native speech requires those files from the upstream distribution.

baked-only 发布包不包含上游 MiniCPM 的 `assets/token2wav/` 文件；原生语音需要从上游发行包另行提供这些文件。
