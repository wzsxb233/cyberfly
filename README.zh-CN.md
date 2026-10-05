# CyberFly-01 Runtime / 运行时

CyberFly-01 connects a baked MiniCPM-o 4.5 checkpoint to an explicit MaleCNS connectome interface and an articulated MuJoCo fly body.

CyberFly-01 将 baked 版 MiniCPM-o 4.5、MaleCNS 果蝇连接组接口和 MuJoCo 果蝇身体连接起来。

## Public artifacts / 公开内容

- Model / 模型: <https://huggingface.co/GizzAI/CyberFly-01>, <https://modelscope.cn/models/GizzAI/CyberFly-01>
- Runtime / 运行代码: <https://github.com/wzsxb233/cyberfly>
- Technical report / 技术报告: [TECH_REPORT.md](TECH_REPORT.md) and [TECH_REPORT.zh-CN.md](TECH_REPORT.zh-CN.md)
- Paper draft / 论文草稿: [PAPER_DRAFT.md](PAPER_DRAFT.md)

The public model contains only the merged baked checkpoint, tokenizer, configuration, provenance and checksums. It does not contain a standalone LoRA adapter, training data, optimizer state or private experiments.

公开模型只包含合并后的 baked 权重、tokenizer、配置、来源信息和校验文件，不包含独立 LoRA、训练数据、优化器状态或私有实验资产。

## Run / 运行

Point a local MiniCPM-compatible service at the baked model, then use `bridge` and the frozen `connectome_adapter` protocols. The public package never starts a private training service or enables learning.

将本地 MiniCPM 兼容服务指向 baked 模型，再使用 `bridge` 和冻结的 `connectome_adapter` 协议。公开包不会启动私有训练服务，也不会开启学习。

Native speech requires MiniCPM's upstream `assets/token2wav/` files; those auxiliary files are intentionally not included in the baked-only artifact.

原生语音需要 MiniCPM 上游的 `assets/token2wav/` 辅助文件；baked-only 发布包有意不包含这些文件。
