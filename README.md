[中文 / Chinese](README.zh-CN.md) · English

# CyberFly-01 Runtime

CyberFly-01 connects a baked MiniCPM-o 4.5 checkpoint to an explicit MaleCNS connectome interface and an articulated MuJoCo fly body. This repository contains the **ordinary runtime code and protocol documentation**. It does not contain the training stack, private datasets, internal experiment artifacts, or a separate LoRA adapter.

The public model release contains only the merged/baked checkpoint. The adapter is intentionally absent: there is no `adapter_model.safetensors` to load or merge at runtime.

## Public release layout

- `bridge/` — validated model and planning clients;
- `connectome_adapter/` — explicit neural input/output ports and graph client;
- `scenarios/` — Gymnasium-compatible fly, neural-link, and sandbox interfaces;
- `shared_io/` — numeric model/brain/body contracts;
- `vllm_integration/` — compatible local gateway helpers;
- `configs/` — example lab configurations;
- `TECH_REPORT.md` — claims, evidence boundaries, and limitations;
- `MODEL_CARD.md` — baked model provenance and intended use;
- `NOTICE.md` — upstream attribution and redistribution notes.

## Install

Use a supported Python environment and install the pinned public dependencies:

```bash
python -m pip install -r requirements-lab.lock.txt
```

The lock file contains the lab runtime dependencies. MuJoCo, FlyGym, GPU runtimes, and a MiniCPM-compatible serving backend may require platform-specific installation. This repository does not include a model server, training stack, or automatic model downloader; start your approved local backend separately.

## Point the runtime at the baked model

First expose the baked checkpoint through a local MiniCPM-compatible gateway. Then configure the bridge explicitly:

```bash
export MINICPM_TRANSPORT=openai
export MINICPM_BASE_URL=http://127.0.0.1:8000/v1
export MINICPM_MODEL=CyberFly-01
unset MINICPM_API_KEY  # set it only when the local gateway requires one
```

The model repository supplies the baked checkpoint, tokenizer/configuration, provenance, and SHA-256 manifest. Load it as a normal MiniCPM-o checkpoint. Do not pass a PEFT adapter path and do not expect adapter files in the release. The baked artifact does not include MiniCPM ssets/token2wav/ files (low.pt, hift.pt, campplus.onnx, speech_tokenizer_v2_25hz.onnx); native speech requires those upstream assets installed separately.

## Minimal request

```bash
python3 -m bridge plan \
  "把目标设为世界坐标 [3,0] 毫米，并解释当前状态" \
  --state-json '{"position_mm":[0,0]}'
```

For Python clients:

```python
from bridge import BridgeConfig, MiniCPMBridge

bridge = MiniCPMBridge(BridgeConfig(
    transport="openai",
    base_url="http://127.0.0.1:8000/v1",
    model="CyberFly-01",
))
result = bridge.chat("解释当前果蝇状态")
print(result.say)
```

A returned plan is an interface proposal. The environment must validate operation names, parameters, ranges, termination, and physical state before applying anything.

## Neural and body experiments

Use `connectome_adapter` for explicit ports and `scenarios` for replayable environments. The validated interface retains:

- 166,700 simulated neurons and 25,582,938 aggregate directed edges;
- 17,336 declared sensory entry IDs and 2,129 motor/descending output IDs;
- 78 body actuators, 103 physical joints, and six wing axes;
- a 28.6 ms neural interval, 10 ms body outer step, and 0.05 ms MuJoCo solver substep in the declared scenarios.

These are engineering interfaces and model dimensions. They do not mean that all joints were independently learned or that the connectome has acquired language semantics. Read `TECH_REPORT.md` before interpreting a demo or combining historical benchmark numbers.

## What is intentionally not included

This public runtime release excludes:

- the standalone LoRA/QLoRA adapter;
- optimizer state and training scripts;
- private datasets and internal experiment artifacts;
- hidden evaluation labels and unreleased checkpoints;
- automatic remote model downloads or cloud credentials.

The baked model and this runtime are separate release artifacts. Their version, checksum, and protocol identity should be recorded together for a reproducible run.

## Evidence boundary

The project demonstrates an observable and replayable model → explicit neural interface → body path. It does not establish consciousness, biological equivalence, unrestricted native language flight, learned Z control, or independent learning of every joint. See the model card and technical report for the protocol-scoped text and physical checks, negative controls, and open gates.

## Attribution

CyberFly uses upstream MiniCPM-o 4.5, MaleCNS/flybrain, FlyBody, FlyGym, and MuJoCo components. Their licenses and attribution requirements remain in force; see `NOTICE.md`.

