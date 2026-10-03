# CyberFly-01 Baked Model Card

## Model summary

**CyberFly-01** is a merged/baked MiniCPM-o 4.5 checkpoint prepared for the CyberFly embodied-AI interface. It combines a short CyberFly v3 language-only QLoRA/LoRA update with the pinned MiniCPM-o 4.5 base language backbone. The public artifact is the merged checkpoint; no standalone LoRA adapter is released.

- **Base model:** `openbmb/MiniCPM-o-4_5`
- **Base revision:** `503e754207c94da6bb26850b4469f367c9ea3582`
- **Release form:** merged/baked model files plus tokenizer, configuration, provenance, and checksums
- **Multimodal base:** retained; multimodal modules were frozen during the CyberFly update
- **Training update:** 150 supervised steps; 420 train examples; 20 evaluation examples; 1,916,928 trainable parameters
- **Primary interface:** standard MiniCPM-o generation, with CyberFly runtime integration supplied separately

The release is intended to be loaded as a normal model checkpoint. A loader should not expect or require `adapter_model.safetensors`, a PEFT adapter directory, or an optimizer state.

## Intended use

Use this model for:

- research on auditable multimodal model interfaces;
- controlled experiments that feed a model signal into an explicit MaleCNS connector and MuJoCo body;
- replayable demonstrations and protocol-scoped evaluation of model output and downstream body/readout signals;
- educational inspection of how a language-model hidden state can be routed through declared numeric ports.

## Out-of-scope use

Do not present this checkpoint as:

- a conscious or sentient agent;
- a biological fly brain, a wet-brain reconstruction, or a biologically equivalent controller;
- a clinical, safety-critical, or autonomous flight system;
- proof that native language directions map reliably to world axes;
- proof that every body joint or wing actuator was independently learned;
- proof that a text holdout score is a flight score.

Run it only in a sandboxed, supervised environment. The model can generate incorrect text, plans, or explanations. The runtime must validate every numeric field and keep the body/environment responsible for range checks and termination.

## Data and training

CyberFly v3 was a short language-only update. The base model and its multimodal modules were frozen. The update was used to validate the release path and a fixed text decision check, not to establish broad behavioral competence. The separate training examples, optimizer state, and LoRA adapter are not part of this public model artifact.

The connectome and body are separate runtime components. They are not silently baked into MiniCPM weights. The standard checkpoint therefore does not, by itself, simulate a brain or control a MuJoCo body; use the companion runtime and an explicit protocol to run that interface.

## Evaluation notes

The fixed v3 text holdout reached 40/40 exact decisions in the declared text check. This result is limited to that text protocol. The learned-current physical gate did not pass the final 3 mm arrival/0.5 s hold under the declared candidate protocol. Native ±Z requests currently fall into existing ±Y templates. The corrected M4b state-randomized memory protocol did not show a positive odor-specific effect (group difference −0.20; exact label-permutation p=0.9326).

Historical M1–M6 results are protocol-scoped and remain in the technical report with their controls and limitations. They must not be merged into a general claim of native LLM flight ability.

## Usage

A model host should expose the model through a local, authenticated or loopback-only MiniCPM-compatible service. The companion runtime uses explicit configuration such as:

```bash
export MINICPM_TRANSPORT=openai
export MINICPM_BASE_URL=http://127.0.0.1:8000/v1
export MINICPM_MODEL=CyberFly-01
python3 -m bridge plan "解释当前果蝇状态" --state-json '{"position_mm":[0,0]}'
```

For connectome/body experiments, use the public `connectome_adapter` and `scenarios` APIs. The model response is a proposal or explanation; only validated environment actions are applied.

## Bias, safety, and limitations

This checkpoint inherits the upstream model's data, language, and multimodal limitations. CyberFly adds an engineering interface; it does not remove hallucination, distribution shift, or control risk. Keep the model offline or loopback-bound for experiments, validate schemas and numeric ranges, cap episode length, and retain raw trajectories.

All reported numbers are tied to the named checkpoint, protocol, seed, and control. The public baked checkpoint is not a replacement for a full reproduction bundle.

## License and attribution

The upstream MiniCPM-o 4.5 license and model terms apply to the base-derived weights. CyberFly documentation and original runtime additions are distributed only under the terms stated in `NOTICE.md`; do not assume that a downstream license supersedes upstream obligations. Cite MiniCPM-o 4.5 and the CyberFly release together with MaleCNS/flybrain, FlyBody, FlyGym, and MuJoCo when those components are used.
