[中文 / Chinese](TECH_REPORT.zh-CN.md) · English

# CyberFly-01: MiniCPM-o 4.5 × MaleCNS × MuJoCo

## Technical report — pre-release v0.1 (2026-10-04)

### Abstract

CyberFly-01 is an auditable embodied-AI interface that connects MiniCPM-o 4.5, a complete male *Drosophila melanogaster* connectome implementation (MaleCNS), and an articulated MuJoCo fly body. The public model artifact is a **baked model**: the CyberFly v3 language LoRA has been merged into the MiniCPM language backbone. The adapter is not distributed separately. The public runtime code is kept separate from training code, private datasets, and internal experiment artifacts.

The system is designed to make the path from model input to neural state and body motion inspectable and replayable. It does not claim a conscious agent, a wet-brain reconstruction, or biological equivalence.

### System

The runtime path is:

```text
text / image / audio
        ↓
MiniCPM-o 4.5 (public baked checkpoint)
        ↓ 4096-dimensional hidden state
ID-conditioned engineering projection
        ↓
17,336 sensory entry IDs → MaleCNS full graph
        ↓
166,700 neurons / 25,582,938 aggregate directed edges
        ↓
2,129 motor or descending output IDs
        ↓
readout and body interface → MuJoCo FlyBody
```

The neural interval is 28.6 ms. The body uses a 10 ms outer step and 0.05 ms MuJoCo solver substep in the validated scenarios. The body model retains 78 actuators, 103 physical joints, and six wing axes. The nine-dimensional public action interface is an engineering control surface; it does not mean that all 78 actuators are independently learned by the connectome.

The model, connectome, readout, and body are connected through explicit numeric ports. A numeric influence through a port is not evidence that the model has read a subjective thought or that the biological circuit has acquired language semantics.

### What is baked

The released checkpoint is derived from `openbmb/MiniCPM-o-4_5`, pinned for provenance at revision `503e754207c94da6bb26850b4469f367c9ea3582`. CyberFly v3 used a short, language-only QLoRA/LoRA update:

- 150 supervised optimization steps;
- 420 training examples and 20 evaluation examples in the training run;
- 1,916,928 trainable parameters;
- the base weights and multimodal modules were frozen;
- the language adapter was merged into the base language backbone for release.

The public artifact contains the merged/baked checkpoint and its tokenizer/configuration files. It does **not** contain `adapter_model.safetensors`, PEFT adapter configuration, optimizer state, training data, or private training scripts. The short update validates the release and interface mechanics; it is not evidence of broad task mastery.

### Evidence and interpretation

The project keeps protocol-scoped evidence rather than combining incompatible measurements. The strongest validated statements are:

- the full graph and its declared sensory and motor ports can be loaded, stepped, and audited;
- the articulated body exposes the declared actuator/joint/wing structure, and physical substeps show condition-dependent multi-joint and attitude changes;
- frozen and trained actors have passed selected, finite-horizon hover and body-link checks on sealed initial states;
- a unified current/readout interface has reproduced short-window four-axis direction transfer under its declared protocol;
- native text direction and learned-current direction are separate interfaces, and native ±Z text requests currently collapse into existing templates rather than proving learned Z control;
- the corrected M4b memory protocol did not support a positive odor-specific effect under its declared test (state-randomized batch: group difference −0.20, exact label-permutation p=0.9326);
- the v3 text holdout result (40/40 on the fixed text set) is a text decision check and must not be reported as flight success;
- earlier M5/M6 numbers are protocol-scoped historical benchmarks and are not evidence that the native language model learned unrestricted flight.

No single demo clip should be read as proof that one native LLM call drove every body clip. Body, brain, readout, and model evidence are replayable components with explicit interfaces; some public visualizations combine separately recorded evidence layers.

### Meaning

CyberFly-01 is useful as a research instrument because it gives an observable, intervenable, and replayable interface across three scales:

1. a multimodal language model produces a measurable internal signal;
2. an explicit projection injects that signal into a complete connectome simulation;
3. a physical simulator exposes the downstream signal as joint, wing, contact, and attitude trajectories.

This makes it possible to ask where a behavior changes, reproduce the same initial state, and compare a signal path against zero-current, swapped, frozen-template, or other declared controls. It does not establish consciousness, sentience, a biological fly mind, or a one-to-one mapping between language and natural neural activity.

### Reproduction

1. Download the public baked model from the model host and verify its supplied SHA-256 manifest.
2. Install the public runtime dependencies listed in `requirements-lab.lock.txt`.
3. Point a local MiniCPM-compatible gateway at the baked model. The public runtime package does not silently download weights or contact a remote API.
4. Use `bridge` or `cyberfly_runtime.py` to send a validated request. Use `scenarios` and `connectome_adapter` for explicit neural/body experiments.
5. Record model revision, graph/checkpoint identity, protocol, seed, and the complete output trajectory. Do not infer an outcome from the generated explanation alone.

The runtime package intentionally omits training code, private datasets, large internal experiment artifacts, and the separate LoRA adapter. Reproducing a historical result requires the protocol and evidence bundle named by that result; the public baked model alone is not sufficient to reproduce every internal experiment.

### Release contents

- **Public runtime repository:** bridge, connectome adapter, scenario interfaces, configuration examples, and documentation. It is ordinary execution code, not the training stack.
- **Public model repository:** the merged/baked MiniCPM checkpoint, tokenizer/configuration, provenance, model card, and checksums. No standalone LoRA adapter is published.
- **Technical report:** this document, with evidence boundaries and known limitations.

### Citation

If you use CyberFly-01, cite the release repository and model revision together with the upstream projects. The upstream MiniCPM-o 4.5, MaleCNS/flybrain, FlyBody, FlyGym, and MuJoCo licenses and attribution requirements remain applicable. See `NOTICE.md` before redistribution.

