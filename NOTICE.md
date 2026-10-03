# CyberFly-01 Notices

This release combines original CyberFly runtime/documentation with upstream model, connectome, physics, and simulation components. Keep the notices below with any redistribution and check each upstream repository for the current license text and terms.

## Upstream components

- **MiniCPM-o 4.5** — OpenBMB, `openbmb/MiniCPM-o 4_5` and the associated MiniCPM repositories. The base-derived baked model remains subject to the upstream model license and usage terms. Provenance is pinned in the technical report and model manifest.
- **MiniCPM source** — OpenBMB, source revision recorded in `sources.lock.json`.
- **MaleCNS / flybrain** — the connectome implementation and annotations from `snedea/flybrain`; use the license and data terms supplied by that project.
- **FlyBody** — TuragaLab, `TuragaLab/flybody`; retain its license and attribution.
- **FlyGym** — NeLy-EPFL, `NeLy-EPFL/flygym`; retain its license and attribution.
- **MuJoCo** — DeepMind / MuJoCo contributors; retain the license and notices of the installed version.
- **ViZDoom / DoomFly (optional scenarios)** — retain the notices and license of the installed upstream packages when those scenarios are used.

Exact upstream URLs and pinned source revisions are recorded in `sources.lock.json`. This file is a routing notice, not a replacement for any upstream LICENSE file.

## CyberFly additions

The original CyberFly runtime glue, protocol documentation, configuration examples, and report text in this repository are the CyberFly project additions. No standalone LoRA adapter, optimizer state, private dataset, or unreleased training artifact is included in this public runtime release.

Before publishing a binary or model mirror, include the corresponding upstream model card and license text where the upstream terms require it. Do not imply that this notice grants rights to upstream weights, datasets, or simulator assets.
