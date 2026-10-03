"""Actual original perception tokens -> bounded, independently addressed cells.

These CPU adapters are engineering interfaces, not recovered biological wiring.
Vision targets annotated ol_sensory cells; audio targets vnc_sensory candidates,
which must not be described as a verified natural fly auditory pathway.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import uuid
import zipfile

import numpy as np
import torch
from torch import nn

from connectome_adapter.neuron_currents import write_neuron_currents

ROOT = Path(__file__).resolve().parents[1]
TOKEN_ROOT = ROOT / "artifacts/shared_modal_embeddings"
GRAPH = ROOT / "vendor/doomfly/outputs/doom/malecns_v1/graph.npz"
N = 166700
IDS_SHA = "6b6b40c3bddf84b1281ef0b18db06927c2bf61c5f4cb32a4219ec9b7839dc2e5"
REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
MAX_BYTES = 70 * 1024 * 1024
SELECTIONS = {
    "vision": ("ol_sensory", 6098, "e11bcd3f4504c02522e9b748b5e688d23d16c042d05c649d0eb41950d7e22f7e"),
    "audio": ("vnc_sensory", 6370, "b6c53381507c727c5621579cc077645585e095c0b9053380d2484be6aa69347b"),
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def array_sha(value):
    return sha(np.ascontiguousarray(value).tobytes())


def model_identity():
    return {"model_revision": REVISION,
            "weights_manifest_sha256": sha((ROOT / "model_training/weights-manifest.json").read_bytes())}


def read_modal_embeddings(descriptor, *, event_id, expected_model_identity):
    """Read the same bounded immutable bytes whose file/tensor hashes are checked.

This verifies recorded identity, not that an arbitrary local author truly ran a
model. Provenance comes from the model service's event and runtime evidence.
"""
    if not isinstance(descriptor, dict) or descriptor.get("schema") != 1:
        raise ValueError("modal_embeddings_file requires schema 1")
    if not isinstance(event_id, str) or not 1 <= len(event_id) <= 128 or descriptor.get("event_id") != event_id:
        raise ValueError("Perception tokens belong to a different event")
    identity = model_identity()
    if not isinstance(expected_model_identity, dict) or any(expected_model_identity.get(k) != v for k, v in identity.items()):
        raise ValueError("Unexpected original model revision/weights identity")
    if any(descriptor.get(k) != v for k, v in identity.items()):
        raise ValueError("Perception tokens belong to a different model identity")
    supplied = Path(descriptor.get("path", ""))
    path = supplied.resolve(strict=True)
    if supplied.is_symlink() or not path.is_relative_to(TOKEN_ROOT.resolve()) or path.suffix != ".npz":
        raise ValueError("Perception token path must be an immutable workspace modal NPZ")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
            raise ValueError("Perception tokens exceed the regular-file size limit")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or sha(raw) != descriptor.get("sha256"):
        raise ValueError("Perception token file SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 3 or {item.filename for item in members} != {"vision.npy", "audio.npy", "language.npy"}:
            raise ValueError("Modal NPZ must contain vision, audio, language only")
        if any(item.flag_bits & 1 for item in members) or sum(item.file_size for item in members) > MAX_BYTES:
            raise ValueError("Oversized or encrypted modal NPZ")
    arrays = {}
    declared = descriptor.get("arrays", {})
    if not isinstance(declared, dict) or set(declared) != {"vision", "audio", "language"}:
        raise ValueError("Missing exact modal array descriptors")
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=1024) as archive:
        for key in ("vision", "audio", "language"):
            value = archive[key].copy()
            if value.dtype != np.dtype("<f4") or value.ndim != 2 or value.shape[1] != 4096 or value.shape[0] > 4096:
                raise ValueError("Modal arrays must be float32[T<=4096,4096]")
            if key == "language" and value.shape != (1, 4096):
                raise ValueError("Language summary must contain the actual single last hidden")
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite modal token")
            item = declared[key]
            if not isinstance(item, dict) or item.get("shape") != list(value.shape) or item.get("dtype") not in ("float32", "<f4") or item.get("sha256") != array_sha(value):
                raise ValueError("Modal token shape/dtype/SHA256 differs from descriptor")
            arrays[key] = value
    if sum(value.shape[0] for value in arrays.values()) > 4097:
        raise ValueError("Combined original modal tokens exceed the recorded context limit")
    return arrays


class _Attention(nn.Module):
    def __init__(self, count, rank):
        super().__init__()
        self.norm = nn.LayerNorm(4096)
        self.keys = nn.Linear(4096, rank, bias=False)
        self.values = nn.Linear(4096, rank, bias=False)
        self.query = nn.Parameter(torch.empty(count, rank))
        self.readout = nn.Parameter(torch.empty(count, rank))
        self.bias = nn.Parameter(torch.zeros(count))
        nn.init.normal_(self.query, std=0.1)
        nn.init.normal_(self.readout, std=0.05)
        self.rank = rank

    def forward(self, tokens, amplitude, *, chunk_size=512):
        normalized = self.norm(tokens)
        keys, values = self.keys(normalized), self.values(normalized)
        results = []
        for start in range(0, len(self.query), chunk_size):
            stop = start + chunk_size
            attention = torch.softmax(self.query[start:stop] @ keys.T / math.sqrt(self.rank), dim=-1)
            attended = attention @ values
            logits = (attended * self.readout[start:stop]).sum(dim=-1) + self.bias[start:stop]
            results.append(float(amplitude) * torch.tanh(logits))
        return torch.cat(results)


class ModalCurrentAdapter:
    """One independently saved/trained route; original MiniCPM stays unchanged.

fit learns explicit current labels by SGD, without reward claims or gradients
through the original encoder, external fly simulation, or physical body.
"""
    def __init__(self, modality, *, seed=42, rank=16, max_current_mv=1.0):
        if modality not in SELECTIONS or type(rank) is not int or rank != 16:
            raise ValueError("Choose vision/audio and rank 16")
        if type(max_current_mv) not in (int, float) or not math.isfinite(max_current_mv) or not 0 < max_current_mv <= 1:
            raise ValueError("Modal additions are bounded to at most 1 mV-equivalent")
        superclass, count, selected_sha = SELECTIONS[modality]
        with np.load(GRAPH, allow_pickle=False) as graph:
            self.ids, classes = graph["ids"].copy(), graph["superclass"].copy()
        if self.ids.dtype != np.dtype("<i8") or self.ids.shape != (N,) or array_sha(self.ids) != IDS_SHA or classes.shape != (N,):
            raise ValueError("Modal route requires the complete verified graph ID order")
        self.indices = np.flatnonzero(classes == superclass)
        self.neuron_ids = self.ids[self.indices]
        if len(self.indices) != count or array_sha(self.neuron_ids) != selected_sha:
            raise ValueError("Modal annotation selection changed")
        self.modality, self.rank, self.max_current_mv = modality, rank, float(max_current_mv)
        self.seed, self.optimizer_updates, self.label_source = int(seed), 0, None
        self.checkpoint = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = _Attention(count, rank).cpu()
        self.network.eval()

    def _weight_sha(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.network.state_dict().items()):
            digest.update(name.encode() + b"\0")
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def describe(self):
        superclass, count, selected_sha = SELECTIONS[self.modality]
        configuration = {"schema": 1, "kind": "original_modal_token_neuron_attention", "modality": self.modality,
                "rank": self.rank, "max_current_mv": self.max_current_mv, "neurons": N,
                "graph_ids_sha256": IDS_SHA, "selected_neurons": count, "superclass": superclass,
                "selected_ids_sha256": selected_sha, **model_identity()}
        return {**configuration, "configuration_sha256": sha(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()), "seed": self.seed,
                "weights_sha256": self._weight_sha(), "optimizer_updates": self.optimizer_updates,
                "optimizer": "SGD without momentum; no hidden optimizer state", "label_source": self.label_source,
                "checkpoint": self.checkpoint, "device": "cpu", "original_model_trainable": False,
                "biological_mapping_claim": False,
                "interpretation": "Per-ID learned engineering queries over original projected modality tokens; not a natural auditory or visual wiring reconstruction"}

    def _selected(self, tokens):
        if not isinstance(tokens, np.ndarray) or tokens.dtype != np.dtype("<f4") or tokens.ndim != 2 or tokens.shape[1] != 4096 or not 1 <= tokens.shape[0] <= 4096 or not np.isfinite(tokens).all():
            raise ValueError("Inference needs nonempty finite original float32[T,4096] tokens")
        return self.network(torch.from_numpy(np.ascontiguousarray(tokens)), self.max_current_mv)

    def project(self, descriptor, *, event_id, model_identity):
        arrays = read_modal_embeddings(descriptor, event_id=event_id, expected_model_identity=model_identity)
        tokens = arrays[self.modality]
        info = {"event_id": event_id, "modality": self.modality, "source_file_sha256": descriptor["sha256"],
                "tokens_sha256": array_sha(tokens), "token_shape": list(tokens.shape), "adapter_identity": self.describe()}
        if not len(tokens):
            return {**info, "active": False, "currents_file": None, "reason": "Actual event has no tokens for this modality"}
        with torch.no_grad():
            selected = self._selected(tokens).numpy()
        currents = np.zeros(N, dtype=np.float32)
        currents[self.indices] = selected
        if not np.isfinite(currents).all() or np.max(np.abs(currents)) > self.max_current_mv:
            raise RuntimeError("Modal attention produced invalid current")
        return {**info, "active": True, "currents_file": write_neuron_currents(currents, self.ids),
                "selected_neurons": len(self.indices), "nonzero_neurons": int(np.count_nonzero(currents)),
                "outside_selection_nonzero": 0, "max_abs_current_mv": float(np.max(np.abs(selected))),
                "unique_selected_current_values": int(np.unique(selected).size)}

    def fit(self, examples, *, model_identity, label_source, steps=2, learning_rate=0.05, progress_callback=None):
        """Examples: {event_id, modal_embeddings_file, target_currents_mv[M]}.

Labels are user/application supervision in selected graph-ID order. They are
not inferred from a file name and never called human/expert or biological data.
        """
        if not isinstance(label_source, str) or not 1 <= len(label_source.strip()) <= 200:
            raise ValueError("An explicit nonempty training label_source is required")
        if type(steps) is not int or not 1 <= steps <= 1000 or type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.1:
            raise ValueError("Invalid bounded training steps/rate")
        if not isinstance(examples, (list, tuple)) or not 1 <= len(examples) <= 64:
            raise ValueError("Provide 1..64 labeled actual token events")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        dataset, sources = [], []
        for example in examples:
            descriptor, event = example["modal_embeddings_file"], example["event_id"]
            arrays = read_modal_embeddings(descriptor, event_id=event, expected_model_identity=model_identity)
            tokens = arrays[self.modality]
            target = np.asarray(example["target_currents_mv"], dtype=np.float32)
            if not len(tokens) or target.shape != (len(self.indices),) or not np.isfinite(target).all() or np.max(np.abs(target)) > self.max_current_mv:
                raise ValueError("Training requires real modality tokens and bounded per-selected-ID current labels")
            dataset.append((tokens, torch.from_numpy(target.copy())))
            sources.append({"event_id": event, "file_sha256": descriptor["sha256"],
                            "tokens_sha256": array_sha(tokens), "target_sha256": array_sha(target)})
        before = {name: value.detach().clone() for name, value in self.network.named_parameters()}
        initial_sha = self._weight_sha()
        self.network.train()
        optimizer = torch.optim.SGD(self.network.parameters(), lr=float(learning_rate))
        losses, grad_norms, interrupted = [], [], False
        try:
            for index in range(steps):
                tokens, target = dataset[index % len(dataset)]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((self._selected(tokens) - target) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite modal loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0, error_if_nonfinite=True)
                grad_norms.append(float(norm)); losses.append(float(loss.detach()))
                optimizer.step()
                self.optimizer_updates += 1
                if progress_callback is not None:
                    changed_now = sum(int(torch.count_nonzero(value.detach() != before[name])) for name, value in self.network.named_parameters())
                    delta_now = math.sqrt(sum(float(torch.sum((value.detach() - before[name]).double() ** 2)) for name, value in self.network.named_parameters()))
                    if progress_callback({'step': len(losses), 'optimizer_updates': self.optimizer_updates,
                        'loss': losses[-1], 'grad_norm': float(norm), 'changed_parameters': changed_now,
                        'parameter_l2_change': delta_now}) is False:
                        interrupted = True
                        break
        finally:
            self.network.eval()
        with torch.no_grad():
            final_losses = [float(torch.mean((self._selected(tokens) - target) ** 2)) for tokens, target in dataset]
        changed = sum(int(torch.count_nonzero(value.detach() != before[name])) for name, value in self.network.named_parameters())
        delta = math.sqrt(sum(float(torch.sum((value.detach() - before[name]).double() ** 2)) for name, value in self.network.named_parameters()))
        self.label_source = label_source
        return {"status": "interrupted" if interrupted else "actual_supervised_parameter_updates", "modality": self.modality,
                "steps": len(losses), "requested_steps": steps, "interrupted": interrupted,
                "optimizer_updates": self.optimizer_updates, "label_source": label_source,
                "changed_parameters": changed, "parameter_delta_l2": delta, "parameter_l2_change": delta, "gradient_norms": grad_norms,
                "step_training_mse": losses, "final_training_mse": final_losses,
                "weights_sha256_before": initial_sha, "weights_sha256_after": self._weight_sha(),
                "source_events": sources, "heldout_evaluation": False,
                "claim": "Supervised engineering current fitting only; no behavioral improvement or natural sensory mapping claim"}

    def save(self, path):
        """Commit a new immutable directory with exact IDs and weight hashes."""
        destination = Path(path).resolve()
        if not destination.is_relative_to(ROOT) or destination.exists():
            raise ValueError("Use a new checkpoint directory inside this workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
        stage.mkdir()
        try:
            values = {name: value.detach().cpu().numpy() for name, value in self.network.state_dict().items()}
            values["neuron_ids"] = self.neuron_ids
            np.savez(stage / "weights.npz", **values)
            metadata = self.describe()
            metadata["checkpoint"] = str(destination)
            metadata["file_sha256"] = sha((stage / "weights.npz").read_bytes())
            (stage / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            stage.rename(destination)
        finally:
            if stage.exists():
                for file in stage.iterdir():
                    file.unlink()
                stage.rmdir()
        self.checkpoint = str(destination)
        return self.describe()

    @classmethod
    def load(cls, path):
        directory = Path(path).resolve()
        if not directory.is_relative_to(ROOT):
            raise ValueError("Checkpoint must be inside the workspace")
        metadata = json.loads((directory / "metadata.json").read_text())
        adapter = cls(metadata.get("modality"), seed=metadata.get("seed"), rank=metadata.get("rank"),
                      max_current_mv=metadata.get("max_current_mv"))
        expected = adapter.describe()
        for key in ("schema", "kind", "neurons", "graph_ids_sha256", "selected_neurons", "selected_ids_sha256",
                    "superclass", "model_revision", "weights_manifest_sha256", "optimizer", "configuration_sha256"):
            if metadata.get(key) != expected[key]:
                raise ValueError("Modal checkpoint identity mismatch: " + key)
        file = directory / "weights.npz"
        if not 0 < file.stat().st_size <= 8 * 1024 * 1024:
            raise ValueError("Invalid modal checkpoint size")
        raw = file.read_bytes()
        if sha(raw) != metadata.get("file_sha256"):
            raise ValueError("Modal checkpoint file SHA256 mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if sum(item.file_size for item in z.infolist()) > 8 * 1024 * 1024:
                raise ValueError("Oversized modal checkpoint members")
        state = adapter.network.state_dict()
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != set(state) | {"neuron_ids"} or not np.array_equal(archive["neuron_ids"], adapter.neuron_ids) or archive["neuron_ids"].dtype != np.dtype("<i8"):
                raise ValueError("Modal checkpoint neuron IDs or parameter keys changed")
            for name, template in state.items():
                value = archive[name].copy()
                if value.dtype != np.dtype("<f4") or value.shape != tuple(template.shape) or not np.isfinite(value).all():
                    raise ValueError("Invalid modal checkpoint parameter: " + name)
                state[name] = torch.from_numpy(value)
        adapter.network.load_state_dict(state, strict=True)
        if adapter._weight_sha() != metadata.get("weights_sha256"):
            raise ValueError("Modal parameter identity mismatch")
        if type(metadata.get("optimizer_updates")) is not int or metadata["optimizer_updates"] < 0:
            raise ValueError("Invalid saved modal optimizer counter")
        adapter.optimizer_updates = metadata["optimizer_updates"]
        adapter.label_source, adapter.checkpoint = metadata.get("label_source"), str(directory)
        return adapter
