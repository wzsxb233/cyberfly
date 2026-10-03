"""Complete neural reads and bounded writes, without modifying graph topology."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
import uuid
import zlib

import numpy as np

from .full_state import ROOT, REPO, array_digest, build_catalog, write_catalog
from .ports import DRIVE_LIMITS_MV, validate_general_stimulation
from .neuron_currents import load_neuron_currents


class FullBrainIO:
    def _init_full_io(self, annotations):
        self.catalog, self.full_layout = build_catalog(self.brain.ids, self.brain.superclass, annotations)
        self.catalog_path = write_catalog(self.catalog)
        self.group_lookup = {group["group_id"]: (level, group) for level, key in
                             (("fine", "groups"), ("macro", "macro_groups")) for group in self.catalog[key]}
        self.sorted_id_order = np.argsort(self.brain.ids)
        self.sorted_ids = self.brain.ids[self.sorted_id_order]
        self.cumulative_counts = np.zeros(self.brain.n, dtype=np.int64)
        self.cumulative_counts_complete = True
        self.last_general_stimulation = []
        self.last_visual_input_enabled = True
        self._init_group_reductions()

    def _init_group_reductions(self):
        """Build read-only partition caches without resetting any neural state."""
        # Graph partitions never change during a worker lifetime. Stable order
        # allows vectorized extrema while sums keep their original bincount
        # order, preserving the existing voltage-mean arithmetic.
        reductions = {}
        for level, labels_key, groups_key in (("macro", "macro_group_index", "macro_groups"),
                                               ("fine", "group_index", "groups")):
            labels = self.full_layout[labels_key]
            counts = np.bincount(labels, minlength=len(self.catalog[groups_key]))
            if np.any(counts == 0) or int(counts.sum()) != self.brain.n:
                raise ValueError("Every full-brain partition must cover all neurons with nonempty groups")
            order = np.argsort(labels, kind="stable")
            starts = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
            reductions[level] = (order, starts, counts)
        self._group_reductions = reductions

    def _indices_for_ids(self, values):
        if not isinstance(values, list) or not 1 <= len(values) <= self.brain.n:
            raise ValueError("neuron_ids must be a nonempty list no longer than the retained graph")
        parsed = []
        for value in values:
            if type(value) is int:
                parsed.append(value)
            elif isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value):
                parsed.append(int(value))
            else:
                raise ValueError("neuron_ids must contain original integer IDs or decimal ID strings")
        ids = np.asarray(parsed, dtype=np.int64)
        positions = np.searchsorted(self.sorted_ids, ids)
        valid = positions < len(self.sorted_ids)
        if not valid.all() or not np.array_equal(self.sorted_ids[positions], ids):
            raise ValueError("Stimulation refers to a neuron absent from the actual retained graph")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Repeated neuron IDs within one stimulus are not allowed")
        return self.sorted_id_order[positions].astype(np.int32)

    def _prepare_general_stimulation(self, value, normalized):
        value = validate_general_stimulation(value)
        current = np.zeros(self.brain.n, dtype=np.float32)
        seen = set()
        for stimulus_index, stimulus in enumerate(value):
            if not isinstance(stimulus, dict):
                raise ValueError("Each general stimulus must be an object")
            keys = set(stimulus)
            if keys == {"neuron_currents_file"}:
                vector = load_neuron_currents(stimulus["neuron_currents_file"], self.brain.ids, self.catalog["ids_sha256"])
                if stimulus_index == 0:
                    # Preserve exact float32 bytes, including masked -0.0, for
                    # the single-file identity check used by the numerical bus.
                    current = vector
                else:
                    current += vector
                continue
            if keys in ({"group_currents_mv"}, {"macro_group_currents_mv"}):
                key = next(iter(keys))
                if key in seen:
                    raise ValueError("Only one dense vector per partition is allowed")
                seen.add(key)
                macro = key == "macro_group_currents_mv"
                size = self.catalog["macro_group_count" if macro else "group_count"]
                vector = np.asarray(stimulus[key])
                if vector.shape != (size,) or vector.dtype.kind not in "iuf" or not np.isfinite(vector).all() or np.any(np.abs(vector) > 30):
                    raise ValueError("Dense group currents require one finite value within [-30,30] mV per catalog index")
                current += vector.astype(np.float32)[self.full_layout["macro_group_index" if macro else "group_index"]]
                continue
            if keys not in ({"group_id", "current_mv"}, {"neuron_ids", "current_mv"}):
                raise ValueError("Use group_id/current_mv, neuron_ids/current_mv, or a catalog-ordered dense current vector")
            amplitude = stimulus["current_mv"]
            if type(amplitude) not in (int, float) or not math.isfinite(amplitude) or not -30 <= amplitude <= 30:
                raise ValueError("General neural current must be finite within [-30,30] mV-equivalent")
            if "group_id" in stimulus:
                group_id = stimulus["group_id"]
                if not isinstance(group_id, str) or group_id not in self.group_lookup or group_id in seen:
                    raise ValueError("General stimulus group_id must identify a unique actual fine or macro group")
                seen.add(group_id)
                level, group = self.group_lookup[group_id]
                labels = self.full_layout["macro_group_index" if level == "macro" else "group_index"]
                indices = np.flatnonzero(labels == group["index"])
            else:
                indices = self._indices_for_ids(stimulus["neuron_ids"])
            current[indices] += float(amplitude)
        combined = current.copy()
        for name, amplitude in normalized.items():
            combined[self.port_indices[name]] += amplitude * DRIVE_LIMITS_MV[name]
        if not np.isfinite(combined).all() or np.any(np.abs(combined) > 30.00001):
            raise ValueError("Overlapping direct inputs exceed the per-neuron total [-30,30] mV-equivalent limit")
        return current, json.loads(json.dumps(value, allow_nan=False))

    def full_group_state(self, level="macro"):
        if level not in ("fine", "macro"):
            raise ValueError("Group level must be fine or macro")
        macro = level == "macro"
        groups = self.catalog["macro_groups" if macro else "groups"]
        indices = self.full_layout["macro_group_index" if macro else "group_index"]
        size = len(groups)
        order, starts, counts = self._group_reductions[level]
        ordered_voltage = self.brain.v[order]
        minimum = np.minimum.reduceat(ordered_voltage, starts)
        maximum = np.maximum.reduceat(ordered_voltage, starts)
        return {"level": level, "coverage": self.brain.n,
                "group_ids": [group["group_id"] for group in groups], "counts": counts.tolist(),
                "spikes": np.bincount(indices, weights=self.brain.counts, minlength=size).astype(np.int64).tolist(),
                "cumulative_spikes": np.bincount(indices, weights=self.cumulative_counts, minlength=size).astype(np.int64).tolist(),
                "voltage_mean": (np.bincount(indices, weights=self.brain.v, minlength=size) / counts).tolist(),
                "voltage_min": minimum.tolist(), "voltage_max": maximum.tolist(),
                "voltage_units": "model membrane mV", "simulation_ms": self.brain.sim_ms,
                "cumulative_counts_complete": self.cumulative_counts_complete,
                "partition_sha256": self.catalog["macro_partition_sha256" if macro else "partition_sha256"]}

    def read_full_state(self, path=None, include_graph=False):
        if type(include_graph) is not bool:
            raise ValueError("include_graph must be bool")
        directory = Path(path).resolve() if path else ROOT / "artifacts/full_brain"
        if directory.suffix == ".npz":
            raise ValueError("Snapshot path is an output directory; each immutable file gets a new UUID")
        directory.mkdir(parents=True, exist_ok=True)
        name = "state-" + uuid.uuid4().hex
        temporary, destination = directory / (name + ".partial.npz"), directory / (name + ".npz")
        b = self.brain
        arrays = {"ids": b.ids, "v": b.v, "counts": b.counts, "cumulative_counts": self.cumulative_counts,
                  "drive": b.drive, **self.full_layout}
        if include_graph:
            arrays.update(ptr=b.ptr, post=b.post, weight=b.weight)
        metadata = {"schema": 1, "neurons": b.n, "edges": int(b.weight.size), "simulation_ms": b.sim_ms,
                    "duration_ms": self.drive_state["last_duration_ms"],
                    "ids_sha256": self.catalog["ids_sha256"],
                    "visual_input_enabled": self.last_visual_input_enabled,
                    "total_spikes": b.total_spikes, "cumulative_counts_complete": self.cumulative_counts_complete,
                    "group_partition_sha256": self.catalog["partition_sha256"],
                    "macro_partition_sha256": self.catalog["macro_partition_sha256"],
                    "graph_source": str(REPO / "outputs/doom/malecns_v1/graph.npz"),
                    "edge_source": str(REPO / "connectome_data/malecns_v1/normalized/edges.arrow"),
                    "graph_is_complete_retained_release": True,
                    "soma_coordinates": self.catalog["soma_coordinates"],
                    "weight_semantics": "Current signed model efficacy for every retained aggregate edge; not raw anatomical contact counts"}
        encoded_metadata = np.asarray(json.dumps(metadata, allow_nan=False))
        if sum(value.nbytes for value in arrays.values()) <= 32 * 1024 * 1024:
            # ZIP emits many small writes/seeks. Packing normal full-neuron
            # snapshots in RAM avoids expensive per-array NTFS round trips.
            # Larger optional full-edge exports retain bounded-memory streaming.
            with io.BytesIO() as packed:
                np.savez(packed, **arrays, metadata=encoded_metadata)
                with temporary.open("xb") as stream:
                    stream.write(packed.getbuffer())
        else:
            np.savez(temporary, **arrays, metadata=encoded_metadata)
        temporary.replace(destination)
        h = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        return {**metadata, "path": str(destination), "sha256": h.hexdigest(), "temporary": path is None,
                "catalog_path": str(self.catalog_path), "include_graph": include_graph,
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}}

    def neuron_details(self, neuron_ids):
        indices = self._indices_for_ids(neuron_ids)
        if len(indices) > 256:
            raise ValueError("At most 256 neuron details per JSON query; use binary read_state for the whole graph")
        b = self.brain
        rows = []
        for index in indices:
            start, end = int(b.ptr[index]), int(b.ptr[index + 1])
            sampled = slice(start, min(end, start + 32))
            rows.append({"id": str(b.ids[index]), "index": int(index), "voltage_mv": float(b.v[index]),
                         "spikes": int(b.counts[index]), "cumulative_spikes": int(self.cumulative_counts[index]),
                         "group": self.catalog["groups"][int(self.full_layout["group_index"][index])],
                         "macro_group": self.catalog["macro_groups"][int(self.full_layout["macro_group_index"][index])],
                         "soma_xyz": self.full_layout["soma_xyz"][index].tolist() if self.full_layout["soma_valid"][index] else None,
                         "outgoing_edges": end - start, "outgoing_sample_limit": 32,
                         "outgoing_sample": [{"post_id": str(b.ids[j]), "weight": float(w)} for j, w in zip(b.post[sampled], b.weight[sampled])]})
        return {"neurons": rows, "full_edges_access": "read_state(include_graph=True) returns every CSR edge"}

    def _full_io_checkpoint(self):
        return {"schema": 1, "partition_sha256": self.catalog["partition_sha256"],
                "cumulative_counts_zlib_base64": base64.b64encode(zlib.compress(self.cumulative_counts.tobytes())).decode(),
                "cumulative_counts_sha256": array_digest(self.cumulative_counts),
                "cumulative_counts_complete": self.cumulative_counts_complete,
                "last_general_stimulation": self.last_general_stimulation,
                "last_visual_input_enabled": self.last_visual_input_enabled}

    def _restore_full_io(self, value):
        if value is None:
            self.cumulative_counts.fill(0)
            self.cumulative_counts_complete = False
            self.last_general_stimulation = []
            self.last_visual_input_enabled = True
            return
        if value.get("schema") != 1 or value.get("partition_sha256") != self.catalog["partition_sha256"]:
            raise ValueError("Full-neuron checkpoint partition mismatch")
        encoded = value.get("cumulative_counts_zlib_base64")
        if not isinstance(encoded, str) or len(encoded) > 4_000_000:
            raise ValueError("Invalid cumulative neural count payload")
        inflater = zlib.decompressobj()
        raw = inflater.decompress(base64.b64decode(encoded, validate=True), self.brain.n * 8 + 1)
        if not inflater.eof or len(raw) != self.brain.n * 8:
            raise ValueError("Cumulative neural count length mismatch")
        counts = np.frombuffer(raw, dtype=np.int64).copy()
        if np.any(counts < 0) or array_digest(counts) != value.get("cumulative_counts_sha256"):
            raise ValueError("Cumulative neural counts failed integrity validation")
        complete = value.get("cumulative_counts_complete")
        if type(complete) is not bool:
            raise ValueError("Invalid count completeness flag")
        # Last-interval descriptors are provenance, not held future stimulation.
        # A portable checkpoint must not reopen an already consumed input file.
        last = validate_general_stimulation(value.get("last_general_stimulation"))
        visual = value.get("last_visual_input_enabled", True)
        if type(visual) is not bool:
            raise ValueError("Invalid saved visual_input_enabled flag")
        nonfile = []
        for item in last:
            if "neuron_currents_file" in item:
                if item["neuron_currents_file"]["ids_sha256"] != self.catalog["ids_sha256"]:
                    raise ValueError("Historical neuron current descriptor graph identity mismatch")
            else:
                nonfile.append(item)
        # Check each historical target independently: an omitted file may have
        # cancelled overlapping currents, so its absence cannot recreate the sum.
        for item in nonfile:
            self._prepare_general_stimulation([item], {name: 0.0 for name in DRIVE_LIMITS_MV})
        self.cumulative_counts[:] = counts
        self.cumulative_counts_complete = complete
        self.last_general_stimulation = last
        self.last_visual_input_enabled = visual
