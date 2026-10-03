"""Preserve existing neural inputs; budget only the new model contribution.

All vectors use actual retained IDs and float32 accumulation in worker order.
Legacy RGB, original tonic/lamina drive and PPL are independent of this explicit
additional-current budget and are never replaced by this helper.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from connectome_adapter.neuron_currents import load_neuron_currents, write_neuron_currents
from connectome_adapter.ports import DRIVE_LIMITS_MV, validate_general_stimulation, validate_neural_drive

ROOT = Path(__file__).resolve().parents[1]
N = 166700


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _snapshot(descriptor, catalog):
    from connectome_adapter.hot_state import read_budget_snapshot
    _, raw = read_budget_snapshot(descriptor)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError("Parallel input snapshot decompressed size exceeds the limit")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ("ids", "group_index", "macro_group_index")}
    for key, dtype, identity in (("ids", np.dtype("<i8"), "ids_sha256"),
                                 ("group_index", np.dtype("<i4"), "partition_sha256"),
                                 ("macro_group_index", np.dtype("<i4"), "macro_partition_sha256")):
        if arrays[key].shape != (N,) or arrays[key].dtype != dtype or array_sha256(arrays[key]) != catalog[identity]:
            raise ValueError("Parallel input neuron ID order or annotation partition mismatch")
    if descriptor.get("ids_sha256") != catalog["ids_sha256"] or descriptor.get("neurons") != N:
        raise ValueError("Parallel input snapshot does not identify the complete retained graph")
    return arrays


def _indices(values, ids):
    requested = np.asarray([int(value) for value in values], dtype=np.int64)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(ids)) or not np.array_equal(sorted_ids[positions], requested) or np.unique(requested).size != requested.size:
        raise ValueError("Parallel stimulus must reference distinct existing original neuron IDs")
    return order[positions]


def expand_general_stimulation(specifications, arrays, catalog):
    """Pure full-node expansion; independently mirrors native addition order."""
    specifications = validate_general_stimulation(specifications)
    lookup = {g["group_id"]: (key, g) for key in ("groups", "macro_groups") for g in catalog[key]}
    current = np.zeros(N, dtype=np.float32)
    seen = set()
    for index, item in enumerate(specifications):
        if "neuron_currents_file" in item:
            vector = load_neuron_currents(item["neuron_currents_file"], arrays["ids"], catalog["ids_sha256"])
            if index == 0:
                current = vector
            else:
                current += vector
        elif "macro_group_currents_mv" in item or "group_currents_mv" in item:
            key = next(iter(item))
            macro = key == "macro_group_currents_mv"
            count = catalog["macro_group_count" if macro else "group_count"]
            values = np.asarray(item[key])
            if key in seen or values.shape != (count,):
                raise ValueError("Legacy dense group currents have duplicate partitions or wrong shape")
            seen.add(key)
            current += values.astype(np.float32)[arrays["macro_group_index" if macro else "group_index"]]
        else:
            if "group_id" in item:
                group_id = item["group_id"]
                if group_id in seen or group_id not in lookup:
                    raise ValueError("Legacy group input is duplicate or absent from the true annotation catalog")
                seen.add(group_id)
                level, group = lookup[group_id]
                indices = np.flatnonzero(arrays["macro_group_index" if level == "macro_groups" else "group_index"] == group["index"])
            else:
                indices = _indices(item["neuron_ids"], arrays["ids"])
            current[indices] += float(item["current_mv"])
    return current


def prepare_parallel_stimulation(brain, raw_state_descriptor, model_currents_descriptor,
                                 legacy_general, legacy_neural_drive):
    """Return one budgeted model descriptor plus hashes for the full input sum.

    Caller passes the exact normalized amplitudes it will actually execute,
    including any float32 action round-trip, and the exact general-stimulus order
    (including an action-derived prefix). No brain time or weights are changed.
    """
    catalog = brain.groups()
    arrays = _snapshot(raw_state_descriptor, catalog)
    existing = validate_general_stimulation(legacy_general)
    if len(existing) >= 16:
        raise ValueError("Existing inputs already use all 16 stimulus slots; cannot silently remove one to add the model")
    normalized = validate_neural_drive(legacy_neural_drive)
    old_general = expand_general_stimulation(existing, arrays, catalog)
    description = brain.describe()
    ports = description["input_ports"]
    mapped = {}
    for name, maximum in DRIVE_LIMITS_MV.items():
        port = ports[name]
        if port.get("additional_current_range_mv_equivalent") != [0.0, maximum]:
            raise ValueError("Actual legacy port amplitude mapping differs from the verified contract")
        indices = _indices(port["neuron_ids"], arrays["ids"])
        if indices.size != port["count"]:
            raise ValueError("Actual legacy input port count mismatch")
        mapped[name] = indices

    def with_ports(general):
        combined = general.copy()
        for name, amplitude in normalized.items():
            combined[mapped[name]] += amplitude * DRIVE_LIMITS_MV[name]
        return combined

    old_combined = with_ports(old_general)
    if not np.isfinite(old_combined).all() or np.any(np.abs(old_combined) > 30):
        raise ValueError("Existing general and sensory-port inputs already exceed [-30,30]; they were not changed")
    requested = load_neuron_currents(model_currents_descriptor, arrays["ids"], catalog["ids_sha256"])
    # Float32 old total is the actual existing stimulus. In particular, a model
    # zero must remain zero even where original additions rounded to the bound.
    applied = np.clip(requested.astype(np.float64), -30.0 - old_combined.astype(np.float64),
                      30.0 - old_combined.astype(np.float64)).astype(np.float32)

    def total_general():
        # First-file assignment preserves negative-zero bytes just as the worker.
        return (old_general + applied).astype(np.float32) if existing else applied.copy()

    # Re-association with original ports can move the rounded total by one ULP.
    # Repair only the new component, toward zero, without changing old inputs.
    for _ in range(8):
        combined = with_ports(total_general())
        overflow = np.abs(combined) > 30
        if not overflow.any():
            break
        excess = combined[overflow].astype(np.float64) - np.clip(combined[overflow].astype(np.float64), -30, 30)
        corrected = (applied[overflow].astype(np.float64) - excess).astype(np.float32)
        lo, hi = np.minimum(applied[overflow], 0), np.maximum(applied[overflow], 0)
        applied[overflow] = np.nextafter(np.clip(corrected, lo, hi), np.float32(0))
    combined = with_ports(total_general())
    if not np.isfinite(combined).all() or np.any(np.abs(combined) > 30):
        raise ValueError("Could not budget the new model current without altering existing inputs")
    changed = applied.view(np.uint32) != requested.view(np.uint32)
    descriptor = write_neuron_currents(applied, arrays["ids"])
    total = total_general()
    return {"schema": 1, "applied_currents_file": descriptor,
            "requested_currents_file": dict(model_currents_descriptor),
            "expected_general_currents_sha256": array_sha256(total),
            "expected_general_driven_neurons": int(np.count_nonzero(total)),
            "expected_explicit_current_sha256": array_sha256(with_ports(total)),
            "existing_general_currents_sha256": array_sha256(old_general),
            "preserved_neural_drive": normalized, "preserved_general": existing,
            "input_ports_sha256": description["input_ports_sha256"],
            "requested_vs_applied": {"changed_neurons": int(np.count_nonzero(changed)),
                "requested_sha256": array_sha256(requested), "applied_sha256": array_sha256(applied),
                "requested_min_mv": float(requested.min()), "requested_max_mv": float(requested.max()),
                "applied_min_mv": float(applied.min()), "applied_max_mv": float(applied.max()),
                "maximum_change_mv": float(np.max(np.abs(requested - applied))),
                "rule": "Only the added model contribution is clipped to the remaining per-neuron explicit-input budget"},
            "existing_inputs_modified": False, "visual_and_ppl_not_in_explicit_budget": True}
