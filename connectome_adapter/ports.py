"""Bounded engineering currents for identified sensory neurons only."""
import math
import json
import re
from pathlib import Path


DRIVE_LIMITS_MV = {"retina_left": 10.0, "retina_right": 10.0, "sugar": 30.0}


def validate_neuron_currents_descriptor(value):
    """Check the small wire descriptor; the worker checks the actual NPZ bytes."""
    required = {"schema", "path", "sha256", "ids_sha256", "currents_sha256", "neurons"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid neuron current file descriptor fields")
    if type(value["schema"]) is not int or value["schema"] != 1 or type(value["neurons"]) is not int or value["neurons"] != 166700:
        raise ValueError("Neuron current descriptor requires schema 1 and all 166700 neurons")
    if any(not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key])
           for key in ("sha256", "ids_sha256", "currents_sha256")):
        raise ValueError("Neuron current hashes must be lowercase SHA256 hex")
    path = value["path"]
    if not isinstance(path, str) or not 1 <= len(path) <= 4096 or "\0" in path or not Path(path).is_absolute() or Path(path).suffix != ".npz":
        raise ValueError("Neuron current path must be an absolute .npz path")
    return dict(value)


def validate_neural_drive(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or not set(value).issubset(DRIVE_LIMITS_MV):
        raise ValueError("neural_drive permits only retina_left, retina_right, sugar")
    normalized = {name: 0.0 for name in DRIVE_LIMITS_MV}
    for name, amplitude in value.items():
        if type(amplitude) not in (int, float) or not math.isfinite(amplitude) or not 0 <= amplitude <= 1:
            raise ValueError("neural_drive amplitudes must be finite numbers within 0..1, not bool")
        normalized[name] = float(amplitude)
    return normalized


def validate_general_stimulation(value=None):
    """Validate syntax/ranges; the live worker additionally verifies all targets."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("general_stimulation must contain at most 16 target specifications")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("General stimulation items must be objects")
        keys = set(item)
        if keys == {"neuron_currents_file"}:
            validate_neuron_currents_descriptor(item["neuron_currents_file"])
            continue
        if keys in ({"group_currents_mv"}, {"macro_group_currents_mv"}):
            vector = item[next(iter(keys))]
            if not isinstance(vector, list) or not 1 <= len(vector) <= 166700:
                raise ValueError("Dense group currents must be a bounded list")
            numbers = vector
        elif keys in ({"group_id", "current_mv"}, {"neuron_ids", "current_mv"}):
            numbers = [item["current_mv"]]
            if "group_id" in item:
                if not isinstance(item["group_id"], str) or not re.fullmatch(r"[gm]_[0-9a-f]{16}", item["group_id"]):
                    raise ValueError("Use an actual fine or macro group_id from group_catalog")
            else:
                ids = item["neuron_ids"]
                if not isinstance(ids, list) or not 1 <= len(ids) <= 166700:
                    raise ValueError("neuron_ids must be a bounded, nonempty list")
                if any(not ((type(value) is int and value >= 0) or (isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value))) for value in ids):
                    raise ValueError("Use original integer neuron IDs or decimal ID strings")
        else:
            raise ValueError("Invalid general stimulation fields")
        if any(type(number) not in (int, float) or not math.isfinite(number) or not -30 <= number <= 30 for number in numbers):
            raise ValueError("General currents must be finite numbers within [-30,30] mV-equivalent")
    return json.loads(json.dumps(value, allow_nan=False))
