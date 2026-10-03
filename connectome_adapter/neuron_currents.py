"""Immutable, ID-ordered binary engineering currents for every retained neuron."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import stat
import uuid
import zipfile

import numpy as np

from .ports import validate_neuron_currents_descriptor


ROOT = Path(__file__).resolve().parents[1]
CURRENT_ROOT = ROOT / "artifacts/neuron_currents"
NEURONS = 166700
MAX_FILE_BYTES = 4_000_000


def array_sha256(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate_arrays(currents, ids):
    if currents.shape != (NEURONS,) or currents.dtype != np.dtype("<f4"):
        raise ValueError("currents_mv must be float32[166700] in actual graph ID order")
    if ids.shape != (NEURONS,) or ids.dtype != np.dtype("<i8"):
        raise ValueError("ids must be int64[166700] in actual graph order")
    if not np.isfinite(currents).all() or np.any(np.abs(currents) > 30):
        raise ValueError("Each neuron current must be finite within [-30,30] model mV-equivalent")
    if np.any(ids < 0) or np.unique(ids).size != NEURONS:
        raise ValueError("Each original neuron ID must occur exactly once")


def write_neuron_currents(currents_mv, ids, *, directory=None):
    """Write one immutable NPZ; caller may delete it after its step finishes.

    Input arrays must already have the exact float32/int64 dtypes. This function
    verifies format and uniqueness; only the live worker verifies graph identity.
    The returned descriptor is the value of a `neuron_currents_file` stimulus.
    """
    currents, ids = np.asarray(currents_mv), np.asarray(ids)
    _validate_arrays(currents, ids)
    currents, ids = np.ascontiguousarray(currents), np.ascontiguousarray(ids)
    directory = Path(directory).resolve() if directory else CURRENT_ROOT.resolve()
    if not directory.is_relative_to(CURRENT_ROOT.resolve()):
        raise ValueError(f"Neuron current files must stay within {CURRENT_ROOT}")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ("currents-" + uuid.uuid4().hex + ".npz")
    temporary = destination.with_suffix(".partial")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, currents_mv=currents, ids=ids)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"schema": 1, "path": str(destination),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "ids_sha256": array_sha256(ids), "currents_sha256": array_sha256(currents),
            "neurons": NEURONS}


def load_neuron_currents(descriptor, expected_ids, expected_ids_sha256):
    """Validate and read the exact same bounded bytes whose SHA was verified."""
    descriptor = validate_neuron_currents_descriptor(descriptor)
    if descriptor["ids_sha256"] != expected_ids_sha256 or len(expected_ids) != NEURONS:
        raise ValueError("Neuron current ID/order identity differs from the actual retained graph")
    supplied = Path(descriptor["path"])
    path = supplied.resolve(strict=True)
    if not path.is_relative_to(CURRENT_ROOT.resolve()) or supplied.is_symlink():
        raise ValueError("Neuron current path is outside the allowed directory or is a symlink")
    if path.suffix != ".npz":
        raise ValueError("Neuron currents require a .npz file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_FILE_BYTES:
            raise ValueError("Neuron current source must be a bounded regular NPZ file")
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Neuron current file SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 2 or {item.filename for item in members} != {"currents_mv.npy", "ids.npy"}:
            raise ValueError("Neuron current NPZ must contain only currents_mv and ids")
        if any(item.flag_bits & 1 for item in members) or sum(item.file_size for item in members) > MAX_FILE_BYTES:
            raise ValueError("Invalid or oversized neuron current NPZ members")
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=1024) as archive:
        currents, ids = archive["currents_mv"].copy(), archive["ids"].copy()
    _validate_arrays(currents, ids)
    if array_sha256(ids) != expected_ids_sha256 or not np.array_equal(ids, expected_ids):
        raise ValueError("NPZ neuron IDs do not match the complete actual graph ordering")
    if array_sha256(currents) != descriptor["currents_sha256"]:
        raise ValueError("Neuron current float32 array SHA256 mismatch")
    return np.ascontiguousarray(currents)
