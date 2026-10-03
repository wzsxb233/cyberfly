"""Complete, annotation-derived neural partitions and binary snapshot metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor/doomfly"


def array_digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_catalog(ids, superclass, annotations):
    def label(value):
        return "unclassified" if value is None or str(value).strip() in ("", "nan", "None", "<NA>") else str(value).strip()

    fine_keys = [(label(sc), label(cell), label(soma), label(root)) for sc, cell, soma, root in zip(
        superclass, annotations.type, annotations.somaSide, annotations.rootSide)]
    macro_keys = [(key[0], key[3]) for key in fine_keys]

    def partition(keys, fields, prefix):
        unique = sorted(set(keys))
        lookup = {key: index for index, key in enumerate(unique)}
        indices = np.fromiter((lookup[key] for key in keys), dtype=np.int32, count=len(keys))
        counts = np.bincount(indices, minlength=len(unique))
        groups = [{"group_id": prefix + hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16],
                   "index": index, "count": int(counts[index]), **dict(zip(fields, key))}
                  for index, key in enumerate(unique)]
        if len({group["group_id"] for group in groups}) != len(groups):
            raise RuntimeError("Group ID collision")
        return groups, indices

    groups, group_index = partition(fine_keys, ("superclass", "cell_type", "soma_side", "root_side"), "g_")
    macro_groups, macro_index = partition(macro_keys, ("superclass", "root_side"), "m_")
    soma_xyz = np.full((len(ids), 3), np.nan, dtype=np.float32)
    soma_valid = np.zeros(len(ids), dtype=np.bool_)
    for index, location in enumerate(annotations.somaLocation):
        if location is not None:
            xyz = np.asarray(location)
            if xyz.shape == (3,) and np.isfinite(xyz).all():
                soma_xyz[index] = xyz
                soma_valid[index] = True
    catalog = {"schema": 1, "groups": groups, "group_count": len(groups), "coverage": len(ids),
               "partition_sha256": array_digest(group_index), "ids_sha256": array_digest(ids),
               "macro_groups": macro_groups, "macro_group_count": len(macro_groups),
               "macro_partition_sha256": array_digest(macro_index),
               "partition_fields": ["superclass", "type", "somaSide", "rootSide"],
               "macro_partition_fields": ["superclass", "rootSide"],
               "annotation_source": "MaleCNS v1 annotations matched by bodyId to every retained graph ID",
               "unclassified_policy": "All absent annotations retained as unclassified; every neuron belongs to exactly one fine and one macro group",
               "soma_coordinates": {"field": "somaLocation", "valid_count": int(soma_valid.sum()),
                                    "missing_count": int((~soma_valid).sum()), "units": "raw annotation coordinate units; physical calibration not asserted",
                                    "missing_values": "NaN; any replacement network layout must be labeled separately"}}
    return catalog, {"group_index": group_index, "macro_group_index": macro_index,
                     "soma_xyz": soma_xyz, "soma_valid": soma_valid}


def write_catalog(catalog, path=None):
    path = Path(path) if path else ROOT / "artifacts/full_brain/catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def main():
    import pyarrow.feather as feather
    with np.load(REPO / "outputs/doom/malecns_v1/graph.npz", allow_pickle=False) as graph:
        ids, superclass = graph["ids"], graph["superclass"]
    annotations = feather.read_table(REPO / "connectome_data/malecns_v1/annotations.feather").to_pandas().set_index("bodyId").loc[ids]
    catalog, arrays = build_catalog(ids, superclass, annotations)
    path = write_catalog(catalog)
    print(json.dumps({"path": str(path), "coverage": catalog["coverage"], "group_count": catalog["group_count"],
                      "macro_group_count": catalog["macro_group_count"], "soma_coordinates": catalog["soma_coordinates"]}), flush=True)


if __name__ == "__main__":
    main()
