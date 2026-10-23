"""BC data from the Rust side (041, AMB-000-4): `uranium._core.Feeder` walks the lanes and
collates each lane group into numpy arrays (030 §1 layout, rows b-major)."""

import hashlib
from collections.abc import Iterator

import numpy as np

# Same order as `uranium_data::bc::GROUP_FIELDS` (checked against `_core.group_fields()`).
GROUP_FIELDS = [
    "entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap",
    "loc_class_mask", "is_building", "action", "delay", "queued", "selected", "target_unit", "target_loc", "privileged", "weights", "results", "value_w", "keep",
]


def group_digest(g: dict) -> str:
    """SHA-256 over a lane group's arrays in GROUP_FIELDS order: int32 / float32 little-endian,
    bool one byte. Computed here from the numpy arrays, independently of the Rust digest."""
    h = hashlib.sha256()
    for k in GROUP_FIELDS:
        a = np.ascontiguousarray(g[k])
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        elif a.dtype.kind == "i":
            a = a.astype("<i4", copy=False)
        elif a.dtype.kind == "f":
            a = a.astype("<f4", copy=False)
        h.update(a.tobytes())
    return h.hexdigest()


def groups(batch, group_size: int, t: int, pad: int, max_select: int) -> Iterator[dict]:
    """The batch's lane groups of `group_size` lanes in order (041-R3 micro-lanes)."""
    lanes = batch.lanes
    step = max(1, group_size or lanes)
    for s in range(0, lanes, step):
        yield batch.group(s, min(s + step, lanes), t, pad, max_select)
