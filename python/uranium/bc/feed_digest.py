import hashlib
import json
from collections.abc import Iterator

import numpy as np

GROUP_FIELDS = [
    "entity_cat", "entity_cont", "entity_mask", "entity_cell", "entity_pos", "spatial", "scalar", "scalar_cat", "z_build", "z_units", "z_present", "type_mask", "unit_cap",
    "loc_class_mask", "is_building", "action", "delay", "queued", "selected", "target_unit", "target_loc", "privileged", "weights", "results", "value_w", "keep", "real",
    "priv_present",
]


def group_digest(g: dict) -> str:
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
    lanes = batch.lanes
    step = max(1, group_size or lanes)
    for s in range(0, lanes, step):
        yield batch.group(s, min(s + step, lanes), t, pad, max_select)


def add_parser(sub) -> None:
    p = sub.add_parser("feed-digest", help="SHA-256 per lane group of the first BC batches (feed determinism check)")
    p.add_argument("--config", default="config/bc.toml")
    p.add_argument("--shards", action="append", required=True)
    p.add_argument("--stage", default="s1", choices=["s1", "s2"])
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--max-select", type=int, default=12)
    p.add_argument("--pack", action="store_true")
    p.add_argument("--synthetic-maps", action="store_true")
    p.set_defaults(fn=main)


def main(a) -> int:
    from uranium import _core

    cfg = json.loads(_core.bc_config(a.config, a.stage)[0])
    feeder = _core.Feeder(a.config, a.shards, stage=a.stage, prefetch=0, pack=a.pack, synthetic_maps=a.synthetic_maps, check_maps=False)
    for k in range(a.batches):
        b = feeder.next()
        if b is None:
            break
        digests = [group_digest(g) for g in groups(b, a.group, cfg["seq_len"], cfg["pad_entities"], a.max_select)]
        print(json.dumps({"batch": k, "valid": b.valid, "value_valid": b.value_valid, "groups": digests, "state": json.loads(b.state_json())}))
    return 0
