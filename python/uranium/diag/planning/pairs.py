import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from uranium.diag.planning import branch as pb
from uranium.diag.planning import pred as pp

PAIR_HOLDOUT = 0.25
BASELINE = "none"
SPLIT_NAME = "pairs"
CHUNK = 256


class PairSet:
    def __init__(self, name: str, folder, dump, main: pp.Data, root=None, lag: int = 2, split_seed: int = 1):
        self.name = name
        folder = Path(folder)
        points = pb.read_jsonl(folder / "points.jsonl")
        branches = pb.read_jsonl(folder / "branches.jsonl")
        c_sha = str(main.checkpoint["checkpoint_sha256"])
        other = sorted({str(p.get("checkpoint_sha256")) for p in points if str(p.get("checkpoint_sha256")) != c_sha})
        if other:
            raise SystemExit(f"plan-s0: --pairs {name}: trunks played by {other}, not the dump checkpoint {c_sha}; "
                             "branch training data must be games of C = F (053-R1); this also keeps B1 (played by gn, test only, 053-R8 2) out")
        B = pb.Branches(points, branches, BASELINE)
        if BASELINE not in B.alts:
            raise SystemExit(f"plan-s0: --pairs {name}: no `{BASELINE}` branches")
        self.data = pp.Data([pp.Source(name, "model", "eval", str(dump))], main.ks, keep_bot=True, design=main.design)
        if str(self.data.checkpoint["checkpoint_sha256"]) != c_sha:
            raise SystemExit(f"plan-s0: --pairs {name}: trunk dump checkpoint {self.data.checkpoint['checkpoint_sha256']} is not {c_sha} (053-R1)")
        if self.data.h.shape[1] != main.h.shape[1] or self.data.n_act != main.n_act or self.data.own_names != main.own_names:
            raise SystemExit(f"plan-s0: --pairs {name}: trunk dump rows differ in shape from the training sources")
        roots = [Path(x) for x in (root, os.environ.get("URANIUM_SHARED_DIR"), folder) if x]
        rows, missing = pb.find_rows(B, self.data, roots, lag)
        trunks = sorted({pb.trunk_of(B.all_points[p]) for p in B.points})
        self.held_trunks = pp.split_eval(trunks, SPLIT_NAME, PAIR_HOLDOUT, split_seed)
        cache: dict = {}
        held_sha = {p: pb.replay_sha(B.all_points[p], roots, cache) for p in B.points if pb.trunk_of(B.all_points[p]) in self.held_trunks}
        if unknown := sorted(p for p, sha in held_sha.items() if not sha):
            raise SystemExit(f"plan-s0: --pairs {name}: held-out trunk points {unknown[:3]} have no replay to check against the training games (--pair-root)")
        self.held_trunk_shas = set(held_sha.values())
        q = list(main.q_names)
        cols = {k: [] for k in ("row", "c", "trunk_held", "point")}
        ds, oks, skip = [], [], {"no_row": len(missing), "closed": 0, "no_label": 0, "no_action": 0, "not_one_decision_force": 0}
        forced = [a for a in B.alts if a != BASELINE and pb.one_decision_force(B.rules[a]) is not None]
        skip["not_one_decision_force"] = sum(a != BASELINE for a in B.alts) - len(forced)
        used: dict = {}
        for pi, p in enumerate(B.points):
            if p not in rows:
                continue
            r = rows[p]
            trunk = pb.trunk_of(B.all_points[p])
            base = np.array([np.nanmean(v) if np.isfinite(v := B.values(p, BASELINE, lambda b, h=h: pb.outcome(b, h))).any() else np.nan for h in q])
            mask = self.data.mask_rows(np.array([r]))[0]
            for a in forced:
                c = pb.one_decision_force(B.point_rules.get((p, a)))
                if c is None:
                    skip["no_action"] += 1
                    continue
                if not mask[c]:
                    skip["closed"] += 1
                    continue
                ya = np.array([np.nanmean(v) if np.isfinite(v := B.values(p, a, lambda b, h=h: pb.outcome(b, h))).any() else np.nan for h in q])
                d = ya - base
                if not np.isfinite(d).any():
                    skip["no_label"] += 1
                    continue
                used.setdefault(a, set()).add(int(c))
                cols["row"].append(r)
                cols["c"].append(c)
                cols["trunk_held"].append(trunk in self.held_trunks)
                cols["point"].append(pi)
                ds.append(np.nan_to_num(d))
                oks.append(np.isfinite(d))
        self.row = np.asarray(cols["row"], np.int64)
        self.c = np.asarray(cols["c"], np.int64)
        self.held = np.asarray(cols["trunk_held"], bool)
        self.point = np.asarray(cols["point"], np.int64)
        self.d = np.asarray(ds, np.float32).reshape(len(self.row), len(q))
        self.ok = np.asarray(oks, np.float32).reshape(len(self.row), len(q))
        self.points = B.points
        self.info = {"points": len(B.points), "pairs": int(len(self.row)), "held_pairs": int(self.held.sum()), "trunks": len(trunks),
                     "held_trunks": len(self.held_trunks), "skipped": skip,
                     "alts": {a: sorted(used.get(a, ())) for a in forced}}

    def held_shas(self) -> set:
        return set(self.held_trunk_shas)

    def digest(self) -> str:
        h = hashlib.sha256(json.dumps(self.data.checkpoint, sort_keys=True).encode())
        h.update(json.dumps([[e.get("sha"), e.get("start")] for e in self.data.game_meta]).encode())
        for x in (self.row, self.c, self.d, self.ok, self.held):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()

    def train_pairs(self) -> np.ndarray:
        return np.nonzero(~self.held)[0]

    def tensors(self, idx, st: dict) -> dict:
        r = self.row[idx]
        t = {"h": torch.from_numpy(self.data.h[r].astype(np.float32)), "c": torch.from_numpy(self.c[idx]),
             "pi": torch.from_numpy(self.data.probs(r).astype(np.float32)), "d": torch.from_numpy(self.d[idx]), "ok": torch.from_numpy(self.ok[idx])}
        if self.data.design.own_input:
            t["own"] = self.data.own_in(r, st)
        return t


class PairUnion:
    def __init__(self, sets: list[PairSet]):
        self.sets = sets
        self.offsets = np.cumsum([0] + [len(s.row) for s in sets])

    def train_pairs(self) -> np.ndarray:
        return np.concatenate([s.train_pairs() + o for s, o in zip(self.sets, self.offsets[:-1])]) if self.sets else np.zeros(0, np.int64)

    def tensors(self, idx, st: dict) -> dict:
        parts = []
        for s, a, b in zip(self.sets, self.offsets[:-1], self.offsets[1:]):
            m = (idx >= a) & (idx < b)
            if m.any():
                parts.append(s.tensors(idx[m] - a, st))
        return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


@torch.no_grad()
def evaluate(net, ps_: PairSet, st: dict) -> dict:
    out = {}
    score = net.score if isinstance(net, pp.Split) else net
    for part, sel in (("train", ~ps_.held), ("held", ps_.held)):
        idx = np.nonzero(sel)[0]
        if not len(idx):
            out[part] = {"pairs": 0}
            continue
        pred = []
        for a in range(0, len(idx), CHUNK):
            t = ps_.tensors(idx[a:a + CHUNK], st)
            Q = torch.sigmoid(score.q_all(t["h"], torch.arange(t["pi"].shape[1]), t.get("own")))
            pred.append((Q[torch.arange(len(t["c"])), t["c"]] - (t["pi"][:, :, None] * Q).sum(1)).numpy())
        pred = np.concatenate(pred)
        d, ok = ps_.d[idx], ps_.ok[idx] > 0
        heads = {}
        for j, h in enumerate(net.q_names):
            m = ok[:, j]
            if not m.any():
                continue
            agree, ties = pb.sign_agreement(pred[m, j], d[m, j])
            heads[h] = {"pairs": int(m.sum()), "mse": float(((pred[m, j] - d[m, j]) ** 2).mean()), "pred_mean": float(pred[m, j].mean()),
                        "measured_mean": float(d[m, j].mean()), "sign_agree": float(np.nanmean(agree)) if np.isfinite(agree).any() else None, "ties": ties}
        out[part] = {"pairs": int(len(idx)), "heads": heads}
    return out
