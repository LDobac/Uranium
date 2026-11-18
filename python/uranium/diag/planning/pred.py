import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.diag.planning import decisions as dec_rows
from uranium.diag.planning import store as dst
from uranium.diag.planning.dump import load as load_dump

VARIANTS = ("U", "A", "Ad", "H", "S")
KINDS = ("human", "teacher", "model")
ROLES = ("train", "eval", "split")
ROLE_IDS = {"train": 0, "eval": 1, "val": 2}
HIDDEN, EMB_A, EMB_D = 512, 32, 16
PARTNER_SEED = 12345
OWN_SD_FLOOR, PRIV_SD_FLOOR, CONST_SD = 0.5, 0.05, 1e-6
Q_NAMES = ("win", "scv8")
NOOP = 0


@dataclass(frozen=True)
class Design:
    chains: str = "drop"
    chain_kinds: tuple = ("teacher", "model")
    own_input: bool = True
    own_target: str = "delta"
    partner: str = "other-game"
    k_unit: str = "dec"

    def __post_init__(self):
        if self.chains == "keep":
            object.__setattr__(self, "chain_kinds", ())
        object.__setattr__(self, "chain_kinds", tuple(self.chain_kinds))
        if self.chains not in ("drop", "keep") or self.own_target not in ("delta", "level") or self.partner not in ("other-game", "same-game") \
                or self.k_unit not in ("dec", "frame") or any(k not in KINDS for k in self.chain_kinds):
            raise SystemExit(f"plan-s0: bad design {self}")

    def as_dict(self) -> dict:
        return {"chains": self.chains, "chain_kinds": list(self.chain_kinds), "own_input": self.own_input, "own_target": self.own_target,
                "partner": self.partner, "k_unit": self.k_unit}


V1 = Design(chains="keep", chain_kinds=(), own_input=False, own_target="level", partner="same-game", k_unit="dec")


@dataclass
class Source:
    name: str
    kind: str
    role: str
    path: str
    rows: str = "all"


ROW_KINDS = ("all", "decision")


def parse_src(spec: str) -> Source:
    head, sep, path = spec.partition("=")
    parts = head.split(":")
    if not sep or len(parts) not in (3, 4) or parts[1] not in KINDS or parts[2] not in ROLES or not parts[0] or parts[3:4] not in ([], ["decision"]):
        raise SystemExit(f"--src {spec!r}: expected NAME:KIND:ROLE[:decision]=PATH, KIND in {KINDS}, ROLE in {ROLES}")
    return Source(parts[0], parts[1], parts[2], path, parts[3] if len(parts) == 4 else "all")


def _ms(x, floor) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float64)
    mu, sd = x.mean(0), x.std(0)
    return mu.astype(np.float32), np.where(sd < CONST_SD, np.inf, np.maximum(sd, floor)).astype(np.float32)


def split_eval(keys: list, name: str, frac: float, seed: int) -> set:
    u = sorted(set(keys))
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    p = rng.permutation(len(u))
    n_eval = int(round(frac * len(u)))
    return {u[i] for i in p[:n_eval]}


class Data:
    def __init__(self, sources: list[Source], ks=(8, 32), eval_frac: float = 0.2, split_seed: int = 1, drop: dict | None = None,
                 keep_bot: bool = False, maps=None, design: Design = Design(), mmap=None, q_names=Q_NAMES, labels: dict | None = None, val_frac: float = 0.0):
        self.sources = sources
        self.ks = list(ks)
        self.design = design
        self.store = None if mmap is None else dst.Store(mmap)
        self.q_names = tuple(q_names)
        self.val_frac = float(val_frac)
        labels = labels or {}
        self.labeled = bool(labels)
        rows, games, mans, decs = [], [], [], []
        for s in sources:
            if s.rows not in ROW_KINDS:
                raise SystemExit(f"plan-s0: source {s.name} rows {s.rows!r}, expected {ROW_KINDS}")
            r, g, m = load_dump(s.path) if self.store is None else dst.load(s.path)
            rows.append(r)
            games.append(g)
            mans.append(m)
            decs.append(dec_rows.load(s.path) if s.rows == "decision" else None)
        if design.k_unit != "dec" and any(d is not None for d in decs):
            raise SystemExit("plan-s0: decision-row sources count k in decisions (053-R5 4); --k-unit frame is not supported with :decision")
        shas = {m["checkpoint_sha256"] for m in mans}
        if len(shas) != 1:
            raise SystemExit(f"plan-s0: the sources come from different checkpoints {sorted(shas)}")
        self.manifest = mans[0]
        self.checkpoint = {"checkpoint": mans[0]["checkpoint"], "checkpoint_sha256": mans[0]["checkpoint_sha256"]}
        for m in mans[1:]:
            if m["own_columns"] != mans[0]["own_columns"]:
                raise SystemExit("plan-s0: sources with different own-state columns")
        self.own_names = mans[0]["own_columns"]
        self.op = np.zeros(rows[0]["logp"].shape[1], bool)
        self.op[mans[0]["op_actions"]] = True
        self.game_meta, gid_map = [], []
        drop = drop or {}
        maps = None if maps is None else {int(x) for x in maps}
        self.dropped = {}
        self.decision_info = {}
        for s, g, dc in zip(sources, games, decs):
            views = drop.get(s.name)
            bot = {(x[0], int(x[1])) for x in views["bot"]} if views else set()
            ours = {(x[0], int(x[1])) for x in views["model"]} if views else set()
            if s.kind == "model" and dc is None and not views and not keep_bot and any(e.get("race") == "T" for e in g.values()):
                raise SystemExit(f"plan-s0: model source {s.name} has Terran-enemy streams (the bot's view of a TvT replay is a stream too): "
                                 f"--drop {s.name}=<views json> or --keep-bot-perspective")
            keyf = lambda gid, e: e.get("sha") or f"gid{gid}"  # noqa: E731
            m, n_bot, n_map, n_dec = {}, 0, 0, {}
            for gid in sorted(g):
                e = g[gid]
                view = (e.get("sha"), int(e.get("start", -1)))
                if views and view not in bot and view not in ours:
                    raise SystemExit(f"plan-s0: source {s.name} stream {view} is in neither view list of --drop")
                st = None if dc is None else dc[1].get(gid, {}).get("status", "no_decisions")
                if view in bot or e.get("skipped"):
                    n_bot += 1
                elif st is not None and st != "ok":
                    n_dec[st] = n_dec.get(st, 0) + 1
                elif maps is not None and int(e.get("map", -1)) not in maps:
                    n_map += 1
                else:
                    m[gid] = None
            self.dropped[s.name] = {"bot_view": n_bot, "map": n_map, **({"decision_status": n_dec} if dc is not None else {})}
            ev = split_eval([keyf(k, g[k]) for k in m], s.name, eval_frac, split_seed) if s.role == "split" else set()
            va = set()
            if val_frac > 0 and s.role != "eval":
                va = split_eval([keyf(k, g[k]) for k in m if keyf(k, g[k]) not in ev], f"{s.name}/val", val_frac, split_seed)
            for gid in m:
                e = g[gid]
                key = keyf(gid, e)
                role = s.role if s.role != "split" else ("eval" if key in ev else "train")
                if role == "train" and key in va:
                    role = "val"
                m[gid] = len(self.game_meta)
                dm = {} if dc is None else {"rows": dc[1][gid]["matched"], "dump_rows": e.get("rows"), "decisions": dc[1][gid]}
                self.game_meta.append({**e, **dm, "src": s.name, "kind": s.kind, "role": role, "gid": gid, "key": f"{s.name}/{key}", "si": len(gid_map)})
            gid_map.append(m)
        G = len(self.game_meta)
        self.g_src = np.array([e["si"] for e in self.game_meta], np.int64)
        self.g_role_eval = np.array([e["role"] == "eval" for e in self.game_meta], bool)
        self.g_role_val = np.array([e["role"] == "val" for e in self.game_meta], bool)
        self.g_kind = np.array([KINDS.index(e["kind"]) for e in self.game_meta], np.int8)
        self.g_q = np.array([[game_q(e, q, labels[e["src"]].get(e["gid"], {}) if e["src"] in labels else None) for q in self.q_names]
                             for e in self.game_meta], np.float64).reshape(len(self.game_meta), len(self.q_names))
        self.g_cls = [e.get("cls_given") or e.get("cls") for e in self.game_meta]
        self.g_race = [e.get("race", "U") for e in self.game_meta]
        keys = sorted({e["key"] for e in self.game_meta})
        kid = {k: i for i, k in enumerate(keys)}
        self.g_unit = np.array([kid[e["key"]] for e in self.game_meta], np.int64)
        self.split_digest = hashlib.sha256(json.dumps(sorted([e["key"], e.get("start"), e["role"]] for e in self.game_meta)).encode()).hexdigest()
        fields = ("h", "y", "priv", "priv_ok", "own", "action", "delay", "weight", "logp", "mask", "win_open", "frame", "dec")
        sel, ovr = [], []
        for si, r in enumerate(rows):
            gm = np.full(max(games[si], default=-1) + 1, -1, np.int64)
            for gid, j in gid_map[si].items():
                gm[gid] = j
            if decs[si] is None:
                game = gm[np.asarray(r["game"])] if len(r["game"]) else np.zeros(0, np.int64)
                keep = np.nonzero(game >= 0)[0]
                sel.append((keep, game[keep]))
                ovr.append({})
                continue
            d = decs[si][0]
            dgame = d["game"].astype(np.int64)
            dg = np.where(dgame < len(gm), gm[np.minimum(dgame, len(gm) - 1)], -1) if len(gm) else np.full(len(dgame), -1)
            on = (dg >= 0) & (d["row"] >= 0)
            o = np.argsort(d["row"][on], kind="stable")
            keep = d["row"][on][o].astype(np.int64)
            sel.append((keep, dg[on][o]))
            ovr.append({"action": d["action"][on][o], "delay": d["delay"][on][o], "weight": np.ones(len(keep), np.float32), "dec": d["ord"][on][o]})
        total = sum(len(k) for k, _ in sel)
        self.game = np.concatenate([g for _, g in sel]).astype(np.int64) if sel else np.zeros(0, np.int64)
        if self.store is not None:
            self.store.bind(sources, sel, ovr)
        for f in fields:
            shape, dtype = rows[0][f].shape[1:], rows[0][f].dtype
            big = self.store is not None and self.store.large(f)
            col = self.store.cached(f, total, shape, dtype) if big else None
            if col is None:
                dest = self.store.column(f, total, shape, dtype) if big else np.empty((total, *shape), dtype)
                off = 0
                for (keep, _), r, ov in zip(sel, rows, ovr):
                    n = len(keep)
                    for c in range(0, n, 200_000):
                        dest[off + c:off + min(c + 200_000, n)] = ov[f][c:c + 200_000] if f in ov else r[f][keep[c:c + 200_000]]
                    off += n
                col = self.store.finish(f, dest) if big else dest
            setattr(self, f, col)
        self.gk_rows = []
        off = 0
        for (keep, _), r in zip(sel, rows):
            self.gk_rows.append((off, keep, {k: r[k] for k in r if k.startswith("g") and k[1:].isdigit()}))
            off += len(keep)
        self.row_src_dec = np.zeros(total, bool)
        off = 0
        for (keep, _), dc in zip(sel, decs):
            self.row_src_dec[off:off + len(keep)] = dc is not None
            off += len(keep)
        self.N = len(self.game)
        self.G = G
        self.order = np.lexsort((self.dec, self.game))
        self.pos = np.empty(self.N, np.int64)
        self.pos[self.order] = np.arange(self.N)
        gs, ds = self.game[self.order], self.dec[self.order]
        self.nxt = {}
        dec_ks = sorted(set(self.ks) | {1}) if design.k_unit == "dec" else [1]
        for k in dec_ks:
            j = np.arange(self.N) + k
            ok = j < self.N
            jj = np.where(ok, j, 0)
            ok &= (gs[jj] == gs) & (ds[jj] == ds + k)
            n = np.full(self.N, -1, np.int64)
            n[self.order[ok]] = self.order[jj[ok]]
            self.nxt[k] = n
        if design.k_unit == "dec":
            self._decision_pairs(sel, decs, dec_ks)
        self.nxt_dec1 = self.nxt[1]
        if design.k_unit == "frame":
            fs = np.rint(self.frame[self.order].astype(np.float64))
            key = gs.astype(np.float64) * 1e7 + fs
            for k in self.ks:
                j = np.searchsorted(key, key + k, side="left")
                ok = j < self.N
                jj = np.where(ok, j, 0)
                ok &= gs[jj] == gs
                n = np.full(self.N, -1, np.int64)
                n[self.order[ok]] = self.order[jj[ok]]
                self.nxt[k] = n
        self.row_eval = self.g_role_eval[self.game] if self.N else np.zeros(0, bool)
        self.row_val = self.g_role_val[self.game] if self.N else np.zeros(0, bool)
        self.row_q = self.g_q[self.game] if self.N else np.zeros((0, len(self.q_names)))
        self.row_unit = self.g_unit[self.game] if self.N else np.zeros(0, np.int64)
        self.row_kind = self.g_kind[self.game] if self.N else np.zeros(0, np.int8)
        self.row_src = self.g_src[self.game] if self.N else np.zeros(0, np.int64)
        self.valid = self.weight > 0
        self.is_op = self.op[self.action.astype(np.int64)] & self.valid
        fo, ao = np.rint(self.frame[self.order].astype(np.float64)), self.action[self.order]
        ch = np.zeros(self.N, bool)
        if self.N:
            ch[1:] = (gs[1:] == gs[:-1]) & (ds[1:] == ds[:-1] + 1) & (fo[1:] - fo[:-1] <= 1) & (ao[:-1] != NOOP)
        self.chained = np.empty(self.N, bool)
        self.chained[self.order] = ch
        drop_kinds = [KINDS.index(x) for x in design.chain_kinds] if design.chains == "drop" else []
        self.chain_drop = self.chained & np.isin(self.row_kind, drop_kinds) & ~self.row_src_dec
        self.usable = self.valid & ~self.chain_drop
        self.chain_counts = {}
        for si, s in enumerate(sources):
            m = self.row_src == si
            self.chain_counts[s.name] = {"rows": int(m.sum()), "valid": int((m & self.valid).sum()), "chained_valid": int((m & self.valid & self.chained).sum()),
                                         "chained_op": int((m & self.is_op & self.chained).sum()), "dropped": int((m & self.valid & self.chain_drop).sum()),
                                         "dropped_op": int((m & self.is_op & self.chain_drop).sum())}
        if design.partner == "same-game":
            starts = np.concatenate([[0], np.nonzero(np.diff(gs))[0] + 1, [self.N]])
            partner = np.arange(self.N, dtype=np.int64)
            rng = np.random.default_rng(PARTNER_SEED)
            valid_o = self.usable[self.order]
            for a, b in zip(starts[:-1], starts[1:]):
                cand = self.order[a:b][valid_o[a:b]]
                if len(cand):
                    partner[self.order[a:b]] = cand[rng.integers(0, len(cand), b - a)]
        else:
            partner = np.arange(self.N, dtype=np.int64)
            rng = np.random.default_rng(PARTNER_SEED)
            for si in range(len(sources)):
                rows_s = np.nonzero(self.row_src == si)[0]
                pool = rows_s[self.usable[rows_s]]
                if not len(pool):
                    continue
                draw = pool[rng.integers(0, len(pool), len(rows_s))]
                for _ in range(64):
                    bad = np.nonzero(self.game[draw] == self.game[rows_s])[0]
                    if not len(bad) or len(np.unique(self.game[pool])) < 2:
                        break
                    draw[bad] = pool[rng.integers(0, len(pool), len(bad))]
                partner[rows_s] = draw
        self.partner = partner
        self.partner_same_game = int((self.game[partner] == self.game).sum()) if self.N else 0
        self.n_act = self.logp.shape[1]
        self.n_delay = int(max(16, int(self.delay.max()) + 1 if self.N else 16))
        c_sha = str(self.checkpoint["checkpoint_sha256"])
        played = np.array([bool((e.get("decisions") or {}).get("policy_sha")) and c_sha.startswith(e["decisions"]["policy_sha"]) for e in self.game_meta], bool)
        self.prop_ok = self.row_src_dec & played[self.game] if self.N else np.zeros(0, bool)
        for si, s in enumerate(sources):
            if s.rows == "decision":
                gm = [e for e in self.game_meta if e["si"] == si]
                m = self.row_src == si
                self.decision_info[s.name] = {"games": len(gm), "rows": int(m.sum()), "noop_rows": int((m & (self.action == NOOP)).sum()),
                                              "unmatched": sum(e["decisions"]["unmatched"] for e in gm),
                                              "played_by_checkpoint_games": int(played[[e["si"] == si for e in self.game_meta]].sum()) if self.G else 0,
                                              "excluded": self.dropped[s.name].get("decision_status", {})}

    def _decision_pairs(self, sel, decs, dec_ks) -> None:
        off = 0
        for (keep, _), dc in zip(sel, decs):
            n = len(keep)
            if dc is not None:
                d = dc[0]
                pos = np.searchsorted(keep, d["row"])
                hit = (d["row"] >= 0) & (pos < n) & (keep[np.minimum(pos, max(n - 1, 0))] == d["row"]) if n else np.zeros(len(d["row"]), bool)
                didx = np.where(hit, off + pos, -1)
                for k in dec_ks:
                    nx = dec_rows.next_decision(d["game"], d["ord"], dec_rows.counted(d["action"], d["rejected"]), k)
                    tgt = np.where(nx >= 0, didx[np.maximum(nx, 0)], -1)
                    self.nxt[k][off:off + n] = -1
                    src = hit & (tgt >= 0)
                    self.nxt[k][didx[src]] = tgt[src]
            off += n

    def propensity(self, idx, action=None) -> np.ndarray:
        idx = np.asarray(idx, np.int64)
        a = (self.action[idx] if action is None else np.asarray(action)).astype(np.int64)
        p = self.probs(idx)[np.arange(len(idx)), a]
        return np.where(self.prop_ok[idx], p, np.nan)

    def pair_rows(self, k: int, role: str, kinds=None) -> np.ndarray:
        if role not in ROLE_IDS:
            raise ValueError(f"role {role!r}")
        m = self.usable & (self.nxt[k] >= 0) & (self.row_eval == (role == "eval")) & (self.row_val == (role == "val"))
        if kinds is not None:
            m &= np.isin(self.row_kind, [KINDS.index(x) for x in kinds])
        return np.nonzero(m)[0]

    def mask_rows(self, idx) -> np.ndarray:
        return np.unpackbits(self.mask[idx], axis=1, count=self.n_act).astype(bool)

    def probs(self, idx) -> np.ndarray:
        lp = self.logp[idx].astype(np.float64)
        m = self.mask_rows(idx)
        p = np.where(m, np.exp(np.where(m, lp, 0.0)), 0.0)
        s = p.sum(1, keepdims=True)
        return np.divide(p, s, out=np.zeros_like(p), where=s > 0)

    def gk(self, k: int, idx) -> np.ndarray | None:
        out = None
        for off, keep, mm in self.gk_rows:
            m = (idx >= off) & (idx < off + len(keep))
            if not m.any():
                continue
            if f"g{k}" not in mm:
                return None
            local = keep[idx[m] - off]
            o = np.argsort(local)
            vals = np.empty((len(local), mm[f"g{k}"].shape[1]), np.float32)
            vals[o] = np.asarray(mm[f"g{k}"][local[o]], np.float32)
            if out is None:
                out = np.zeros((len(idx), vals.shape[1]), np.float32)
            out[np.nonzero(m)[0]] = vals
        return out

    def stats(self, rows, k: int | None = None) -> dict:
        sub = rows if len(rows) <= 200_000 else rows[np.linspace(0, len(rows) - 1, 200_000).astype(np.int64)]

        st = {}
        st["h_mu"], st["h_sd"] = _ms(self.h[sub], 0.0)
        st["h_sd"] = np.where(np.isfinite(st["h_sd"]), st["h_sd"], 1.0).astype(np.float32)
        pk = sub[self.priv_ok[sub] > 0]
        st["p_mu"], st["p_sd"] = _ms(self.priv[pk], PRIV_SD_FLOOR) if len(pk) else (np.zeros(self.priv.shape[1], np.float32), np.full(self.priv.shape[1], np.inf, np.float32))
        st["o_mu"], st["o_sd"] = _ms(self.own[sub], OWN_SD_FLOOR)
        if k is not None:
            st.update(self.delta_stats(rows, k))
        return st

    def delta_stats(self, rows, k: int) -> dict:
        sub = rows if len(rows) <= 200_000 else rows[np.linspace(0, len(rows) - 1, 200_000).astype(np.int64)]
        d_mu, d_sd = _ms(self.own[self.nxt[k][sub]] - self.own[sub], OWN_SD_FLOOR)
        return {"d_mu": d_mu, "d_sd": d_sd}

    def own_in(self, rows, st: dict) -> torch.Tensor:
        on = np.isfinite(st["o_sd"])
        x = np.where(on, (self.own[rows] - st["o_mu"]) / np.where(on, st["o_sd"], 1.0), 0.0)
        return torch.from_numpy(x.astype(np.float32))

    def own_target(self, idx, t, st: dict) -> np.ndarray:
        if self.design.own_target == "delta":
            x, mu, sd = self.own[t] - self.own[idx], st["d_mu"], st["d_sd"]
        else:
            x, mu, sd = self.own[t], st["o_mu"], st["o_sd"]
        return np.nan_to_num((x - mu) / sd, posinf=0.0, neginf=0.0).astype(np.float32)

    def batch(self, idx, k: int, variant: str, st: dict) -> dict:
        t = self.nxt[k][idx]
        b = {"h": torch.from_numpy(self.h[idx].astype(np.float32)), "y": torch.from_numpy(self.y[t].astype(np.float32))}
        p = (self.priv[t].astype(np.float32) - st["p_mu"]) / st["p_sd"]
        b["p"] = torch.from_numpy(np.nan_to_num(p, posinf=0.0, neginf=0.0))
        b["p_ok"] = torch.from_numpy((self.priv_ok[t] > 0).astype(np.float32))
        b["o"] = torch.from_numpy(self.own_target(idx, t, st))
        q = self.row_q[idx]
        b["q"] = torch.from_numpy(np.nan_to_num(q, nan=0.0).astype(np.float32))
        b["q_ok"] = torch.from_numpy(np.isfinite(q).astype(np.float32))
        b.update(self.inputs(idx, k, variant, st))
        return b

    def inputs(self, idx, k: int, variant: str, st: dict | None = None, action=None, in_rows=None) -> dict:
        out = {}
        if self.design.own_input and st is not None:
            out["own"] = self.own_in(idx if in_rows is None else in_rows, st)
        if variant in ("A", "Ad"):
            out["a"] = torch.from_numpy((self.action[idx] if action is None else np.asarray(action)).astype(np.int64))
        if variant == "Ad":
            out["d"] = torch.from_numpy(self.delay[idx].astype(np.int64))
        if variant == "S":
            out["a"] = torch.from_numpy(self.action[self.partner[idx]].astype(np.int64))
        if variant == "H":
            if self.design.k_unit == "dec" and not self.row_src_dec.any():
                win = self.order[self.pos[idx][:, None] + np.arange(k)[None]]
                acts = self.action[win].astype(np.int64)
                w = self.valid[win].astype(np.float32)
                n_win = k
            else:
                p0 = self.pos[idx]
                span = self.pos[self.nxt[k][idx]] - p0
                n_win = int(span.max()) if len(idx) else 0
                j = np.arange(n_win)[None]
                inside = j < span[:, None]
                win = self.order[np.where(inside, p0[:, None] + j, p0[:, None])]
                acts = self.action[win].astype(np.int64)
                w = (self.valid[win] & inside).astype(np.float32)
            hist = np.zeros((len(idx), self.n_act), np.float32)
            np.add.at(hist, (np.repeat(np.arange(len(idx)), n_win), acts.reshape(-1)), w.reshape(-1))
            hist /= np.maximum(hist.sum(1, keepdims=True), 1.0)
            out["hist"] = torch.from_numpy(hist)
        return out


def game_q(e: dict, name: str, lab: dict | None) -> float:
    if name == "scv8":
        return np.nan if e.get("scv8") is None else float(e["scv8"])
    if lab is not None:
        v = lab.get(name)
        return np.nan if v is None else float(v)
    if name == "win":
        r = e.get("result", 0)
        return np.nan if r == 0 else float(r > 0)
    return np.nan


class Predictor(nn.Module):
    def __init__(self, variant: str, d_h: int, d_y: int, d_p: int, d_o: int, n_act: int, n_delay: int, own_in: int = 0, q_names=Q_NAMES):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant {variant!r}")
        self.variant = variant
        self.own_in = own_in
        self.q_names = tuple(q_names)
        self.register_buffer("h_mu", torch.zeros(d_h))
        self.register_buffer("h_sd", torch.ones(d_h))
        self.emb_a = nn.Embedding(n_act, EMB_A) if variant in ("A", "Ad", "S") else None
        self.emb_d = nn.Embedding(n_delay, EMB_D) if variant == "Ad" else None
        d_in = d_h + own_in + (EMB_A if self.emb_a is not None else 0) + (EMB_D if self.emb_d is not None else 0) + (n_act if variant == "H" else 0)
        self.l1 = nn.Linear(d_in, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.y = nn.Linear(HIDDEN, d_y)
        self.p = nn.Linear(HIDDEN, d_p)
        self.o = nn.Linear(HIDDEN, d_o)
        self.q = nn.Linear(HIDDEN, len(self.q_names))

    def _state_in(self, h, own):
        x = [(h - self.h_mu) / self.h_sd]
        if self.own_in:
            if own is None:
                raise ValueError("this predictor reads o_t (own_in)")
            x.append(own)
        return x

    def body(self, h, a=None, d=None, hist=None, own=None):
        x = self._state_in(h, own)
        if self.emb_a is not None:
            x.append(self.emb_a(a))
        if self.emb_d is not None:
            x.append(self.emb_d(d))
        if self.variant == "H":
            x.append(hist)
        return F.relu(self.l2(F.relu(self.l1(torch.cat(x, 1)))))

    def forward(self, h, a=None, d=None, hist=None, own=None) -> dict:
        z = self.body(h, a, d, hist, own)
        return {"y": self.y(z), "p": self.p(z), "o": self.o(z), "q": self.q(z)}

    def q_all(self, h, cand, own=None) -> torch.Tensor:
        assert self.variant == "A"
        d_s = self.h_mu.shape[0] + self.own_in
        w = self.l1.weight
        zh = F.linear(torch.cat(self._state_in(h, own), 1), w[:, :d_s], self.l1.bias)
        za = F.linear(self.emb_a(cand), w[:, d_s:])
        z = F.relu(zh[:, None] + za[None])
        return self.q(F.relu(self.l2(z)))


def loss_terms(out: dict, b: dict, o_on: torch.Tensor, p_on: torch.Tensor) -> dict:
    y = 1.0 - F.cosine_similarity(out["y"], b["y"], dim=1)
    p = ((out["p"] - b["p"]) ** 2 * p_on).sum(1) / p_on.sum().clamp_min(1)
    o = ((out["o"] - b["o"]) ** 2 * o_on).sum(1) / o_on.sum().clamp_min(1)
    q = F.binary_cross_entropy_with_logits(out["q"], b["q"], reduction="none") * b["q_ok"]
    return {"y": y, "p": p * b["p_ok"], "o": o, "q": q}


WORLD_HEADS, SCORE_HEADS = ("y", "p", "o"), ("q",)


class Split(nn.Module):
    def __init__(self, world: Predictor, score: Predictor):
        super().__init__()
        self.world = world
        self.score = score
        self.variant = world.variant
        self.own_in = world.own_in
        self.q_names = score.q_names

    def forward(self, h, a=None, d=None, hist=None, own=None) -> dict:
        out = self.world(h, a, d, hist, own)
        out["q"] = self.score(h, a, d, hist, own)["q"]
        return out

    def q_all(self, h, cand, own=None) -> torch.Tensor:
        return self.score.q_all(h, cand, own)


def new_net(data: Data, variant: str, st: dict) -> Predictor:
    net = Predictor(variant, data.h.shape[1], data.y.shape[1], data.priv.shape[1], data.own.shape[1], data.n_act, data.n_delay,
                    own_in=data.own.shape[1] if data.design.own_input else 0, q_names=data.q_names)
    net.h_mu.copy_(torch.from_numpy(st["h_mu"]))
    net.h_sd.copy_(torch.from_numpy(st["h_sd"]))
    return net


def parse_mix(spec: str) -> dict | None:
    if not spec or spec == "off":
        return None
    out = {}
    for part in spec.split(","):
        k, sep, v = part.partition("=")
        if not sep or k not in KINDS or k in out:
            raise SystemExit(f"--mix {spec!r}: expected KIND=SHARE[,...] with KIND in {KINDS}, each once")
        out[k] = float(v)
    if any(v < 0 for v in out.values()) or sum(out.values()) > 1 + 1e-9:
        raise SystemExit(f"--mix {spec!r}: shares must be >= 0 and sum to 1 or less")
    return out


def mix_weights(data: Data, rows: np.ndarray, shares: dict, op_x: float = 5.0, rare_share: float = 0.001, rare_x: float = 20.0) -> tuple[np.ndarray, dict]:
    kind = data.row_kind[rows].astype(np.int64)
    act = data.action[rows].astype(np.int64)
    op = data.is_op[rows]
    n = len(rows)
    a_rows = np.bincount(act[op], minlength=data.n_act)
    a_share = a_rows / max(n, 1)
    mult_a = np.where(a_share < rare_share, np.maximum(op_x, np.minimum(rare_x, rare_share / np.maximum(a_share, 1e-300))), op_x)
    m = np.where(op, mult_a[act], 1.0)
    n_k = np.bincount(kind, minlength=len(KINDS))
    tgt = np.zeros(len(KINDS))
    given = {KINDS.index(k): v for k, v in shares.items()}
    rest = [i for i in range(len(KINDS)) if i not in given]
    for i, v in given.items():
        tgt[i] = v
    n_rest = sum(int(n_k[i]) for i in rest)
    for i in rest:
        tgt[i] = (1.0 - sum(given.values())) * n_k[i] / n_rest if n_rest else 0.0
    tgt[n_k == 0] = 0.0
    if tgt.sum() <= 0:
        raise SystemExit(f"plan-s0: --mix {shares} gives no weight to the kinds of these rows ({dict(zip(KINDS, n_k.tolist()))})")
    tgt = tgt / tgt.sum()
    w = np.zeros(n)
    for i in range(len(KINDS)):
        sel = kind == i
        if sel.any():
            w[sel] = tgt[i] * m[sel] / m[sel].sum()
    plan = {"shares": dict(shares), "op_x": op_x, "rare_share": rare_share, "rare_x": rare_x,
            "kinds": {KINDS[i]: {"rows": int(n_k[i]), "target": float(tgt[i])} for i in range(len(KINDS)) if n_k[i]},
            "op_actions": {int(x): {"rows": int(a_rows[x]), "row_share": float(a_share[x]), "multiplier": float(mult_a[x]),
                                    "expected_share": float(w[op & (act == x)].sum())} for x in np.nonzero(a_rows)[0]}}
    return w, plan


def loss_parts(lt: dict, b: dict) -> dict:
    return {"y": lt["y"].mean(), "p": lt["p"].sum() / b["p_ok"].sum().clamp_min(1),
            "o": lt["o"].mean(), "q": lt["q"].sum(0).div(b["q_ok"].sum(0).clamp_min(1)).sum()}


@torch.no_grad()
def val_loss(net, data: Data, rows: np.ndarray, k: int, variant: str, st: dict, o_on, p_on, chunk: int = 4096) -> dict:
    nq = len(net.q_names)
    y = p = p_n = o = 0.0
    q, q_n = np.zeros(nq), np.zeros(nq)
    for s in range(0, len(rows), chunk):
        b = data.batch(rows[s:s + chunk], k, variant, st)
        out = net(b["h"], b.get("a"), b.get("d"), b.get("hist"), b.get("own"))
        lt = loss_terms(out, b, o_on, p_on)
        y += float(lt["y"].double().sum())
        p += float(lt["p"].double().sum())
        p_n += float(b["p_ok"].double().sum())
        o += float(lt["o"].double().sum())
        q += lt["q"].double().sum(0).numpy()
        q_n += b["q_ok"].double().sum(0).numpy()
    n = max(len(rows), 1)
    heads = {nm: (float(q[i] / q_n[i]) if q_n[i] else None) for i, nm in enumerate(net.q_names)}
    return {"y": y / n, "p": p / max(p_n, 1.0), "o": o / n, "q": heads, "world": y / n + p / max(p_n, 1.0) + o / n,
            "score": float(sum(v for v in heads.values() if v is not None))}


def train_one(data: Data, variant: str, k: int, rows: np.ndarray, st: dict, *, steps: int, batch: int, lr: float, seed: int, log_every: int = 500,
              weights: np.ndarray | None = None, val_rows: np.ndarray | None = None, val_every: int = 500, patience: int = 4,
              split_nets: bool = False, score_wd: float = 0.0, info: dict | None = None) -> tuple[Predictor, list]:
    torch.manual_seed(seed)
    if split_nets:
        net = Split(new_net(data, variant, st), new_net(data, variant, st))
        opts = {"world": torch.optim.Adam(net.world.parameters(), lr=lr), "score": torch.optim.AdamW(net.score.parameters(), lr=lr, weight_decay=score_wd)}
        groups = {"world": (net.world, WORLD_HEADS), "score": (net.score, SCORE_HEADS)}
    else:
        net = new_net(data, variant, st)
        opts = {"all": torch.optim.Adam(net.parameters(), lr=lr)}
        groups = {"all": (net, WORLD_HEADS + SCORE_HEADS)}
    o_on = torch.from_numpy(np.isfinite(st["d_sd" if data.design.own_target == "delta" else "o_sd"]).astype(np.float32))
    p_on = torch.from_numpy(np.isfinite(st["p_sd"]).astype(np.float32))
    rng = np.random.default_rng(seed)
    cum = None if weights is None else np.cumsum(np.asarray(weights, np.float64))
    val = None if val_rows is None else {g: {"best": float("inf"), "step": 0, "state": None, "bad": 0, "stopped": False} for g in groups}
    curve = []
    drawn_kind, drawn_act = np.zeros(len(KINDS), np.int64), np.zeros(data.n_act, np.int64)
    log = []
    acc = {}
    step = 0
    for step in range(1, steps + 1):
        if cum is None:
            idx = np.sort(rows[rng.integers(0, len(rows), batch)])
        else:
            idx = np.sort(rows[np.minimum(np.searchsorted(cum, rng.random(batch) * cum[-1], side="right"), len(rows) - 1)])
        if info is not None:
            drawn_kind += np.bincount(data.row_kind[idx], minlength=len(KINDS))
            drawn_act += np.bincount(data.action[idx][data.is_op[idx]].astype(np.int64), minlength=data.n_act)
        b = data.batch(idx, k, variant, st)
        out = net(b["h"], b.get("a"), b.get("d"), b.get("hist"), b.get("own"))
        lt = loss_terms(out, b, o_on, p_on)
        parts = loss_parts(lt, b)
        active = [g for g in groups if val is None or not val[g]["stopped"]]
        loss = sum(parts[h] for g in active for h in groups[g][1])
        for g in active:
            opts[g].zero_grad(set_to_none=True)
        loss.backward()
        for g in active:
            opts[g].step()
        for kk, v in parts.items():
            acc[kk] = acc.get(kk, 0.0) + float(v.detach())
        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            log.append({"step": step, **{kk: v / n for kk, v in acc.items()}})
            acc = {}
        if val is not None and (step % val_every == 0 or step == steps):
            v = val_loss(net, data, val_rows, k, variant, st, o_on, p_on)
            curve.append({"step": step, **v})
            for g in active:
                cur = v["world"] + v["score"] if g == "all" else v[g]
                if cur < val[g]["best"]:
                    val[g].update(best=cur, step=step, bad=0, state={kk: t.detach().clone() for kk, t in groups[g][0].state_dict().items()})
                else:
                    val[g]["bad"] += 1
                    val[g]["stopped"] = val[g]["bad"] >= patience
            if all(e["stopped"] for e in val.values()):
                break
    if acc:
        log.append({"step": step, **{kk: v / (step % log_every) for kk, v in acc.items()}})
    if val is not None:
        for g, e in val.items():
            if e["state"] is not None:
                groups[g][0].load_state_dict(e["state"])
    net.eval()
    if info is not None:
        total = max(int(drawn_kind.sum()), 1)
        info["drawn"] = {"rows": int(drawn_kind.sum()), "kinds": {KINDS[i]: float(drawn_kind[i] / total) for i in range(len(KINDS)) if drawn_kind[i]},
                         "op_actions": {int(x): float(drawn_act[x] / total) for x in np.nonzero(drawn_act)[0]}}
        if val is not None:
            info["val"] = {"rows": int(len(val_rows)), "every": val_every, "patience": patience, "last_step": step, "curve": curve,
                           "best": {g: {"step": e["step"], "loss": e["best"], "stopped": e["stopped"]} for g, e in val.items()}}
    return net, log


@torch.no_grad()
def predict(net: Predictor, data: Data, idx: np.ndarray, k: int, st: dict, chunk: int = 4096, action=None, inputs_h=None, keep=("y", "p", "o", "q")) -> dict:
    out = {k_: [] for k_ in keep}
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        hin = sl if inputs_h is None else inputs_h[s:s + chunk]
        x = data.inputs(sl, k, net.variant, st, None if action is None else action[s:s + chunk], in_rows=hin)
        o = net(torch.from_numpy(data.h[hin].astype(np.float32)), x.get("a"), x.get("d"), x.get("hist"), x.get("own"))
        for k_ in out:
            out[k_].append(o[k_].numpy())
    return {k_: (np.concatenate(v) if v else np.zeros((0, 1), np.float32)) for k_, v in out.items()}


def own_spaces(data: Data, pred_o: np.ndarray, rows, st: dict) -> tuple[np.ndarray, np.ndarray]:
    o_on, d_on = np.isfinite(st["o_sd"]), np.isfinite(st["d_sd"])
    o_sd, d_sd = np.where(o_on, st["o_sd"], 1.0), np.where(d_on, st["d_sd"], 1.0)
    own_t = data.own[rows]
    if data.design.own_target == "delta":
        dlt = pred_o
        lvl = (own_t + np.where(d_on, pred_o * d_sd + st["d_mu"], st["d_mu"]) - st["o_mu"]) / o_sd
    else:
        lvl = pred_o
        dlt = (np.where(o_on, pred_o * o_sd + st["o_mu"], st["o_mu"]) - own_t - st["d_mu"]) / d_sd
    return np.where(o_on, lvl, 0.0).astype(np.float32), np.where(d_on, dlt, 0.0).astype(np.float32)


def save(net, st: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {"state": net.state_dict(), "variant": net.variant, "own_in": net.own_in, "stats": {k: v.tolist() for k, v in st.items()}, "meta": meta}
    if tuple(net.q_names) != Q_NAMES:
        ck["q_names"] = list(net.q_names)
    if isinstance(net, Split):
        ck["split"] = True
    torch.save(ck, path)


def load(path: Path, data: Data) -> tuple[Predictor, dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    q = tuple(ck.get("q_names", Q_NAMES))

    def mk():
        return Predictor(ck["variant"], data.h.shape[1], data.y.shape[1], data.priv.shape[1], data.own.shape[1], data.n_act, data.n_delay,
                         own_in=ck.get("own_in", 0), q_names=q)

    net = Split(mk(), mk()) if ck.get("split") else mk()
    net.load_state_dict(ck["state"])
    net.eval()
    st = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
    return net, st, ck["meta"]


def describe(data: Data) -> dict:
    out = {"rows": int(data.N), "games": int(data.G), "design": data.design.as_dict(), "partner_same_game_rows": data.partner_same_game, "sources": []}
    for s in data.sources:
        gm = [e for e in data.game_meta if e["src"] == s.name]
        out["sources"].append({"name": s.name, "kind": s.kind, "role": s.role, "games": len(gm), "eval_games": sum(e["role"] == "eval" for e in gm),
                               "rows": int(sum(e.get("rows", 0) for e in gm)), "chains": data.chain_counts[s.name],
                               **({"decision_rows": data.decision_info[s.name]} if s.name in data.decision_info else {})})
        if data.val_frac > 0:
            out["sources"][-1]["val_games"] = sum(e["role"] == "val" for e in gm)
    if data.q_names != Q_NAMES:
        out["q_heads"] = list(data.q_names)
    return json.loads(json.dumps(out))
