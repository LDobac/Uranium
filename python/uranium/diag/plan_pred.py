"""Data and predictors of the S0 feasibility test (jepa_planning S0, reports/jepa_planning.md 6.2-6.4;
050-R9 diagnostics). CPU, deterministic for a seed and thread count.

Data (`Data`): one or more `diag plan-dump` folders, each a source `NAME:KIND:ROLE=PATH`:
- kind `human` / `teacher` / `model` (M3b reads teacher and model games, M3c trains on human-only and
  model-only rows),
- role `train` (every game trains), `eval` (every game evaluates) or `split` (a fixed-seed game-level
  split, `eval_frac` evaluates; both streams of one replay on the same side, as 041-R12). The split
  draws from a generator seeded by (`split_seed`, crc32 of the source NAME): renaming a source moves it.
Pairs (t, t+k): two rows of one game k decisions apart (`dec`); a row is used when its label is valid
(weight > 0) and its t+k row was dumped. Every predictor of one k sees the same rows.

Predictors (6.4, one per variant and k; 3.2 shape): input -> Linear 512 -> ReLU -> Linear 512 -> ReLU,
heads ŷ (d_y, 1 - cos to LN y_{t+k}), p̂ (privileged at t+k, standardised, MSE, rows with one), ô (own
state at t+k, standardised, MSE), q̂ (2 logits: game won, 8:00 SCVs >= 8; BCE where defined). Inputs:
  U   h'_t
  A   h'_t ⊕ Emb(a_t) (32)
  Ad  h'_t ⊕ Emb(a_t) ⊕ Emb(delay bucket) (16)              (A+d)
  H   h'_t ⊕ action-type shares over decisions [t, t+k) (101) (oracle)
  S   h'_t ⊕ Emb(action of a fixed random valid-label row of the same game) (control, `PARTNER_SEED`)
h' is standardised with the training rows' mean / sd (part of the predictor). Same steps, batch and seed
for every predictor."""

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.diag.plan_dump import load as load_dump

VARIANTS = ("U", "A", "Ad", "H", "S")
KINDS = ("human", "teacher", "model")
ROLES = ("train", "eval", "split")
HIDDEN, EMB_A, EMB_D = 512, 32, 16
PARTNER_SEED = 12345
# Standardisation floors (sd used = max(sd, floor); a column constant on the training rows, sd < 1e-6,
# is left out): own state half a count (supply columns are BWAPI half-supply units, the same floor),
# privileged 0.05 in its encoded units (a count is log1p(n) / 4: 0 -> 1 is 0.17).
OWN_SD_FLOOR, PRIV_SD_FLOOR, CONST_SD = 0.5, 0.05, 1e-6
Q_NAMES = ("win", "scv8")


@dataclass
class Source:
    name: str
    kind: str
    role: str
    path: str


def parse_src(spec: str) -> Source:
    """`NAME:KIND:ROLE=PATH`."""
    head, sep, path = spec.partition("=")
    parts = head.split(":")
    if not sep or len(parts) != 3 or parts[1] not in KINDS or parts[2] not in ROLES or not parts[0]:
        raise SystemExit(f"--src {spec!r}: expected NAME:KIND:ROLE=PATH, KIND in {KINDS}, ROLE in {ROLES}")
    return Source(parts[0], parts[1], parts[2], path)


def split_eval(keys: list, name: str, frac: float, seed: int) -> set:
    """The eval share of a `split` source: a fixed-seed permutation of the sorted replay keys."""
    u = sorted(set(keys))
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    p = rng.permutation(len(u))
    n_eval = int(round(frac * len(u)))
    return {u[i] for i in p[:n_eval]}


class Data:
    """Every row of the sources in memory except the g_k outputs (memmaps): h, y, priv, logp float16."""

    def __init__(self, sources: list[Source], ks=(8, 32), eval_frac: float = 0.2, split_seed: int = 1, drop: dict | None = None,
                 keep_bot: bool = False, maps=None):
        """`drop` {source name: {"bot": [[sha256, start], ...], "model": [...]}}: the views of a source
        (e.g. config/plan_s0/jepa_diag_tp_views.json); a stream (replay sha256, own start slot
        `scalar_cat[2]`) in "bot" is left out, and every stream must be in one of the lists. A `model`
        source with Terran-enemy streams (TvT: both players' views are streams) needs one, or
        `keep_bot`. `maps`: only games of these map families (0 / 1 = the target maps)."""
        self.sources = sources
        self.ks = list(ks)
        rows, games, mans = [], [], []
        for s in sources:
            r, g, m = load_dump(s.path)
            rows.append(r)
            games.append(g)
            mans.append(m)
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
        # global games
        self.game_meta, gid_map = [], []
        drop = drop or {}
        maps = None if maps is None else {int(x) for x in maps}
        self.dropped = {}
        for s, g in zip(sources, games):
            views = drop.get(s.name)
            bot = {(x[0], int(x[1])) for x in views["bot"]} if views else set()
            ours = {(x[0], int(x[1])) for x in views["model"]} if views else set()
            if s.kind == "model" and not views and not keep_bot and any(e.get("race") == "T" for e in g.values()):
                raise SystemExit(f"plan-s0: model source {s.name} has Terran-enemy streams (the bot's view of a TvT replay is a stream too): "
                                 f"--drop {s.name}=<views json> or --keep-bot-perspective")
            keyf = lambda gid, e: e.get("sha") or f"gid{gid}"  # noqa: E731
            m, n_bot, n_map = {}, 0, 0
            for gid in sorted(g):
                e = g[gid]
                view = (e.get("sha"), int(e.get("start", -1)))
                if views and view not in bot and view not in ours:
                    raise SystemExit(f"plan-s0: source {s.name} stream {view} is in neither view list of --drop")
                if view in bot:
                    n_bot += 1
                elif maps is not None and int(e.get("map", -1)) not in maps:
                    n_map += 1
                else:
                    m[gid] = None
            self.dropped[s.name] = {"bot_view": n_bot, "map": n_map}
            ev = split_eval([keyf(k, g[k]) for k in m], s.name, eval_frac, split_seed) if s.role == "split" else set()
            for gid in m:
                e = g[gid]
                key = keyf(gid, e)
                role = s.role if s.role != "split" else ("eval" if key in ev else "train")
                m[gid] = len(self.game_meta)
                self.game_meta.append({**e, "src": s.name, "kind": s.kind, "role": role, "gid": gid, "key": f"{s.name}/{key}"})
            gid_map.append(m)
        G = len(self.game_meta)
        self.g_role_eval = np.array([e["role"] == "eval" for e in self.game_meta], bool)
        self.g_kind = np.array([KINDS.index(e["kind"]) for e in self.game_meta], np.int8)
        res = np.array([e.get("result", 0) for e in self.game_meta], np.float64)
        scv = np.array([np.nan if e.get("scv8") is None else e["scv8"] for e in self.game_meta], np.float64)
        self.g_q = np.stack([np.where(res != 0, (res > 0).astype(np.float64), np.nan), scv], 1)  # [G, 2], NaN = undefined
        self.g_cls = [e.get("cls_given") or e.get("cls") for e in self.game_meta]
        self.g_race = [e.get("race", "U") for e in self.game_meta]
        keys = sorted({e["key"] for e in self.game_meta})
        kid = {k: i for i, k in enumerate(keys)}
        self.g_unit = np.array([kid[e["key"]] for e in self.game_meta], np.int64)  # bootstrap unit: the replay
        # the games and their roles (the report refuses models trained on another split)
        self.split_digest = hashlib.sha256(json.dumps(sorted([e["key"], e.get("start"), e["role"]] for e in self.game_meta)).encode()).hexdigest()
        # rows: preallocated, filled source by source from the memmaps (no second copy)
        fields = ("h", "y", "priv", "priv_ok", "own", "action", "delay", "weight", "logp", "mask", "win_open", "frame", "dec")
        sel = []
        for si, r in enumerate(rows):
            gm = np.full(max(games[si], default=-1) + 1, -1, np.int64)
            for gid, j in gid_map[si].items():
                gm[gid] = j
            game = gm[np.asarray(r["game"])] if len(r["game"]) else np.zeros(0, np.int64)
            keep = np.nonzero(game >= 0)[0]
            sel.append((keep, game[keep]))
        total = sum(len(k) for k, _ in sel)
        for f in fields:
            setattr(self, f, np.empty((total, *rows[0][f].shape[1:]), rows[0][f].dtype))
        self.game = np.empty(total, np.int64)
        self.gk_rows = []  # per source: (global row offset, local row index array, memmaps)
        off = 0
        for (keep, game), r in zip(sel, rows):
            n = len(keep)
            self.game[off:off + n] = game
            for f in fields:
                for c in range(0, n, 200_000):  # bounded temporaries
                    getattr(self, f)[off + c:off + min(c + 200_000, n)] = r[f][keep[c:c + 200_000]]
            self.gk_rows.append((off, keep, {k: r[k] for k in r if k.startswith("g") and k[1:].isdigit()}))
            off += n
        self.N = len(self.game)
        self.G = G
        # game-ordered rows: pos, and the t+k row of the same game
        self.order = np.lexsort((self.dec, self.game))
        self.pos = np.empty(self.N, np.int64)
        self.pos[self.order] = np.arange(self.N)
        gs, ds = self.game[self.order], self.dec[self.order]
        self.nxt = {}
        for k in sorted(set(self.ks) | {1}):
            j = np.arange(self.N) + k
            ok = j < self.N
            jj = np.where(ok, j, 0)
            ok &= (gs[jj] == gs) & (ds[jj] == ds + k)
            n = np.full(self.N, -1, np.int64)
            n[self.order[ok]] = self.order[jj[ok]]
            self.nxt[k] = n
        # the S control's partner: a random valid-label row of the same game. One generator (seed
        # PARTNER_SEED) walks the games in global order (sources in --src order, then game id), so the
        # draws depend on the source list and the games kept.
        starts = np.concatenate([[0], np.nonzero(np.diff(gs))[0] + 1, [self.N]])
        partner = np.arange(self.N, dtype=np.int64)
        rng = np.random.default_rng(PARTNER_SEED)
        valid_o = self.weight[self.order] > 0
        for a, b in zip(starts[:-1], starts[1:]):
            cand = self.order[a:b][valid_o[a:b]]
            if len(cand):
                partner[self.order[a:b]] = cand[rng.integers(0, len(cand), b - a)]
        self.partner = partner
        self.row_eval = self.g_role_eval[self.game] if self.N else np.zeros(0, bool)
        self.row_q = self.g_q[self.game] if self.N else np.zeros((0, 2))
        self.row_unit = self.g_unit[self.game] if self.N else np.zeros(0, np.int64)
        self.row_kind = self.g_kind[self.game] if self.N else np.zeros(0, np.int8)
        self.valid = self.weight > 0
        self.is_op = self.op[self.action.astype(np.int64)] & self.valid
        self.n_act = self.logp.shape[1]
        self.n_delay = int(max(16, int(self.delay.max()) + 1 if self.N else 16))

    # ------------------------------------------------------------- selections
    def pair_rows(self, k: int, role: str, kinds=None) -> np.ndarray:
        """Rows with a valid label and a t+k row, of `role` games (`kinds`: only those source kinds)."""
        m = self.valid & (self.nxt[k] >= 0) & (self.row_eval == (role == "eval"))
        if kinds is not None:
            m &= np.isin(self.row_kind, [KINDS.index(x) for x in kinds])
        return np.nonzero(m)[0]

    def mask_rows(self, idx) -> np.ndarray:
        """[rows, A] bool type_mask."""
        return np.unpackbits(self.mask[idx], axis=1, count=self.n_act).astype(bool)

    def probs(self, idx) -> np.ndarray:
        """[rows, A] policy probabilities renormalised over the open types (float16 log-probs)."""
        lp = self.logp[idx].astype(np.float64)
        m = self.mask_rows(idx)
        p = np.where(m, np.exp(np.where(m, lp, 0.0)), 0.0)
        s = p.sum(1, keepdims=True)
        return np.divide(p, s, out=np.zeros_like(p), where=s > 0)

    def gk(self, k: int, idx) -> np.ndarray | None:
        """[rows, d_y] float32 g_k(h'_t) of the global rows `idx`, None if a source lacks it."""
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

    # ------------------------------------------------------------- standardisation
    def stats(self, rows) -> dict:
        """Mean / sd of h', privileged and own state over `rows` (targets: their t+k rows are the
        same population). Targets: sd floored (`OWN_SD_FLOOR`, `PRIV_SD_FLOOR`); a column constant on
        these rows gets sd inf (left out of losses and R²). h': a constant column is divided by 1."""
        sub = rows if len(rows) <= 200_000 else rows[np.linspace(0, len(rows) - 1, 200_000).astype(np.int64)]

        def ms(x, floor):
            x = np.asarray(x, np.float64)
            mu, sd = x.mean(0), x.std(0)
            return mu.astype(np.float32), np.where(sd < CONST_SD, np.inf, np.maximum(sd, floor)).astype(np.float32)

        st = {}
        st["h_mu"], st["h_sd"] = ms(self.h[sub], 0.0)
        st["h_sd"] = np.where(np.isfinite(st["h_sd"]), st["h_sd"], 1.0).astype(np.float32)
        pk = sub[self.priv_ok[sub] > 0]
        st["p_mu"], st["p_sd"] = ms(self.priv[pk], PRIV_SD_FLOOR) if len(pk) else (np.zeros(self.priv.shape[1], np.float32), np.full(self.priv.shape[1], np.inf, np.float32))
        st["o_mu"], st["o_sd"] = ms(self.own[sub], OWN_SD_FLOOR)
        return st

    def batch(self, idx, k: int, variant: str, st: dict) -> dict:
        """Tensors of rows `idx` for predictor `variant` at k."""
        t = self.nxt[k][idx]
        b = {"h": torch.from_numpy(self.h[idx].astype(np.float32)), "y": torch.from_numpy(self.y[t].astype(np.float32))}
        p = (self.priv[t].astype(np.float32) - st["p_mu"]) / st["p_sd"]
        b["p"] = torch.from_numpy(np.nan_to_num(p, posinf=0.0, neginf=0.0))
        b["p_ok"] = torch.from_numpy((self.priv_ok[t] > 0).astype(np.float32))
        o = (self.own[t] - st["o_mu"]) / st["o_sd"]
        b["o"] = torch.from_numpy(np.nan_to_num(o, posinf=0.0, neginf=0.0).astype(np.float32))
        q = self.row_q[idx]
        b["q"] = torch.from_numpy(np.nan_to_num(q, nan=0.0).astype(np.float32))
        b["q_ok"] = torch.from_numpy(np.isfinite(q).astype(np.float32))
        b.update(self.inputs(idx, k, variant))
        return b

    def inputs(self, idx, k: int, variant: str, action=None) -> dict:
        """The conditioning inputs of `variant` (`action` overrides a_t: candidate scoring)."""
        out = {}
        if variant in ("A", "Ad"):
            out["a"] = torch.from_numpy((self.action[idx] if action is None else np.asarray(action)).astype(np.int64))
        if variant == "Ad":
            out["d"] = torch.from_numpy(self.delay[idx].astype(np.int64))
        if variant == "S":
            out["a"] = torch.from_numpy(self.action[self.partner[idx]].astype(np.int64))
        if variant == "H":
            win = self.order[self.pos[idx][:, None] + np.arange(k)[None]]
            acts = self.action[win].astype(np.int64)
            w = self.valid[win].astype(np.float32)
            hist = np.zeros((len(idx), self.n_act), np.float32)
            np.add.at(hist, (np.repeat(np.arange(len(idx)), k), acts.reshape(-1)), w.reshape(-1))
            hist /= np.maximum(hist.sum(1, keepdims=True), 1.0)
            out["hist"] = torch.from_numpy(hist)
        return out


class Predictor(nn.Module):
    def __init__(self, variant: str, d_h: int, d_y: int, d_p: int, d_o: int, n_act: int, n_delay: int):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant {variant!r}")
        self.variant = variant
        self.register_buffer("h_mu", torch.zeros(d_h))
        self.register_buffer("h_sd", torch.ones(d_h))
        self.emb_a = nn.Embedding(n_act, EMB_A) if variant in ("A", "Ad", "S") else None
        self.emb_d = nn.Embedding(n_delay, EMB_D) if variant == "Ad" else None
        d_in = d_h + (EMB_A if self.emb_a is not None else 0) + (EMB_D if self.emb_d is not None else 0) + (n_act if variant == "H" else 0)
        self.l1 = nn.Linear(d_in, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.y = nn.Linear(HIDDEN, d_y)
        self.p = nn.Linear(HIDDEN, d_p)
        self.o = nn.Linear(HIDDEN, d_o)
        self.q = nn.Linear(HIDDEN, len(Q_NAMES))

    def body(self, h, a=None, d=None, hist=None):
        x = [(h - self.h_mu) / self.h_sd]
        if self.emb_a is not None:
            x.append(self.emb_a(a))
        if self.emb_d is not None:
            x.append(self.emb_d(d))
        if self.variant == "H":
            x.append(hist)
        return F.relu(self.l2(F.relu(self.l1(torch.cat(x, 1)))))

    def forward(self, h, a=None, d=None, hist=None) -> dict:
        z = self.body(h, a, d, hist)
        return {"y": self.y(z), "p": self.p(z), "o": self.o(z), "q": self.q(z)}

    def q_all(self, h, cand) -> torch.Tensor:
        """q̂ logits [rows, C, 2] of variant A for every action of `cand` [C] (the same numbers as
        `forward` with a = c, the first layer split into its h' and Emb(a) parts)."""
        assert self.variant == "A"
        d_h = self.h_mu.shape[0]
        w = self.l1.weight
        zh = F.linear((h - self.h_mu) / self.h_sd, w[:, :d_h], self.l1.bias)
        za = F.linear(self.emb_a(cand), w[:, d_h:])
        z = F.relu(zh[:, None] + za[None])
        return self.q(F.relu(self.l2(z)))


def loss_terms(out: dict, b: dict, o_on: torch.Tensor, p_on: torch.Tensor) -> dict:
    """Per-row loss terms (6.4): y 1 - cos, p / o mean squared error over the kept dimensions,
    q BCE per target (0 where undefined)."""
    y = 1.0 - F.cosine_similarity(out["y"], b["y"], dim=1)
    p = ((out["p"] - b["p"]) ** 2 * p_on).sum(1) / p_on.sum().clamp_min(1)
    o = ((out["o"] - b["o"]) ** 2 * o_on).sum(1) / o_on.sum().clamp_min(1)
    q = F.binary_cross_entropy_with_logits(out["q"], b["q"], reduction="none") * b["q_ok"]
    return {"y": y, "p": p * b["p_ok"], "o": o, "q": q}


def train_one(data: Data, variant: str, k: int, rows: np.ndarray, st: dict, *, steps: int, batch: int, lr: float, seed: int, log_every: int = 500) -> tuple[Predictor, list]:
    """One predictor (fixed seed: init, batches); -> (predictor, loss log)."""
    torch.manual_seed(seed)
    net = Predictor(variant, data.h.shape[1], data.y.shape[1], data.priv.shape[1], data.own.shape[1], data.n_act, data.n_delay)
    net.h_mu.copy_(torch.from_numpy(st["h_mu"]))
    net.h_sd.copy_(torch.from_numpy(st["h_sd"]))
    o_on = torch.from_numpy(np.isfinite(st["o_sd"]).astype(np.float32))
    p_on = torch.from_numpy(np.isfinite(st["p_sd"]).astype(np.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    log = []
    acc = {}
    for step in range(1, steps + 1):
        idx = np.sort(rows[rng.integers(0, len(rows), batch)])
        b = data.batch(idx, k, variant, st)
        out = net(b["h"], b.get("a"), b.get("d"), b.get("hist"))
        lt = loss_terms(out, b, o_on, p_on)
        parts = {"y": lt["y"].mean(), "p": lt["p"].sum() / b["p_ok"].sum().clamp_min(1),
                 "o": lt["o"].mean(), "q": lt["q"].sum(0).div(b["q_ok"].sum(0).clamp_min(1)).sum()}
        loss = sum(parts.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for kk, v in parts.items():
            acc[kk] = acc.get(kk, 0.0) + float(v.detach())
        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            log.append({"step": step, **{kk: v / n for kk, v in acc.items()}})
            acc = {}
    net.eval()
    return net, log


@torch.no_grad()
def predict(net: Predictor, data: Data, idx: np.ndarray, k: int, st: dict, chunk: int = 4096, action=None, inputs_h=None, keep=("y", "p", "o", "q")) -> dict:
    """Outputs `keep` on rows `idx` (`action`: a_t replaced, `inputs_h`: rows whose h' is the input
    instead, e.g. the t+k rows for the oracle q̂_U(h'_{t+k})) -> {head: numpy float32}."""
    out = {k_: [] for k_ in keep}
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        hin = sl if inputs_h is None else inputs_h[s:s + chunk]
        x = data.inputs(sl, k, net.variant, None if action is None else action[s:s + chunk])
        o = net(torch.from_numpy(data.h[hin].astype(np.float32)), x.get("a"), x.get("d"), x.get("hist"))
        for k_ in out:
            out[k_].append(o[k_].numpy())
    return {k_: (np.concatenate(v) if v else np.zeros((0, 1), np.float32)) for k_, v in out.items()}


def save(net: Predictor, st: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "variant": net.variant, "stats": {k: v.tolist() for k, v in st.items()}, "meta": meta}, path)


def load(path: Path, data: Data) -> tuple[Predictor, dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Predictor(ck["variant"], data.h.shape[1], data.y.shape[1], data.priv.shape[1], data.own.shape[1], data.n_act, data.n_delay)
    net.load_state_dict(ck["state"])
    net.eval()
    st = {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}
    return net, st, ck["meta"]


def describe(data: Data) -> dict:
    out = {"rows": int(data.N), "games": int(data.G), "sources": []}
    for s in data.sources:
        gm = [e for e in data.game_meta if e["src"] == s.name]
        out["sources"].append({"name": s.name, "kind": s.kind, "role": s.role, "games": len(gm), "eval_games": sum(e["role"] == "eval" for e in gm),
                               "rows": int(sum(e.get("rows", 0) for e in gm))})
    return json.loads(json.dumps(out))
