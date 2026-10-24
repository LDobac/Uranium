"""Autoregressive action heads (032 §2.5): ActionType -> Delay -> Queued -> Selected ->
TargetUnit | Location. One code path serves teacher forcing (labels injected) and sampling
(Gumbel-max inside the forward, 032-R3, 032-R9). Same math and names as Burn `heads.rs`."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uranium.model.config import HeadsConfig, InputDims, SpatialConfig
from uranium.model.layers import GruCell, Linear, Mlp, ResMlp, entropy, env_choice, gumbel_argmax, log_softmax_f32, masked_log_softmax, pick, tempered

HEAD_NAMES = ("action", "delay", "queued", "selected", "target_unit", "target_loc")
TARGET_UNIT = 1
TARGET_LOC = 2
# AMB-000-4 step 2: teacher forcing computes the location head on the rows whose action uses a
# location (032-R10). `URANIUM_LOC_ROWS` picks how those rows are found and shaped:
#   off   - `nonzero` on the device (a host sync; the row count changes every call)
#   exact - the row list from the host labels (no sync; same shapes as `off`)
#   mult8 / pow2 - the host list padded to a multiple of 8 / a power of two by repeating a used
#                  row (fewer distinct shapes for cuDNN / MIOpen and graph capture); the padded
#                  rows' outputs are dropped, so the result is the same
LOC_ROW_MODES = ("off", "exact", "mult8", "pow2")
# `URANIUM_LOC_OUT`: the location head's last 1x1 conv (32 -> 1) as `conv` or as the same sum
# written out (`mulsum`: multiply, sum the channels), which MIOpen's slow 1x1 backward avoids.
LOC_OUT_MODES = ("conv", "mulsum")


def bucket_rows(k: int, mode: str) -> int:
    if mode == "mult8":
        return -(-k // 8) * 8
    if mode == "pow2":
        return 1 << math.ceil(math.log2(k)) if k > 1 else k
    return k


def location_rows(action, target_kind, mode: str) -> tuple[np.ndarray, int]:
    """Host row list for the location head from the action labels (numpy): (row indices padded
    to the bucket, number of real rows). No used row, or a bucket that covers the batch: every
    row, all real (the values of computing every row; one code path for torch.compile)."""
    a = np.clip(np.asarray(action).astype(np.int64).reshape(-1), 0, len(target_kind) - 1)
    used = np.flatnonzero(np.asarray(target_kind)[a] == TARGET_LOC)
    k, b = len(used), len(a)
    kb = bucket_rows(k, mode)
    if k == 0 or kb >= b:
        return np.arange(b), b
    return np.concatenate([used, np.full(kb - k, used[0], dtype=used.dtype)]), k


class ActionTable:
    """034 table columns as device constants for the mask gather (032 §5, 034-R4)."""

    def __init__(self, cols: dict, device):
        self.c = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in cols.items()}

    def to(self, device) -> "ActionTable":
        t = ActionTable.__new__(ActionTable)
        t.c = {k: v.to(device) for k, v in self.c.items()}
        return t

    def info(self, action):
        g = {k: v[action] for k, v in self.c.items()}
        return {
            "uses_select": g["uses_select"] == 1,
            "max_select": g["max_select"],
            "select_cap": g["select_cap"],
            "uses_queued": g["uses_queued"] == 1,
            "target_unit": g["target_kind"] == TARGET_UNIT,
            "target_loc": g["target_kind"] == TARGET_LOC,
            "target_cap": g["target_cap"],
            "loc_class": g["loc_class"],
            "self_target": g["self_target"] == 1,
        }


def gather_cap(unit_cap, cap):
    """`unit_cap[b, n, cap[b]]` -> [B,N] (034-R4)."""
    b, n, _ = unit_cap.shape
    return unit_cap.gather(2, cap.view(b, 1, 1).expand(b, n, 1)).view(b, n)


def gather_loc(loc_class_mask, cls):
    """`loc_class_mask[b, class[b]]` flattened -> [B,M*M] (034-R4)."""
    b, l, h, w = loc_class_mask.shape
    return loc_class_mask.view(b, l, h * w).gather(1, cls.view(b, 1, 1).expand(b, 1, h * w)).view(b, h * w)


class LocationHead(nn.Module):
    def __init__(self, inp: InputDims, h: HeadsConfig, s: SpatialConfig):
        super().__init__()
        side = inp.map_size // 16
        c1, c2, c3, c4 = s.stages
        up = lambda i, o: nn.ConvTranspose2d(i, o, 4, stride=2, padding=1)  # noqa: E731
        self.ar = Linear(h.ar, h.loc_ar_channels * side * side)
        self.ups = nn.ModuleList([up(h.loc_ar_channels + c4, c3), up(2 * c3, c2), up(2 * c2, c1), up(2 * c1, 32)])
        self.out = nn.Conv2d(32, 1, 1)
        self.ar_channels = h.loc_ar_channels
        self.out_mode = env_choice("URANIUM_LOC_OUT", LOC_OUT_MODES, "conv")

    def forward(self, ar, sp: dict):
        """-> logits [B, M*M]"""
        b, _, s, _ = sp["s8"].shape
        x = torch.cat([F.relu(self.ar(ar)).view(b, self.ar_channels, s, s), sp["s8"]], 1)
        skips = [sp["skips"][2], sp["skips"][1], sp["skips"][0]]
        for i, up in enumerate(self.ups):
            x = F.relu(up(x))
            if i < 3:
                x = torch.cat([x, skips[i]], 1)
        m = x.shape[2]
        # mulsum is the 1x1 conv's sum written out: only for a 1x1, single-output conv.
        if self.out_mode == "mulsum" and self.out.kernel_size == (1, 1) and self.out.out_channels == 1 and self.out.groups == 1:
            w = self.out.weight.view(1, -1, 1, 1).to(x.dtype)
            return ((x * w).sum(1) + self.out.bias.to(x.dtype)).view(b, m * m)
        return self.out(x).view(b, m * m)


class Heads(nn.Module):
    def __init__(self, inp: InputDims, h: HeadsConfig, s: SpatialConfig, d_core: int, d_ctx: int, d_entity: int):
        super().__init__()
        # 1 ActionType
        self.type_in = Linear(d_core, h.ar)
        self.type_res = nn.ModuleList(ResMlp(h.ar) for _ in range(2))
        self.type_gate = Linear(d_ctx, h.ar)
        self.type_out = Linear(h.ar, inp.actions)
        self.ar_h = Linear(d_core, h.ar)
        self.ar_action = nn.Embedding(inp.actions, h.ar)
        # 2 Delay
        self.delay = Mlp(h.ar, 128, h.delay_buckets)
        self.ar_delay = nn.Embedding(h.delay_buckets, h.ar)
        # 3 Queued
        self.queued = Mlp(h.ar, 64, 2)
        self.ar_queued = nn.Embedding(2, h.ar)
        # 4 Selected
        self.sel_key = Linear(d_entity, h.key)
        self.sel_end = nn.Parameter(torch.randn(h.key) * 0.02)
        self.sel_query = Linear(h.ar, h.key)
        self.sel_gru = GruCell(h.key, h.key)
        self.ar_selected = Linear(h.key, h.ar)
        # 5 TargetUnit
        self.tgt_key = Linear(d_entity, h.key)
        self.tgt_query = Linear(h.ar, h.key)
        # 6 Location
        self.location = LocationHead(inp, h, s)
        self.max_select = h.max_select
        # Tests only: teacher forcing computes the location head on every row.
        self.location_all_rows = False

    def forward(self, labels: dict | None, temp: float, h, c, e, sp: dict, obs: dict, table: ActionTable, generator=None) -> dict:
        """`labels` None = sample (temp 0 = argmax); else teacher forcing (032-R3). Returns
        `chosen` (labels or samples), per-head `logp` / `entropy` [B] (already times the head's
        use, 032-R10), and the distributions `dists`."""
        b, n, _ = e.shape
        temp = 1.0 if labels is not None else temp

        def choose(logp, key):
            if labels is not None:
                return labels[key].long().clamp(0, logp.shape[1] - 1)
            return gumbel_argmax(logp, temp, generator)

        # 1 ActionType
        x = F.relu(self.type_in(h))
        for r in self.type_res:
            x = r(x)
        x = F.relu(x) * torch.sigmoid(self.type_gate(c))
        lp_action = tempered(masked_log_softmax(self.type_out(x), obs["type_mask"]), temp)
        action = choose(lp_action, "action")
        info = table.info(action)
        ar = self.ar_h(h) + self.ar_action(action)

        # 2 Delay (always used)
        lp_delay = tempered(log_softmax_f32(self.delay(ar)), temp)
        delay = choose(lp_delay, "delay")
        ar = ar + self.ar_delay(delay)

        # 3 Queued (fixed to 0 when unused)
        lp_queued = tempered(log_softmax_f32(self.queued(ar)), temp)
        queued = choose(lp_queued, "queued") * info["uses_queued"].long()
        ar = ar + self.ar_queued(queued)

        # 4 Selected
        sel, ar = self.selected(ar, e, obs, info, None if labels is None else labels["selected"].long(), temp, generator)

        # 5 TargetUnit
        logits = (self.tgt_key(e) @ self.tgt_query(ar)[:, :, None]).squeeze(2)
        # 034-R9: never a unit of this step's selection, unless every eligible target was
        # selected (then the plain mask; the executor rejects the action, 031-R4).
        cap_ok = gather_cap(obs["unit_cap"], info["target_cap"]) & obs["entity_mask"]
        taken = sel["units"] & ~info["self_target"][:, None]
        free = cap_ok & ~taken
        none_free = ~free.any(1, keepdim=True)
        lp_tu = tempered(masked_log_softmax(logits, free | (cap_ok & none_free)), temp)
        target_unit = choose(lp_tu, "target_unit")

        # 6 Location
        loc_ok = gather_loc(obs["loc_class_mask"], info["loc_class"])
        if labels is not None:
            lp_loc = self.location_used_rows(ar, sp, loc_ok, info["target_loc"], labels.get("loc_rows"), labels.get("loc_w"))
        else:
            lp_loc = tempered(masked_log_softmax(self.location(ar, sp), loc_ok), temp)
        target_loc = choose(lp_loc, "target_loc")

        u_q, u_s, u_tu, u_tl = (info[k].float() for k in ("uses_queued", "uses_select", "target_unit", "target_loc"))
        logp = {
            "action": pick(lp_action, action),
            "delay": pick(lp_delay, delay),
            "queued": pick(lp_queued, queued) * u_q,
            "selected": sel["logp"] * u_s,
            "target_unit": pick(lp_tu, target_unit) * u_tu,
            "target_loc": pick(lp_loc, target_loc) * u_tl,
        }
        ent = {
            "action": entropy(lp_action),
            "delay": entropy(lp_delay),
            "queued": entropy(lp_queued) * u_q,
            "selected": sel["entropy"] * u_s,
            "target_unit": entropy(lp_tu) * u_tu,
            "target_loc": entropy(lp_loc) * u_tl,
        }
        chosen = {"action": action, "delay": delay, "queued": queued, "selected": sel["chosen"], "target_unit": target_unit, "target_loc": target_loc}
        dists = {
            "action": lp_action, "delay": lp_delay, "queued": lp_queued, "selected": sel["dists"], "selected_active": sel["active"],
            "target_unit": lp_tu, "target_loc": lp_loc, "uses_queued": info["uses_queued"], "uses_select": info["uses_select"],
            "uses_target_unit": info["target_unit"], "uses_target_loc": info["target_loc"],
        }
        return {"chosen": chosen, "logp": logp, "entropy": ent, "dists": dists}

    def location_used_rows(self, ar, sp: dict, loc_ok, used, rows=None, w=None):
        """Masked location log-probs [B, M*M] for the rows whose action uses a location; other rows
        0 (their terms are multiplied by 0, 032-R10). No used row or all rows: every row. `rows` /
        `w`: the host row list (`location_rows`) and its weights (1 real, 0 padding: a repeat of a
        real row, so adding it times 0 keeps the value and gradient); else the rows are found on
        the device."""
        if self.location_all_rows:
            return masked_log_softmax(self.location(ar, sp), loc_ok)
        b, mm = loc_ok.shape
        if rows is None:
            rows = used.nonzero().squeeze(1)
            if rows.numel() in (0, b):
                return masked_log_softmax(self.location(ar, sp), loc_ok)
        sub = {"s8": sp["s8"][rows], "skips": [s[rows] for s in sp["skips"]], "pool": sp["pool"][rows]}
        lp = masked_log_softmax(self.location(ar[rows], sub), loc_ok[rows])
        if w is not None:
            lp = lp * w[:, None]
        return torch.zeros(b, mm, dtype=lp.dtype, device=lp.device).index_add(0, rows, lp)

    def selected(self, ar, e, obs: dict, info: dict, labels, temp: float, generator):
        """Pointer network over own eligible units with an END key (032 §2.5 #4, 031-R3, 032-R12).
        At most `max_select[a]` picks; END is allowed after the first pick (or when nothing is
        eligible); reaching `max_select` ends the selection. 034-R7: a building only alone."""
        b, n, _ = e.shape
        dev = e.device
        keys = self.sel_key(e)
        kd = keys.shape[2]
        keys_all = torch.cat([keys, self.sel_end.to(keys.dtype).view(1, 1, kd).expand(b, 1, kd)], 1)
        is_self = obs["entity_cat"][..., 1] == 0
        base = gather_cap(obs["unit_cap"], info["select_cap"]) & is_self & obs["entity_mask"]
        not_building = ~obs["is_building"]
        first_building = torch.zeros(b, 1, dtype=torch.bool, device=dev)
        chosen_units = torch.zeros(b, n, dtype=torch.bool, device=dev)
        ended = ~info["uses_select"]
        q = self.sel_query(ar)
        sum_keys = torch.zeros(b, kd, dtype=q.dtype, device=dev)
        count = torch.zeros(b, device=dev)
        logp = torch.zeros(b, device=dev)
        ent = torch.zeros(b, device=dev)
        true_col = torch.ones(b, 1, dtype=torch.bool, device=dev)
        choices, dists, actives = [], [], []
        for k in range(self.max_select):
            active = ~ended & (info["max_select"] > k)
            units_ok = base & ~chosen_units
            if k > 0:
                units_ok = units_ok & not_building & ~first_building
            end_ok = ~units_ok.any(1, keepdim=True) if k == 0 else true_col
            allowed = torch.cat([units_ok, end_ok], 1)
            logits = (keys_all @ q[:, :, None]).squeeze(2)
            lp = tempered(masked_log_softmax(logits, allowed), temp)
            choice = labels[:, k].clamp(0, n) if labels is not None else gumbel_argmax(lp, temp, generator)
            af = active.float()
            logp = logp + pick(lp, choice) * af
            ent = ent + entropy(lp) * af
            is_end = choice == n
            take = active & ~is_end
            # The chosen row gathered (the value and gradient of one-hot @ keys; a one-hot needs
            # the entity count as a Python int, which pins torch.compile to one shape).
            key = keys_all.gather(1, choice[:, None, None].expand(b, 1, kd)).squeeze(1)
            picked = (torch.arange(n, device=dev)[None] == choice[:, None]) & take[:, None]
            tf = take.float()
            if k == 0:
                first_building = (picked & obs["is_building"]).any(1, keepdim=True)
            chosen_units = chosen_units | picked
            sum_keys = sum_keys + key * tf[:, None].to(key.dtype)
            count = count + tf
            q = self.sel_gru(key, q)
            ended = ended | (active & is_end) | (info["max_select"] <= k + 1)
            # END for inactive steps so `chosen.selected` is canonical (032-R12).
            choices.append(torch.where(active, choice, torch.full_like(choice, n))[:, None])
            dists.append(lp[:, None])
            actives.append(active[:, None])
        mean_key = sum_keys / count.clamp_min(1.0)[:, None].to(sum_keys.dtype)
        ar = ar + self.ar_selected(mean_key) * info["uses_select"].to(ar.dtype)[:, None]
        sel = {"units": chosen_units, "chosen": torch.cat(choices, 1), "logp": logp, "entropy": ent, "dists": torch.cat(dists, 1), "active": torch.cat(actives, 1)}
        return sel, ar
