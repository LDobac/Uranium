"""Per-row features of a lane's chunk (own-vision inputs and the privileged offline labels) and the
per-game labels (reports/jepa_survey.md 7.2-7.3), for one enemy race.

The focus building of the race (Barracks / Gateway / Spawning Pool, `tables.race`):
- the row's focus token (`tok_k`): a visible one first, else the most recently seen remembered one
  (the probes' token, as the earlier Zerg run)
- the game's anchor (`a_k`): the first focus token seen in the game, followed by its position (the
  counterfactual edits' token, as the earlier Terran / Protoss run)
- own-vision start estimate of an unfinished token: seen frame - (hp - 0.1 max) * build / (0.9 max)
"""

import numpy as np

from uranium.diag import tables as tb

RACE_INPUT = {0: "T", 1: "P", 2: "Z"}  # scalar_cat[0]: enemy race input (3 = unknown)


def race_of(hg: dict, rows: np.ndarray) -> str:
    """The enemy race of a lane's game from the privileged counts of the first real row (workers
    and town halls exist from the start), else the race input, else "U"."""
    if len(rows):
        pv = np.asarray(hg["privileged"])[rows[0]]
        score = {}
        for r in tb.RACES:
            s = tb.race(r)
            cols = [tb.priv_col(s["worker"]), tb.priv_col(s["worker"], True)] + [tb.priv_col(t) for t in s["town"]]
            score[r] = float(tb.decode_count(pv[cols]).sum())
        best = max(score, key=score.get)
        if score[best] > 0:
            return best
        ri = int(np.asarray(hg["scalar_cat"])[rows[0], 0])
        return RACE_INPUT.get(ri, "U")
    return "U"


def est_start(seen, hp, max_hp, build_time):
    return seen - (hp - 0.1 * max_hp) * build_time / (0.9 * max_hp)


def lane_features(g: dict, race: str, lg: dict, n: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Features of one lane's chunk `g` (T rows, the first `n` real) for enemy race `race`;
    `lg` carries the game's anchor across chunks. -> (features {name: [T]}, tok_k [T], a_k [T])
    (token indices, -1 = none)."""
    cat = np.asarray(g["entity_cat"])
    cont = np.asarray(g["entity_cont"])
    m = np.asarray(g["entity_mask"]).astype(bool)
    pos = np.asarray(g["entity_pos"])
    sc = np.asarray(g["scalar"])
    scat = np.asarray(g["scalar_cat"])
    pv = np.asarray(g["privileged"])
    bld = np.asarray(g["is_building"]).astype(bool)
    frame = sc[:, 0] * tb.FRAME_SCALE
    t = len(frame)
    rows = np.arange(t)
    ty = cat[:, :, 0]
    en = (cat[:, :, 1] == tb.OWNER_ENEMY) & m
    own = (cat[:, :, 1] == tb.OWNER_SELF) & m
    rem = cont[:, :, tb.REMEMBERED] > 0.5
    vis = en & ~rem
    f = {"frame": frame, "map": scat[:, 1], "start": scat[:, 2], "race_in": scat[:, 0]}
    if race in tb.RACES:
        s = tb.race(race)
        fid, mx, msh, bt = s["focus"], s["max_hp"], s["max_shields"], s["build_time"]
        foc = en & (ty == fid)
        hp_all = np.rint((cont[:, :, tb.HP_SQRT] * 16.0) ** 2)
        seen_all = frame[:, None] - cont[:, :, tb.SINCE_SEEN] * tb.SINCE_SCALE * rem
        comp = cont[:, :, tb.COMPLETED] > 0.5
        unf = foc & ~comp
        est_all = est_start(seen_all, hp_all, mx, bt)
        est_min = np.where(unf, est_all, np.inf).min(1)
        # the row's token: visible first, else the most recently seen remembered one
        score = np.where(foc, 1e6 * (~rem) - cont[:, :, tb.SINCE_SEEN], -np.inf)
        has = foc.any(1)
        tok = np.where(has, score.argmax(1), -1)
        k = np.where(has, tok, 0)
        pc = cont[rows, k]
        f.update({
            "n_foc": foc.sum(1), "n_foc_vis": (foc & ~rem).sum(1), "n_foc_unf": unf.sum(1), "n_foc_done": (foc & comp).sum(1),
            "foc_vis": has & ~rem[rows, k], "foc_rem": has & rem[rows, k],
            "foc_hp": np.where(has, hp_all[rows, k], np.nan), "foc_hpr": np.where(has, pc[:, tb.HP_RATIO], np.nan),
            "foc_shield": np.where(has, np.rint((pc[:, tb.SH_SQRT] * 16.0) ** 2), np.nan),
            "foc_comp": np.where(has, pc[:, tb.COMPLETED], np.nan), "foc_cons": np.where(has, pc[:, tb.CONSTRUCTING], np.nan),
            "foc_order": np.where(has, cat[rows, k, 2], -1), "foc_since": np.where(has, pc[:, tb.SINCE_SEEN] * tb.SINCE_SCALE, np.nan),
            "foc_seen": np.where(has, seen_all[rows, k], np.nan), "foc_est": np.where(has & ~comp[rows, k], est_all[rows, k], np.nan),
            "est_min": np.where(np.isfinite(est_min), est_min, np.nan),
        })
        # anchor: the first focus token of the game, by position
        anc = np.full(t, -1)
        for r in range(n):
            if lg.get("anchor") is None and foc[r].any():
                kk = int(np.argmax(foc[r]))
                lg["anchor"] = pos[r, kk].tolist()
                lg["anchor_frame"] = float(frame[r])
            if lg.get("anchor") is not None:
                d = np.abs(pos[r] - np.asarray(lg["anchor"])[None]).sum(1)
                hit = foc[r] & (d < 1.0)
                if hit.any():
                    anc[r] = int(np.argmax(hit))
        ka = np.where(anc >= 0, anc, 0)
        ha = anc >= 0
        f.update({"a_has": ha, "a_comp": np.where(ha, cont[rows, ka, tb.COMPLETED], np.nan), "a_seen": np.where(ha, seen_all[rows, ka], np.nan)})
        f["vis_worker"] = (vis & (ty == s["worker"])).sum(1)
        f["vis_army"] = (vis & np.isin(ty, s["army"])).sum(1)
        f["tok_army"] = (en & np.isin(ty, s["army"])).sum(1)
        f["tok_town"] = (en & np.isin(ty, s["town"])).sum(1)
        f["tok_gas"] = (en & (ty == s["gas"])).sum(1)
        f["sc_worker"] = tb.decode_count(sc[:, tb.sc_seen_col(s["worker"])])
        f["sc_focus"] = tb.decode_count(sc[:, tb.sc_seen_col(fid)])
        f["sc_town"] = sum(tb.decode_count(sc[:, tb.sc_seen_col(x)]) for x in s["town"])
        f["sc_gas"] = tb.decode_count(sc[:, tb.sc_seen_col(s["gas"])])
        f["sc_army"] = sum(tb.decode_count(sc[:, tb.sc_seen_col(x)]) for x in s["army"])
    else:
        tok = anc = np.full(t, -1)
    f["vis_enemy"] = vis.sum(1)
    f["tok_enemy"] = en.sum(1)
    f["tok_bld"] = (en & bld).sum(1)
    f["sc_enemy_total"] = tb.decode_count(sc[:, tb.sc_enemy_cols()]).sum(1)
    for nm, ty_ in tb.own_ids().items():
        d, b = tb.sc_own_cols(ty_)
        f[f"own_{nm}"] = tb.decode_count(sc[:, d]) + tb.decode_count(sc[:, b])
    for x in tb.priv_types():
        f[f"pv{x}_d"] = tb.decode_count(pv[:, tb.priv_col(x)]).astype(np.int16)
        f[f"pv{x}_p"] = tb.decode_count(pv[:, tb.priv_col(x, True)]).astype(np.int16)
    tm = np.asarray(g["type_mask"]).astype(bool)
    acts = tb.actions()
    f["el_b"], f["el_m"], f["el_r"] = tm[:, acts["bunker"]], tm[:, acts["marine"]], tm[:, acts["rax"]]
    f["label"] = np.asarray(g["action"]).astype(np.int64)
    f["ok"] = np.asarray(g["weights"]) > 0
    return f, tok, anc


# Columns every race's rows have (so rows of different races concatenate); filled with a default
# for a race without them ("U").
RACE_COLS = {
    "n_foc": 0, "n_foc_vis": 0, "n_foc_unf": 0, "n_foc_done": 0, "foc_vis": False, "foc_rem": False, "foc_hp": np.nan, "foc_hpr": np.nan, "foc_shield": np.nan,
    "foc_comp": np.nan, "foc_cons": np.nan, "foc_order": -1, "foc_since": np.nan, "foc_seen": np.nan, "foc_est": np.nan, "est_min": np.nan,
    "a_has": False, "a_comp": np.nan, "a_seen": np.nan, "vis_worker": 0, "vis_army": 0, "tok_army": 0, "tok_town": 0, "tok_gas": 0,
    "sc_worker": 0, "sc_focus": 0, "sc_town": 0, "sc_gas": 0, "sc_army": 0,
}


def fill_race_cols(f: dict, t: int) -> dict:
    for k, v in RACE_COLS.items():
        if k not in f:
            f[k] = np.full(t, v)
    return f


# ---------------------------------------------------------------- per-game labels

F230, F300, F400, F430, F500 = 3600, 4320, 5760, 6480, 7200
RUSH_Z = 2280  # the teacher's pool start threshold (react.rs pool_start_max, 95 s)


def _first(mask, fr):
    i = np.nonzero(mask)[0]
    return float(fr[i[0]]) if len(i) else None


def game_labels(r: dict, race: str) -> dict:
    """Labels of one game from its rows `r` (frame-sorted, every dumped row): true starts of the
    focus type (privileged), the build class (7.3 draft rules), first sight and the own-vision
    estimate at first sight."""
    s = tb.race(race)
    F, mx, bt = s["focus"], s["max_hp"], s["build_time"]
    fr = r["frame"]
    tot = lambda x: r[f"pv{x}_p"].astype(int) + r[f"pv{x}_d"].astype(int)  # noqa: E731
    done = lambda x: r[f"pv{x}_d"].astype(int)  # noqa: E731
    by = lambda x, k, f_: bool(((tot(x) >= k) & (fr < f_)).any())  # noqa: E731
    u = tb.unit_id
    e = {"race": race, "rows": int(len(fr)), "last_frame": float(fr[-1])}
    tf = tot(F)
    for k in (1, 2, 3):
        e[f"start{k}"] = _first(tf >= k, fr)
        d = _first(done(F) >= k, fr)
        e[f"done{k}_minus_bt"] = None if d is None else d - bt
    e["n_focus_by_400"] = int(tf[fr < F400].max()) if (fr < F400).any() else 0
    e["focus2_by_300"] = by(F, 2, F300)
    if race == "T":
        rush = by(u("Terran_Barracks"), 2, F230) and not by(u("Terran_Refinery"), 1, F300)
        macro = by(u("Terran_Factory"), 1, F400) or by(u("Terran_Command_Center"), 2, F500)
        e["cls"] = "rush" if rush else ("macro" if macro else "other")
    elif race == "P":
        core1 = _first(tot(u("Protoss_Cybernetics_Core")) >= 1, fr)
        rush = by(u("Protoss_Gateway"), 2, F230) and not by(u("Protoss_Nexus"), 2, F400)
        one_gate_core = core1 is not None and core1 < F430 and int(tf[fr <= core1].max()) == 1
        macro = by(u("Protoss_Nexus"), 2, F430) or one_gate_core
        e["cls"] = "rush" if rush else ("macro" if macro else "other")
    else:
        # the HP clock runs from completion - build time; else the first in-production step
        d1 = _first(done(F) >= 1, fr)
        e["start1"] = (d1 - bt) if d1 is not None else e["start1"]
        hatch2 = (tot(u("Zerg_Hatchery")) >= 2) & (fr < (d1 if d1 is not None else np.inf))
        ps = e["start1"]
        e["cls"] = "no_pool" if ps is None else ("rush" if ps <= RUSH_Z else ("macro" if hatch2.any() else "onebase"))
    vis = np.asarray(r["foc_vis"]).astype(bool)
    i = np.nonzero(vis)[0]
    e["first_sight"] = float(fr[i[0]]) if len(i) else None
    e["sight_comp"] = None
    e["est_at_sight"] = None
    if len(i):
        j = i[0]
        comp = bool(r["foc_comp"][j] > 0.5)
        e["sight_comp"] = comp
        e["sight_hp_ratio"] = float(r["foc_hp"][j]) / mx
        if not comp:
            e["est_at_sight"] = float(est_start(r["foc_seen"][j], r["foc_hp"][j], mx, bt))
            if e["start1"] is not None and int(tf[j]) == 1:
                e["est_err_start1"] = e["est_at_sight"] - e["start1"]
    a = np.nonzero(np.asarray(r["a_has"]).astype(bool))[0]
    e["anchor_first"] = float(fr[a[0]]) if len(a) else None
    e["anchor_unf_at_sight"] = bool(len(a) and r["a_comp"][a[0]] < 0.5)
    e["seen2_frame"] = _first(r["sc_focus"] >= 2, fr)
    return e
