import math

import pytest
import torch

from uranium import tables as tb
from uranium.bc import eval as ev

M = 128
CENTER = 64 * M + 64
ROWS = [
    ("Build_Barracks", (0, 0), True, 1.0),
    ("Move_Pos", (3, 4), True, 1.0),
    ("Attack_Pos", (0, 4), True, 1.0),
    ("SetRally_Pos", (1, 1), True, 1.0),
    ("ScannerSweep_Pos", (4, 1), True, 1.0),
    ("Attack_Unit", (9, 9), False, 1.0),
    ("Move_Pos", (0, 0), True, 0.0),
    ("Build_SupplyDepot", (2, 0), True, 1.0),
]
DELAY_FRAMES = [0, 2, 6, 16]


def batch():
    n = len(ROWS)
    i = torch.arange(n, dtype=torch.float32)
    action = torch.tensor([tb.action_id(a) for a, *_ in ROWS])
    a_logp = torch.log_softmax(torch.sin(torch.arange(n * 101, dtype=torch.float32).view(n, 101) * 0.37), 1)
    a_logp[torch.arange(n), action] = a_logp.max(1).values + torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    delay = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    d_logp = torch.log_softmax(torch.cos(torch.arange(n * 4, dtype=torch.float32).view(n, 4)), 1)
    loc = torch.full((n, M * M), -5.0)
    for r, (_, (dy, dx), _, _) in enumerate(ROWS):
        loc[r, CENTER + dy * M + dx] = 3.0
    sel = torch.zeros(n, 3, 5)
    lp = {"action": a_logp.gather(1, action[:, None])[:, 0], "delay": d_logp.gather(1, delay[:, None])[:, 0],
          "queued": -0.1 * (i + 1), "selected": -0.2 * (i + 1), "target_unit": -0.3 * (i + 1), "target_loc": -0.5 * (i + 1)}
    uses = torch.tensor([1, 1, 1, 1, 1, 0, 1, 1], dtype=torch.bool)
    out = {"dists": {"action": a_logp, "delay": d_logp, "target_loc": loc, "selected": sel,
                     "uses_queued": uses, "uses_select": uses, "uses_target_unit": ~uses, "uses_target_loc": uses}, "logp": lp}
    g = {"weights": torch.tensor([w for *_, w in ROWS]), "action": action, "delay": delay, "selected": torch.zeros(n, 3, dtype=torch.long),
         "target_loc": torch.full((n,), CENTER)}
    has_loc = [h for _, _, h, _ in ROWS]
    return out, g, has_loc, [0] * n


def scored(races=None):
    acc = ev.Acc(DELAY_FRAMES)
    out, g, has_loc, sel_len = batch()
    acc.score(out, g, has_loc, sel_len, races)
    return acc.report("ck", "val", 3, 1)


def test_loc_groups_known_input_045_ac2():
    rep = scored()["target_loc_groups"]
    want = {
        "all": (6, (0.5 + 1 + 1.5 + 2 + 2.5 + 4) / 6, 4 / 6, 1 / 6, (0 + 5 + 4 + math.sqrt(2) + math.sqrt(17) + 2) / 6),
        "build": (2, (0.5 + 4) / 2, 1.0, 0.5, 1.0),
        "move": (2, (1 + 1.5) / 2, 0.5, 0.0, 4.5),
        "rally": (1, 2.0, 1.0, 0.0, math.sqrt(2)),
        "other": (1, 2.5, 0.0, 0.0, math.sqrt(17)),
    }
    assert list(rep) == ["all", "build", "move", "rally", "other"]
    for k, (n, nll, within, exact, err) in want.items():
        assert rep[k]["steps"] == n, k
        assert [rep[k][x] for x in ("nll", "within4", "exact", "err_mean")] == pytest.approx([nll, within, exact, err], rel=1e-12), k


def test_loc_groups_split_the_location_steps_045_r7():
    rep = scored()
    groups = rep["target_loc_groups"]
    assert sum(groups[k]["steps"] for k in ev.LOC_GROUPS) == groups["all"]["steps"] == rep["target_loc_steps"]
    assert groups["all"]["exact"] == rep["target_loc_exact"]
    assert groups["all"]["err_mean"] == pytest.approx(rep["target_loc_err_mean"], rel=1e-6)
    empty = ev.Acc(DELAY_FRAMES).report("ck", "val", 0, 1)["target_loc_groups"]
    assert all(v == {"steps": 0, "nll": None, "within4": None, "exact": None, "err_mean": None} for v in empty.values())


def test_loc_group_classes_045_r7():
    names = tb.action_names()
    groups = ev.loc_group_of()
    loc = {a["id"] for a in tb.rows("actions") if a["target"] == "loc"}
    by = {k: sorted(names[i] for i in loc if int(groups[i]) == n) for n, k in enumerate(ev.LOC_GROUPS)}
    builds = sorted(names[i] for i in loc if names[i].startswith("Build_"))
    assert by["build"] == builds and len(builds) == 12
    assert by["move"] == ["Attack_Pos", "Move_Pos", "Patrol_Pos", "Smart_Pos"]
    assert by["rally"] == ["SetRally_Pos"]
    assert not set(by["other"]) & {"SetRally_Pos", *builds}
    assert len(groups) == len(names) == 101


OLD = {
    "checkpoint": "ck",
    "split": "val",
    "games": 3,
    "selection_seed": 1,
    "steps": 7,
    "action_top1": 0.7142857142857143,
    "action_top5": 0.7142857142857143,
    "delay_mae_frames": 6.571428571428571,
    "delay_mae_buckets": 1.1428571428571428,
    "target_loc_steps": 6,
    "target_loc_err_mean": 2.7562198440233865,
    "target_loc_err_median": 4.0,
    "target_loc_exact": 0.16666666666666666,
    "selected_steps": 0,
    "selected_f1": 0.0,
    "nll": {"action": 3.4308618818010603, "delay": 1.6733811923435755, "queued": 0.38333333904544514, "selected": 0.7666666780908903, "target_unit": 1.8000000715255737, "target_loc": 1.9166666666666667},
    "nll_steps": {"action": 7, "delay": 7, "queued": 6, "selected": 6, "target_unit": 1, "target_loc": 6},
    "ac1_pass": None,
    "by_enemy_race": {
        "terran": {"steps": 4, "action_top1": 0.5,
            "nll": {"action": 3.8568268418312073, "delay": 1.6880014091730118, "queued": 0.25000000558793545, "selected": 0.5000000111758709, "target_unit": 0.0, "target_loc": 1.25},
            "nll_steps": {"action": 4, "delay": 4, "queued": 4, "selected": 4, "target_unit": 0, "target_loc": 4}},
        "protoss": {"steps": 0, "action_top1": None,
            "nll": {"action": 0.0, "delay": 0.0, "queued": 0.0, "selected": 0.0, "target_unit": 0.0, "target_loc": 0.0},
            "nll_steps": {"action": 0, "delay": 0, "queued": 0, "selected": 0, "target_unit": 0, "target_loc": 0}},
        "zerg": {"steps": 3, "action_top1": 1.0,
            "nll": {"action": 2.8629086017608643, "delay": 1.6538875699043274, "queued": 0.6500000059604645, "selected": 1.300000011920929, "target_unit": 1.8000000715255737, "target_loc": 3.25},
            "nll_steps": {"action": 3, "delay": 3, "queued": 2, "selected": 2, "target_unit": 1, "target_loc": 2}},
    },
    "win_open_mse": None,
    "surrender": None,
    "units": "UNITS",
}
UNITS_KEYS = ["action_top1 / action_top5", "delay_mae_frames", "delay_mae_buckets", "target_loc_err_*", "nll.<head>", "steps", "win_open_mse",
              "surrender.*", "by_enemy_race.<race>", "aux_*, jepa_tgt_std"]


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}")
    else:
        yield path, x


def test_existing_report_unchanged_045_ac2():
    rep = scored({"terran": torch.tensor([True] * 4 + [False] * 4), "zerg": torch.tensor([False] * 4 + [True] * 4)})
    assert set(rep) - set(OLD) == {"target_loc_groups", "target_loc_groups_unit"}
    assert rep["units"] is ev.UNITS and list(ev.UNITS) == UNITS_KEYS
    got = dict(leaves({k: rep[k] for k in OLD if k != "units"}))
    want = dict(leaves({k: v for k, v in OLD.items() if k != "units"}))
    assert list(got) == list(want)
    for k, v in want.items():
        assert got[k] == (pytest.approx(v, rel=1e-6) if isinstance(v, float) else v), k
