"""G0 decision-granularity report (tools/eval/granularity.py), input of AMB-031-7 (031-R3 12-unit
cap): synthetic per-game dumps in the `uranium-replay granularity` layout."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import granularity as gr  # noqa: E402

START = (1000, 1000)
A = gr.AID


def scv(i, x=1000, y=1000):
    return [i, gr.SCV, x, y, gr.COMPLETED, 60, False]


def ling(i, x=1000, y=1100):
    return [i, gr.LING, x, y]


def game(states, labels, scv_at=10, end_frame=12000, result=0):
    return gr.Game({"start": list(START), "states": states, "labels": labels, "scv_at": scv_at, "end_frame": end_frame, "result": result, "replay_sha256": "0" * 64})


def test_held_8m_and_winpass_031_r3():
    assert game([], [], scv_at=8).held_8m and game([], [], scv_at=8).held_8m_winpass
    assert not game([], [], scv_at=7).held_8m
    won_early = game([], [], scv_at=None, end_frame=9000, result=1)
    assert (won_early.held_8m, won_early.held_8m_winpass) == (False, True)
    lost_early = game([], [], scv_at=None, end_frame=9000, result=-1)
    assert (lost_early.held_8m, lost_early.held_8m_winpass) == (False, False)


def test_people_rush_contact_rule_031_r3():
    # 2 lings by 3:30, 4 by 4:30: a rush.
    s = [{"f": 4000, "own": [], "enemy": [ling(1), ling(2)]}, {"f": 5000, "own": [], "enemy": [ling(i) for i in range(4)]}]
    assert game(s, []).rush
    # 2 lings only after 3:30.
    late = [{"f": 5100, "own": [], "enemy": [ling(i) for i in range(4)]}]
    assert not game(late, []).rush
    # 2 lings by 3:30 but never 4, or the lings far from the main.
    assert not game([{"f": 4000, "own": [], "enemy": [ling(1), ling(2)]}], []).rush
    far = [{"f": 4000, "own": [], "enemy": [ling(i, 5000, 5000) for i in range(6)]}]
    assert not game(far, []).rush


def sat_game(eligible):
    own = [scv(i) for i in range(eligible)]
    state = {"f": 4000, "own": own, "enemy": []}
    label = [4000, A["Move_Pos"], 8, False, list(range(12)), None]
    return game([state], [label])


def test_sat12_needs_more_than_12_eligible_031_r3():
    s13 = gr.summarize("x", [sat_game(13)])["sel"]["all"]
    s12 = gr.summarize("x", [sat_game(12)])["sel"]["all"]
    assert (s13["n"], s13["eq12"], s13["sat12"]) == (1, 100.0, 100.0)
    assert (s12["n"], s12["eq12"], s12["sat12"]) == (1, 100.0, 0.0)


def test_burst_le2_031_r3():
    state = {"f": 3600, "own": [scv(1)], "enemy": []}
    labels = [[4000, A["Move_Pos"], 2, False, [1], None], [4002, A["Move_Pos"], 3, False, [1], None], [4005, A["Move_Pos"], 8, False, [1], None]]
    assert gr.summarize("x", [game([state], labels)])["burst_le2"] == 50.0


def test_smart_on_neutral_is_gather_031_r3():
    state = {"f": 3600, "own": [scv(1), [7, gr.BUNKER, 1100, 1000, gr.COMPLETED, 350, False]], "enemy": [ling(50)]}
    mineral = [4000, A["Smart_Unit"], 8, False, [1], 999]
    on_bunker = [4000, A["Smart_Unit"], 8, False, [1], 7]
    on_ling = [4000, A["Smart_Unit"], 8, False, [1], 50]
    assert gr.group_of(mineral, state) == "Gather"
    assert gr.group_of(on_bunker, state) == "Smart"
    assert gr.group_of(on_ling, state) == "Smart"
    # Mining clicks are not combat commands in the SCV micro count.
    fight = {"f": 3990, "own": [scv(1)], "enemy": [ling(50)]}
    st = gr.game_stats(game([fight], [mineral]))
    assert st["micro"] == [0] and st["micro_any"] == [1]
