"""052-R7: views of the comparison sets measured outside the eval jobs (tools/eval/ops_sets.py)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
import ops_sets  # noqa: E402


def test_botgen_view_is_the_teacher_slot_in_tvt_052_r7(tmp_path):
    rows = [
        {"game_id": 1, "slot": 0, "self_race": "Terran", "t0_build": "terran_rush", "enemy_race": "Terran"},
        {"game_id": 1, "slot": 1, "self_race": "Terran", "t0_build": "terran_react", "enemy_race": "Terran"},
        {"game_id": 2, "slot": 0, "self_race": "Terran", "t0_build": "terran_react", "enemy_race": "Zerg"},
        {"game_id": 2, "slot": 1, "self_race": "Zerg", "t0_build": "zerg_rush", "enemy_race": "Terran"},
        {"game_id": 3, "slot": 0, "self_race": "Zerg", "t0_build": "standard", "enemy_race": "Terran"},
    ]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = ops_sets.teacher_games(tmp_path, "terran_react")
    assert {n: r["slot"] for n, r in got.items()} == {1: 1, 2: 0}, "the teacher, not its Terran opponent; no teacher row, no game"


def test_human_replay_found_by_sha256_052_r7(tmp_path):
    h = tmp_path / "datasets/replays/build/hashes"
    h.mkdir(parents=True)
    (h / "a.sha256").write_text("ab12 *stardata/x/1.rep\ncd34 *scr_ladder/2.rep\n", encoding="utf-8")
    idx = ops_sets.sha_index(tmp_path)
    assert idx == {"ab12": tmp_path / "datasets/replays/raw/stardata/x/1.rep", "cd34": tmp_path / "datasets/replays/raw/scr_ladder/2.rep"}
