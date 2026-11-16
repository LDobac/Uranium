import json

import pytest

from eval_helpers import LIN, all_outputs, fake_game_order, record

import diag_table


@pytest.fixture(autouse=True)
def _game_order(monkeypatch):
    fake_game_order(monkeypatch)


def test_diag_table_050_r9(tmp_path):
    out = all_outputs(tmp_path / "d")
    res = diag_table.folder_table(LIN, out, ["cue_diag", "dead_channels"])
    hp = res["cue"]["teacher"]["Z"]["hp|unf_token"]
    assert (hp["rows"], hp["games"], hp["stages"]["h"]["pt"], hp["h_mlp"]) == (17097, 154, 0.5, 0.45)
    assert res["cue"]["human"]["P"]["count|focus_rows"]["stages"]["h"]["pt"] == 0.5
    assert res["intervention"]["teacher"]["Z"]["effects"]["start"]["bunker_any"]["mean"] == 0.01
    md = "\n".join(diag_table.markdown(LIN, {"m2-s1": res}))
    assert "### cue probe, human hold-out, T `count|focus_rows`" in md and "| m2-s1 | 17097 / 154 | 0.500 [0.400, 0.600]" in md
    assert res["cue"]["human"]["T"]["count|focus_rows"]["not_measurable"].startswith("TvT") and "| m2-s1 | 측정 불가 (not measurable): TvT" in md
    assert "### counterfactual intervention, teacher hold-out, Z" in md and "r4 39.8% (51 @ 3000)" in md
    ck1 = all_outputs(tmp_path / "c")
    (ck1 / "spatial_log.json").unlink()
    r1 = diag_table.folder_table(LIN, record(ck1, metrics=["cue_diag"]), ["cue_diag"]) | {"train_log": None}
    assert "| ck1 | not measured (no training-log statistics) |" in "\n".join(diag_table.markdown(LIN, {"ck1": r1}))
    iv = json.loads((out / "teacher" / "intervene.json").read_text(encoding="utf-8"))
    (out / "teacher" / "intervene.json").write_text(json.dumps({**iv, "window": [0.0, 1.0]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSED: .*intervene window"):
        diag_table.main([f"x={out}"])
