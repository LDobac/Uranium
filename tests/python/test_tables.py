import json
import re
from pathlib import Path

from uranium import _core
from uranium import tables as tb

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "crates" / "uranium-env" / "src" / "features"


def body_of(src: str, head: str) -> str:
    body = src[src.index(head):]
    return body[:re.search(r"\n\}\r?\n", body).start()]


def test_entity_columns_match_the_extractor_030():
    src = (FEATURES / "extract.rs").read_text(encoding="utf-8")
    body = body_of(src, "fn entity_features(")
    stats = re.search(r"for \(i, \(v, m\)\) in \[(.*?)\]\.into_iter\(\)\.enumerate\(\) \{\s*f\[i \* 2\] = ratio\(.*?\);\s*f\[i \* 2 \+ 1\] = \(v as f32\)\.sqrt\(\) / 16\.0;", body, re.S)
    order = re.findall(r"\(u\.(\w+), max_\w+\)", stats.group(1))
    assert (tb.HP_RATIO, tb.HP_SQRT) == (2 * order.index("hp"), 2 * order.index("hp") + 1)
    assert (tb.SH_RATIO, tb.SH_SQRT) == (2 * order.index("shield"), 2 * order.index("shield") + 1)
    flags = re.search(r"let flag_bits = \[(.*?)\];", body, re.S).group(1)
    bits = [x.strip() for x in flags.split(",\n") if x.strip()]
    base = int(re.search(r"f\[(\d+) \+ i\] = b as u8 as f32;", body).group(1))
    assert tb.COMPLETED == base + bits.index("u.has(flags::COMPLETED)")
    assert tb.CONSTRUCTING == base + bits.index("u.has(flags::CONSTRUCTING)")
    assert tb.REMEMBERED == int(re.search(r"f\[(\d+)\] = e\.remembered as u8 as f32;", body).group(1))
    col, scale = re.search(r"f\[(\d+)\] = w\.frame\.frame\.saturating_sub\(e\.seen_frame\) as f32 / ([\d.]+);", body).groups()
    assert (tb.SINCE_SEEN, tb.SINCE_SCALE) == (int(col), float(scale))


def test_normalisers_match_the_extractor_030():
    src = (FEATURES / "extract.rs").read_text(encoding="utf-8")
    assert tb.COUNT_LOG == float(re.search(r"pub const COUNT_LOG: f32 = ([\d.]+);", src).group(1))
    scalar = body_of(src, "fn scalar(w: &World)")
    assert tb.FRAME_SCALE == float(re.search(r"v\.push\(f / ([\d.]+)\);", scalar).group(1))
    assert scalar.count("/ norm::COUNT_LOG") == 3


def test_owner_and_privileged_layout_match_the_schema_030():
    schema = (FEATURES / "schema.rs").read_text(encoding="utf-8")
    assert tb.OWNER_SELF == int(re.search(r"pub const OWNER_SELF: i32 = (\d+);", schema).group(1))
    assert tb.OWNER_ENEMY == int(re.search(r"pub const OWNER_ENEMY: i32 = (\d+);", schema).group(1))
    fixed = re.search(r"pub fn privileged\(&self\) -> usize \{\s*([\d +]+) \+ 2 \* self\.observed_types\.len\(\)", schema).group(1)
    assert tb.PRIV_BASE == sum(int(x) for x in fixed.split("+"))
    d = json.loads(_core.feature_dims())
    assert d["privileged"] == tb.PRIV_BASE + 2 * len(tb.observed())
    assert d["z_units"] == len(tb.terran())
    assert tb.priv_col(tb.observed()[-1], True) == d["privileged"] - 1
    assert tb.sc_enemy_cols().stop + d["actions"] + len(d["delay_buckets"]) + 1 == d["scalar"]


def test_action_names_cover_the_action_table_034():
    names = tb.action_names()
    assert len(names) == json.loads(_core.feature_dims())["actions"]
    assert names[tb.action_id("Build_Bunker")] == "Build_Bunker"
