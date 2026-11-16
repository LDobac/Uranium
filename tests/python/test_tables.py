import json
import re
from pathlib import Path

from uranium import _core
from uranium import tables as tb

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = "crates/uranium-env/src/features/extract.rs"
SCHEMA = "crates/uranium-env/src/features/schema.rs"


def source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def found(pattern: str, text: str, name: str, flags: int = 0) -> re.Match:
    m = re.search(pattern, text, flags)
    assert m is not None, f"{name}: no match for {pattern!r} (the Rust source changed shape; update this test)"
    return m


def body_of(src: str, head: str, name: str) -> str:
    assert head in src, f"{name}: no {head!r} (the Rust source changed shape; update this test)"
    body = src[src.index(head):]
    return body[:found(r"\n\}\r?\n", body, name).start()]


def position(items: list, item: str, name: str) -> int:
    assert item in items, f"{name}: {item!r} not in {items} (the Rust source changed shape; update this test)"
    return items.index(item)


def test_entity_columns_match_the_extractor_030():
    body = body_of(source(EXTRACT), "fn entity_features(", EXTRACT)
    stats = found(r"for \(i, \(v, m\)\) in \[(.*?)\]\.into_iter\(\)\.enumerate\(\) \{\s*f\[i \* 2\] = ratio\(.*?\);\s*f\[i \* 2 \+ 1\] = \(v as f32\)\.sqrt\(\) / 16\.0;", body, EXTRACT, re.S)
    order = re.findall(r"\(u\.(\w+), max_\w+\)", stats.group(1))
    hp, shield = position(order, "hp", EXTRACT), position(order, "shield", EXTRACT)
    assert (tb.HP_RATIO, tb.HP_SQRT) == (2 * hp, 2 * hp + 1)
    assert (tb.SH_RATIO, tb.SH_SQRT) == (2 * shield, 2 * shield + 1)
    flags = found(r"let flag_bits = \[(.*?)\];", body, EXTRACT, re.S).group(1)
    bits = [x.strip() for x in flags.split(",\n") if x.strip()]
    base = int(found(r"f\[(\d+) \+ i\] = b as u8 as f32;", body, EXTRACT).group(1))
    assert tb.COMPLETED == base + position(bits, "u.has(flags::COMPLETED)", EXTRACT)
    assert tb.CONSTRUCTING == base + position(bits, "u.has(flags::CONSTRUCTING)", EXTRACT)
    assert tb.REMEMBERED == int(found(r"f\[(\d+)\] = e\.remembered as u8 as f32;", body, EXTRACT).group(1))
    col, scale = found(r"f\[(\d+)\] = w\.frame\.frame\.saturating_sub\(e\.seen_frame\) as f32 / ([\d.]+);", body, EXTRACT).groups()
    assert (tb.SINCE_SEEN, tb.SINCE_SCALE) == (int(col), float(scale))


def test_normalisers_match_the_extractor_030():
    src = source(EXTRACT)
    assert tb.COUNT_LOG == float(found(r"pub const COUNT_LOG: f32 = ([\d.]+);", src, EXTRACT).group(1))
    scalar = body_of(src, "fn scalar(w: &World)", EXTRACT)
    assert tb.FRAME_SCALE == float(found(r"v\.push\(f / ([\d.]+)\);", scalar, EXTRACT).group(1))
    assert scalar.count("/ norm::COUNT_LOG") == 3


def test_owner_and_privileged_layout_match_the_schema_030():
    schema = source(SCHEMA)
    assert tb.OWNER_SELF == int(found(r"pub const OWNER_SELF: i32 = (\d+);", schema, SCHEMA).group(1))
    assert tb.OWNER_ENEMY == int(found(r"pub const OWNER_ENEMY: i32 = (\d+);", schema, SCHEMA).group(1))
    fixed = found(r"pub fn privileged\(&self\) -> usize \{\s*([\d +]+) \+ 2 \* self\.observed_types\.len\(\)", schema, SCHEMA).group(1)
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
