import functools
import tomllib
from pathlib import Path

import numpy as np

TABLES = Path(__file__).resolve().parents[2] / "crates" / "uranium-env" / "tables"

HP_RATIO, HP_SQRT, SH_RATIO, SH_SQRT = 0, 1, 2, 3
COMPLETED = 48
CONSTRUCTING = 59
REMEMBERED = 72
SINCE_SEEN = 73
SINCE_SCALE = 2048.0
OWNER_SELF, OWNER_ENEMY = 0, 1
FRAME_SCALE = 86400.0
COUNT_LOG = 4.0


@functools.cache
def rows(name: str) -> list:
    key = "action" if name == "actions" else "rows"
    return tomllib.loads((TABLES / f"{name}.toml").read_text(encoding="utf-8"))[key]


@functools.cache
def units() -> dict:
    return {u["id"]: u for u in rows("unit_types")}


def _playable(u) -> bool:
    return u["race"] in ("Terran", "Zerg", "Protoss") and "Hero" not in u["flags"] and "Subunit" not in u["flags"]


@functools.cache
def observed() -> list[int]:
    us = sorted(units().values(), key=lambda u: u["id"])
    return [u["id"] for u in us if _playable(u) and u["name"] != "Terran_Civilian" and not u["name"].startswith("Zerg_Broodling")]


@functools.cache
def terran() -> list[int]:
    us = sorted(units().values(), key=lambda u: u["id"])
    return [u["id"] for u in us if u["race"] == "Terran" and _playable(u) and u["name"] != "Terran_Civilian"]


@functools.cache
def _sc_enemy_seen() -> int:
    return 2 + 5 + 63 + 47 + 47 + 63 + 2 * len(terran())


def sc_seen_col(ty: int) -> int:
    return _sc_enemy_seen() + observed().index(ty)


def sc_own_cols(ty: int) -> tuple[int, int]:
    base = _sc_enemy_seen() - 2 * len(terran()) + 2 * terran().index(ty)
    return base, base + 1


def sc_enemy_cols() -> slice:
    return slice(_sc_enemy_seen(), _sc_enemy_seen() + len(observed()))


PRIV_BASE = 115


def priv_col(ty: int, in_prod: bool = False) -> int:
    return PRIV_BASE + 2 * observed().index(ty) + int(in_prod)


def decode_count(x) -> np.ndarray:
    return np.rint(np.expm1(np.asarray(x, dtype=np.float64) * COUNT_LOG)).astype(np.int64)


def encode_count(n) -> np.ndarray:
    return (np.log1p(np.asarray(n, dtype=np.float64)) / COUNT_LOG).astype(np.float32)


def unit_id(name: str) -> int:
    return next(u["id"] for u in units().values() if u["name"] == name)


@functools.cache
def action_id(name: str) -> int:
    return next(a["id"] for a in rows("actions") if a["name"] == name)


@functools.cache
def action_names() -> dict:
    return {a["id"]: a["name"] for a in rows("actions")}


@functools.cache
def order_id(name: str) -> int:
    return next(o["id"] for o in rows("order_types") if o["name"] == name)


RACES = ("T", "P", "Z")
RACE_NAMES = {"T": "Terran", "P": "Protoss", "Z": "Zerg"}
_SPEC = {
    "T": {"focus": "Terran_Barracks", "worker": "Terran_SCV", "town": ["Terran_Command_Center"], "gas": "Terran_Refinery",
          "army": ["Terran_Marine", "Terran_Ghost", "Terran_Vulture", "Terran_Goliath", "Terran_Firebat", "Terran_Medic", "Terran_Siege_Tank_Tank_Mode"]},
    "P": {"focus": "Protoss_Gateway", "worker": "Protoss_Probe", "town": ["Protoss_Nexus"], "gas": "Protoss_Assimilator",
          "army": ["Protoss_Zealot", "Protoss_Dragoon", "Protoss_Dark_Templar", "Protoss_High_Templar"]},
    "Z": {"focus": "Zerg_Spawning_Pool", "worker": "Zerg_Drone", "town": ["Zerg_Hatchery", "Zerg_Lair"], "gas": "Zerg_Extractor",
          "army": ["Zerg_Zergling", "Zerg_Hydralisk", "Zerg_Ultralisk"]},
}
_PRIV_TYPES = {
    "T": ["Terran_SCV", "Terran_Marine", "Terran_Barracks", "Terran_Refinery", "Terran_Factory", "Terran_Command_Center", "Terran_Bunker", "Terran_Academy"],
    "P": ["Protoss_Probe", "Protoss_Zealot", "Protoss_Gateway", "Protoss_Forge", "Protoss_Cybernetics_Core", "Protoss_Nexus", "Protoss_Assimilator", "Protoss_Photon_Cannon"],
    "Z": ["Zerg_Drone", "Zerg_Zergling", "Zerg_Spawning_Pool", "Zerg_Hatchery", "Zerg_Lair", "Zerg_Extractor"],
}


@functools.cache
def race(r: str) -> dict:
    s = _SPEC[r]
    f = units()[unit_id(s["focus"])]
    return {
        "focus": f["id"], "worker": unit_id(s["worker"]), "town": [unit_id(n) for n in s["town"]], "gas": unit_id(s["gas"]),
        "army": [unit_id(n) for n in s["army"]], "max_hp": f["max_hp"], "max_shields": f["max_shields"], "build_time": f["build_time"],
    }


@functools.cache
def priv_types() -> list[int]:
    return sorted({unit_id(n) for r in RACES for n in _PRIV_TYPES[r]})


def own_ids() -> dict:
    return {"rax": unit_id("Terran_Barracks"), "bunker": unit_id("Terran_Bunker"), "marine": unit_id("Terran_Marine")}


def actions() -> dict:
    return {"bunker": action_id("Build_Bunker"), "marine": action_id("Train_Marine"), "rax": action_id("Build_Barracks")}
