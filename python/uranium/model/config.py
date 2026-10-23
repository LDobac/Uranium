"""`UraniumNetConfig` (032 §6): the same fields and JSON as the Burn config, so checkpoints and
configs carry over (032-R6). Keys missing from a JSON take the defaults (032-R18 ②)."""

from dataclasses import asdict, dataclass, field, fields


def _load(cls, d: dict | None):
    d = d or {}
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        sub = _SUB.get((cls, f.name))
        kw[f.name] = _load(sub, v) if sub else v
    return cls(**kw)


@dataclass
class InputDims:
    unit_types: int = 234
    owners: int = 3
    orders: int = 190
    actions: int = 101
    entity_cont: int = 76
    spatial_channels: int = 26
    map_size: int = 128
    scalar: int = 512
    races: int = 4
    map_families: int = 3
    start_locs: int = 8
    z_build_len: int = 20
    z_units: int = 34
    caps: int = 46
    loc_classes: int = 8
    privileged: int = 256


@dataclass
class EntityConfig:
    d_model: int = 256
    layers: int = 3
    heads: int = 8
    ffn: int = 1024
    dropout: float = 0.1
    unit_type_emb: int = 64
    owner_emb: int = 8
    order_emb: int = 32
    last_action_emb: int = 16
    distance_buckets: int = 16


@dataclass
class SpatialConfig:
    scatter_channels: int = 32
    stem_channels: int = 32
    stages: list = field(default_factory=lambda: [64, 64, 128, 128])
    pool: int = 256


@dataclass
class ScalarConfig:
    hidden: int = 256
    race_emb: int = 32
    map_family_emb: int = 16
    start_loc_emb: int = 8
    z_build_emb: int = 32
    z_units_emb: int = 32
    z_out: int = 64


@dataclass
class CoreConfig:
    kind: str = "Lstm"
    hidden: int = 512
    gtrxl_layers: int = 2
    gtrxl_heads: int = 8
    gtrxl_memory: int = 64


@dataclass
class HeadsConfig:
    ar: int = 256
    key: int = 64
    max_select: int = 12
    delay_buckets: int = 16
    loc_ar_channels: int = 16


@dataclass
class ValueConfig:
    hidden: int = 256
    privileged: int = 128
    win_open: bool = False


@dataclass
class UraniumNetConfig:
    input: InputDims = field(default_factory=InputDims)
    entity: EntityConfig = field(default_factory=EntityConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    scalar: ScalarConfig = field(default_factory=ScalarConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    value: ValueConfig = field(default_factory=ValueConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "UraniumNetConfig":
        return _load(cls, d)

    def to_dict(self) -> dict:
        return asdict(self)


_SUB = {
    (UraniumNetConfig, "input"): InputDims,
    (UraniumNetConfig, "entity"): EntityConfig,
    (UraniumNetConfig, "spatial"): SpatialConfig,
    (UraniumNetConfig, "scalar"): ScalarConfig,
    (UraniumNetConfig, "core"): CoreConfig,
    (UraniumNetConfig, "heads"): HeadsConfig,
    (UraniumNetConfig, "value"): ValueConfig,
}


def input_dims() -> InputDims:
    """Model input dimensions from the feature schema (030) and the action table (034)."""
    import json

    from uranium import _core

    d = json.loads(_core.feature_dims())
    return InputDims(
        actions=d["actions"], entity_cont=d["entity_cont"], spatial_channels=d["spatial"], map_size=d["map"], scalar=d["scalar"],
        z_units=d["z_units"], caps=d["caps"], loc_classes=d["loc_classes"], privileged=d["privileged"],
    )
