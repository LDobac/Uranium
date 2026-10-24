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
class VariantConfig:
    """032 §2.7 / 032-R20: structure switches for the architecture comparison (the BC config's
    `[model]` table). All off = the default model, bit for bit."""

    terrain: bool = False
    ctx_token: bool = False
    attn_pool: bool = False
    outer: int = 0  # outer LSTM segment length in decisions, 0 = off

    def any(self) -> bool:
        return self.terrain or self.ctx_token or self.attn_pool or self.outer > 0

    def tokens(self) -> int:
        """Extra tokens of the entity transformer (`ctx_token`, `outer`)."""
        return int(self.ctx_token) + int(self.outer > 0)

    @classmethod
    def from_table(cls, table: dict | None) -> "VariantConfig":
        """The `[model]` table of a BC config; an unknown key or a wrong type is an error
        (032-AC11), so a typo cannot train the default model under a variant's run id."""
        table = table or {}
        names = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(table) - set(names))
        if unknown:
            raise ValueError(f"[model]: unknown keys {unknown} (known: {sorted(names)})")
        for k, v in table.items():
            want = bool if names[k] in (bool, "bool") else int
            if type(v) is not want:
                raise ValueError(f"[model] {k} = {v!r}: expected {want.__name__}")
        v = cls(**table)
        if v.outer < 0:
            raise ValueError(f"[model] outer = {v.outer}: expected 0 (off) or a segment length")
        return v


@dataclass
class UraniumNetConfig:
    input: InputDims = field(default_factory=InputDims)
    entity: EntityConfig = field(default_factory=EntityConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    scalar: ScalarConfig = field(default_factory=ScalarConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    variant: VariantConfig = field(default_factory=VariantConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "UraniumNetConfig":
        return _load(cls, d)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.variant.any():
            # 032 §2.7: the default model's config (checkpoint meta) stays as it was.
            del d["variant"]
        return d


_SUB = {
    (UraniumNetConfig, "input"): InputDims,
    (UraniumNetConfig, "entity"): EntityConfig,
    (UraniumNetConfig, "spatial"): SpatialConfig,
    (UraniumNetConfig, "scalar"): ScalarConfig,
    (UraniumNetConfig, "core"): CoreConfig,
    (UraniumNetConfig, "heads"): HeadsConfig,
    (UraniumNetConfig, "value"): ValueConfig,
    (UraniumNetConfig, "variant"): VariantConfig,
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
