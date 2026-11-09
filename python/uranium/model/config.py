"""`UraniumNetConfig` (032 §6): the same fields and JSON as the Burn config, so checkpoints and
configs carry over (032-R6). Keys missing from a JSON take the defaults (032-R18 ②)."""

import math
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


# 032 §2.8 (032-R21..R24): the model 2 switches of `[model]`, all off by default. Kept out of
# `to_dict` while off, so the checkpoint meta of a model without them (Checkpoint 1) is unchanged.
MODEL2_KEYS = ("bypass", "spatial_preact", "core_skip", "recency_init", "aux_priv", "aux_jepa")
# 041-R23 ⑤ / 042-R21 ③: the logs have three fixed `aux_jepa_slot<i>` columns, one per k.
AUX_JEPA_SLOTS = 3


@dataclass
class VariantConfig:
    """032 §2.7 / 032-R20: structure switches for the architecture comparison (the BC config's
    `[model]` table), and the 032 §2.8 model 2 switches (032-R21..R24). All off = the default
    model, bit for bit (032-AC11, AC13)."""

    terrain: bool = False
    ctx_token: bool = False
    attn_pool: bool = False
    outer: int = 0  # outer LSTM segment length in decisions, 0 = off
    # 032 §2.8 (model 2). Built after every other module (as win_open), fixed seeds (variants.py).
    bypass: bool = False  # 032-R21 head bypass + raw scalars into the ActionType head
    spatial_preact: bool = False  # 032-R22 pre-activation spatial ResBlocks (no new parameters)
    core_skip: bool = False  # 032-R23 F1: zero-initialised Linear(core input) added to h
    recency_init: bool = False  # 032-R23 F2: rel_table init term b(d), [SPEC DEFERRED(U2)] AMB-032-14
    aux_priv: bool = False  # 032-R24 training-only privileged prediction from h_t
    aux_jepa: list = field(default_factory=list)  # 032-R24 JEPA-style predictors, one per k; [] = off

    def arch(self) -> bool:
        """Any 032 §2.7 switch on."""
        return self.terrain or self.ctx_token or self.attn_pool or self.outer > 0

    def model2(self) -> bool:
        """Any 032 §2.8 switch on."""
        return self.bypass or self.spatial_preact or self.core_skip or self.recency_init or self.aux_priv or bool(self.aux_jepa)

    def any(self) -> bool:
        return self.arch() or self.model2()

    def aux(self) -> bool:
        """032-R24: a training-only aux module is built (`forward_seq(aux=True)` returns its
        predictions; never part of `act`)."""
        return self.aux_priv or bool(self.aux_jepa)

    def tokens(self) -> int:
        """Extra tokens of the entity transformer (`ctx_token`, `outer`); the 032 §2.8 switches
        add none (the bypass reads E without its extra tokens)."""
        return int(self.ctx_token) + int(self.outer > 0)

    def to_table(self) -> dict:
        """As in the checkpoint meta: the 032 §2.7 keys always, a 032 §2.8 key only when on."""
        d = asdict(self)
        for k in MODEL2_KEYS:
            if not d[k]:
                del d[k]
        return d

    def check(self) -> "VariantConfig":
        """Value checks shared by the `[model]` table and a checkpoint's config."""
        if self.outer < 0:
            raise ValueError(f"[model] outer = {self.outer}: expected 0 (off) or a segment length")
        ks = self.aux_jepa
        if not isinstance(ks, list) or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError(f"[model] aux_jepa = {ks!r}: expected a list of step offsets k >= 1 (032-R24)")
        if len(set(ks)) != len(ks):
            raise ValueError(f"[model] aux_jepa = {ks!r}: a k appears twice (032-R24)")
        if len(ks) > AUX_JEPA_SLOTS:
            raise ValueError(f"[model] aux_jepa = {ks!r}: at most {AUX_JEPA_SLOTS} k (the logs' aux_jepa_slot1..{AUX_JEPA_SLOTS}, 041-R23 ⑤)")
        if self.recency_init:
            # 032-R23: the F2 term b(d) waits for the core attention study (U2); no code may
            # depend on it before AMB-032-14 is resolved.
            raise ValueError("[model] recency_init: the F2 term b(d) is [SPEC DEFERRED(U2)] AMB-032-14, not available yet (032-R23)")
        return self

    @classmethod
    def from_table(cls, table: dict | None) -> "VariantConfig":
        """The `[model]` table of a BC config; an unknown key or a wrong type is an error
        (032-AC11, AC13), so a typo cannot train the default model under a variant's run id."""
        table = table or {}
        names = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(table) - set(names))
        if unknown:
            raise ValueError(f"[model]: unknown keys {unknown} (known: {sorted(names)})")
        for k, v in table.items():
            if names[k] in (list, "list"):
                # 032 §2.8: `aux_jepa` is the one list value (integers, checked in `check`).
                if type(v) is not list:
                    raise ValueError(f"[model] {k} = {v!r}: expected a list of integers")
                continue
            want = bool if names[k] in (bool, "bool") else int
            if type(v) is not want:
                raise ValueError(f"[model] {k} = {v!r}: expected {want.__name__}")
        return cls(**{k: list(v) if isinstance(v, list) else v for k, v in table.items()}).check()


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
        else:
            # 032 §2.8: the model 2 keys only when on (a §2.7 model's meta stays as it was).
            d["variant"] = self.variant.to_table()
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


# 041-R23 ① / 042-R21 ①: `[aux]` key -> the `[model]` module whose loss weight it is.
AUX_KEYS = {"priv": "aux_priv", "jepa": "aux_jepa"}


def check_aux(variant: VariantConfig, aux: dict | None) -> dict:
    """041-R23 ④ / 042-R21 ①: an `[aux]` key without its `[model]` module, or a module without
    its key, is a start error (as a [model] mismatch, 032-AC11). `aux` is the training config's
    `[aux]` table as the Rust config passes it (absent keys left out). Returns {key: weight} for
    the modules the model has (weight 0 = that loss is not added, 041-R23 ④)."""
    aux = {k: v for k, v in (aux or {}).items() if v is not None}
    unknown = sorted(set(aux) - set(AUX_KEYS))
    if unknown:
        raise ValueError(f"[aux]: unknown keys {unknown} (known: {sorted(AUX_KEYS)})")
    for key, module in AUX_KEYS.items():
        has = bool(getattr(variant, module))
        if key in aux and not has:
            raise ValueError(f"[aux] {key} is set but the model has no {module} module ([model] {module}, 041-R23 ④)")
        if has and key not in aux:
            raise ValueError(f"the model has the {module} module but [aux] {key} is not set (041-R23 ④)")
        w = aux.get(key)
        if w is not None and not (type(w) in (int, float) and w >= 0 and math.isfinite(w)):
            raise ValueError(f"[aux] {key} = {w!r}: expected a finite weight >= 0")
    return {k: float(v) for k, v in aux.items()}


def input_dims() -> InputDims:
    """Model input dimensions from the feature schema (030) and the action table (034)."""
    import json

    from uranium import _core

    d = json.loads(_core.feature_dims())
    return InputDims(
        actions=d["actions"], entity_cont=d["entity_cont"], spatial_channels=d["spatial"], map_size=d["map"], scalar=d["scalar"],
        z_units=d["z_units"], caps=d["caps"], loc_classes=d["loc_classes"], privileged=d["privileged"],
    )
