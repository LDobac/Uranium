import math
from dataclasses import asdict, dataclass, field, fields, replace


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
    patch_dim: int = 256


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


MODEL2_KEYS = ("bypass", "spatial_preact", "core_skip", "aux_priv", "aux_jepa", "spatial_norm", "spatial_end_relu")
OFF_KEYS = (*MODEL2_KEYS, "spatial_patch")
CNN_KEYS = ("spatial_preact", "spatial_norm", "spatial_end_relu")
AUX_JEPA_SLOTS = 3


@dataclass
class VariantConfig:
    terrain: bool = False
    ctx_token: bool = False
    attn_pool: bool = False
    outer: int = 0
    bypass: bool = False
    spatial_preact: bool = False
    core_skip: bool = False
    aux_priv: bool = False
    aux_jepa: list = field(default_factory=list)
    spatial_norm: int = 0
    spatial_end_relu: bool = False
    spatial_patch: int = 0

    def arch(self) -> bool:
        return self.terrain or self.ctx_token or self.attn_pool or self.outer > 0

    def model2(self) -> bool:
        return self.bypass or self.spatial_preact or self.core_skip or self.aux_priv or bool(self.aux_jepa) or self.spatial_norm != 0 or self.spatial_end_relu

    def any(self) -> bool:
        return self.arch() or self.model2() or self.spatial_patch != 0

    def aux(self) -> bool:
        return self.aux_priv or bool(self.aux_jepa)

    def tokens(self) -> int:
        return int(self.ctx_token) + int(self.outer > 0)

    def to_table(self) -> dict:
        d = asdict(self)
        for k in OFF_KEYS:
            if not d[k]:
                del d[k]
        return d

    def check(self) -> "VariantConfig":
        if self.outer < 0:
            raise ValueError(f"[model] outer = {self.outer}: expected 0 (off) or a segment length")
        ks = self.aux_jepa
        if not isinstance(ks, list) or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError(f"[model] aux_jepa = {ks!r}: expected a list of step offsets k >= 1 (032-R24)")
        if len(set(ks)) != len(ks):
            raise ValueError(f"[model] aux_jepa = {ks!r}: a k appears twice (032-R24)")
        if len(ks) > AUX_JEPA_SLOTS:
            raise ValueError(f"[model] aux_jepa = {ks!r}: at most {AUX_JEPA_SLOTS} k (the logs' aux_jepa_slot1..{AUX_JEPA_SLOTS}, 041-R23 ⑤)")
        if self.spatial_norm < 0:
            raise ValueError(f"[model] spatial_norm = {self.spatial_norm}: expected 0 (off) or the channels per group (032-R26)")
        if self.spatial_norm and not self.spatial_preact:
            raise ValueError(f"[model] spatial_norm = {self.spatial_norm} needs spatial_preact = true (032-R26)")
        if self.spatial_end_relu and not self.spatial_norm:
            raise ValueError("[model] spatial_end_relu = true needs spatial_norm (032-R27)")
        if self.spatial_patch < 0:
            raise ValueError(f"[model] spatial_patch = {self.spatial_patch}: expected 0 (CNN) or the patch side p (032-R29)")
        if self.spatial_patch:
            on = [k for k in CNN_KEYS if getattr(self, k)]
            if on:
                raise ValueError(f"[model] spatial_patch = {self.spatial_patch} with the CNN-only switches {on} (032-R29 ⑤)")
        return self

    def grafts_onto(self, base: "VariantConfig") -> bool:
        if self == base or self.spatial_norm <= 0 or base.spatial_norm not in (0, self.spatial_norm) or (base.spatial_end_relu and not self.spatial_end_relu):
            return False
        return replace(self, spatial_norm=base.spatial_norm, spatial_end_relu=base.spatial_end_relu) == base

    @classmethod
    def from_table(cls, table: dict | None) -> "VariantConfig":
        table = table or {}
        names = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(table) - set(names))
        if unknown:
            raise ValueError(f"[model]: unknown keys {unknown} (known: {sorted(names)})")
        for k, v in table.items():
            if names[k] in (list, "list"):
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

    def patch(self) -> int:
        return self.variant.spatial_patch

    def core_in_width(self) -> int:
        return self.entity.d_model + (0 if self.patch() else self.spatial.pool) + self.scalar.hidden

    def jepa_width(self) -> int:
        return self.entity.d_model + (0 if self.patch() else self.spatial.pool)

    def check_patch(self) -> "UraniumNetConfig":
        p, side = self.patch(), self.input.map_size
        if p and side % p:
            raise ValueError(f"[model] spatial_patch = {p} does not divide the map side {side} (032-R29 ②)")
        if p and self.spatial.patch_dim <= 0:
            raise ValueError(f"[net] spatial.patch_dim = {self.spatial.patch_dim}: expected a width > 0 (032-R28)")
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        if not self.patch():
            del d["spatial"]["patch_dim"]
        if not self.variant.any():
            del d["variant"]
        else:
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


AUX_KEYS = {"priv": "aux_priv", "jepa": "aux_jepa"}
AUX_EPS_KEY = "adam_eps"


def check_aux(variant: VariantConfig, aux: dict | None) -> dict:
    aux = {k: v for k, v in (aux or {}).items() if v is not None and k != AUX_EPS_KEY}
    unknown = sorted(set(aux) - set(AUX_KEYS))
    if unknown:
        raise ValueError(f"[aux]: unknown keys {unknown} (known: {sorted([*AUX_KEYS, AUX_EPS_KEY])})")
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


def aux_adam_eps(variant: VariantConfig, aux: dict | None) -> float | None:
    eps = (aux or {}).get(AUX_EPS_KEY)
    if eps is None:
        return None
    if not (type(eps) in (int, float) and eps > 0 and math.isfinite(eps)):
        raise ValueError(f"[aux] {AUX_EPS_KEY} = {eps!r}: expected a finite value > 0 (041-R23 ⑦)")
    if not check_aux(variant, aux):
        raise ValueError(f"[aux] {AUX_EPS_KEY} is set without priv or jepa (041-R23 ⑦)")
    return float(eps)


def input_dims() -> InputDims:
    import json

    from uranium import _core

    d = json.loads(_core.feature_dims())
    return InputDims(
        actions=d["actions"], entity_cont=d["entity_cont"], spatial_channels=d["spatial"], map_size=d["map"], scalar=d["scalar"],
        z_units=d["z_units"], caps=d["caps"], loc_classes=d["loc_classes"], privileged=d["privileged"],
    )


CORES = {"lstm": "Lstm", "gtrxl": "Gtrxl"}


NET_TABLES = {"entity": EntityConfig, "spatial": SpatialConfig, "scalar": ScalarConfig, "core": CoreConfig, "heads": HeadsConfig, "value": ValueConfig}
NET_REFUSED = {"input": "observation sizes come from the feature schema (030)", "core.kind": "use --core", "value.win_open": "use [value] win_open"}
NET_CNN_KEYS = ("spatial.stem_channels", "spatial.stages", "spatial.pool", "heads.loc_ar_channels")
NET_PATCH_KEYS = ("spatial.patch_dim",)
NET_NOT_SIZE = ("entity.dropout",)
CNN_STAGES = 4


def _net_value(name: str, want, v):
    if want in (float, "float"):
        if type(v) is not float or not math.isfinite(v) or not 0.0 <= v < 1.0:
            raise ValueError(f"[net] {name} = {v!r}: expected a float in [0, 1) (032-R28 ②)")
        return v
    if want in (list, "list"):
        if type(v) is not list or len(v) != CNN_STAGES or any(type(x) is not int or x <= 0 for x in v):
            raise ValueError(f"[net] {name} = {v!r}: expected a list of {CNN_STAGES} integers > 0 (032-R28 ②)")
        return list(v)
    if type(v) is not int or v <= 0:
        raise ValueError(f"[net] {name} = {v!r}: expected an integer > 0 (032-R28 ②)")
    return v


def net_sizes_from_table(table: dict | None) -> dict:
    table = table or {}
    if not isinstance(table, dict):
        raise ValueError(f"[net] = {table!r}: expected a table of sub-tables (032-R28)")
    out = {}
    for sec, v in table.items():
        if sec in NET_REFUSED:
            raise ValueError(f"[net] {sec}: not a [net] key, {NET_REFUSED[sec]} (032-R28 ②)")
        cls = NET_TABLES.get(sec)
        if cls is None:
            raise ValueError(f"[net]: unknown table {sec!r} (known: {sorted(NET_TABLES)})")
        if not isinstance(v, dict):
            raise ValueError(f"[net] {sec} = {v!r}: expected a table (032-R28 ②)")
        names = {f.name: f.type for f in fields(cls)}
        sub = {}
        for k, x in v.items():
            name = f"{sec}.{k}"
            if name in NET_REFUSED:
                raise ValueError(f"[net] {name}: not a [net] key, {NET_REFUSED[name]} (032-R28 ②)")
            if k not in names:
                raise ValueError(f"[net] {sec}: unknown key {k!r} (known: {sorted(n for n in names if f'{sec}.{n}' not in NET_REFUSED)})")
            sub[k] = _net_value(name, names[k], x)
        out[sec] = sub
    return out


def net_keys(sizes: dict | None) -> list[str]:
    return [f"{sec}.{k}" for sec, sub in (sizes or {}).items() for k in sub]


def check_net(variant: VariantConfig, sizes: dict | None) -> None:
    keys = net_keys(sizes)
    if variant.spatial_patch:
        bad = [k for k in keys if k in NET_CNN_KEYS]
        if bad:
            raise ValueError(f"[net] {bad}: CNN-only sizes with [model] spatial_patch = {variant.spatial_patch} (032-R29 ⑤)")
    else:
        bad = [k for k in keys if k in NET_PATCH_KEYS]
        if bad:
            raise ValueError(f"[net] {bad}: patch-path sizes without [model] spatial_patch (032-R29)")


def apply_net(cfg: UraniumNetConfig, sizes: dict | None) -> UraniumNetConfig:
    for sec, sub in (sizes or {}).items():
        part = getattr(cfg, sec)
        for k, v in sub.items():
            setattr(part, k, list(v) if isinstance(v, list) else v)
    return cfg


def net_size_table(cfg: UraniumNetConfig) -> dict:
    d = {sec: asdict(getattr(cfg, sec)) for sec in NET_TABLES}
    for name in NET_REFUSED:
        if "." in name:
            sec, k = name.split(".")
            del d[sec][k]
    return d


def net_sizes_of(cfg: UraniumNetConfig) -> dict:
    d = net_size_table(cfg)
    for name in (*NET_NOT_SIZE, *(NET_CNN_KEYS if cfg.patch() else NET_PATCH_KEYS)):
        sec, k = name.split(".")
        del d[sec][k]
    return d


def size_mismatch(have: UraniumNetConfig, want: UraniumNetConfig, sizes: dict | None = None) -> list[str]:
    a, b = net_size_table(have), net_size_table(want)
    skip = set(NET_NOT_SIZE) - set(net_keys(sizes))
    return [f"{sec}.{k}: checkpoint {a[sec][k]!r} != requested {b[sec][k]!r}" for sec in NET_TABLES for k in a[sec] if a[sec][k] != b[sec][k] and f"{sec}.{k}" not in skip]


def new_net_config(core: str, win_open: bool, variant: VariantConfig | None = None, sizes: dict | None = None) -> UraniumNetConfig:
    cfg = UraniumNetConfig()
    cfg.input = input_dims()
    if variant is not None:
        cfg.variant = variant
    check_net(cfg.variant, sizes)
    apply_net(cfg, sizes)
    cfg.core.kind = CORES[core]
    cfg.value.win_open = win_open
    return cfg.check_patch()


def _table(config: str) -> dict:
    import tomllib

    with open(config, "rb") as f:
        return tomllib.load(f)


def model_variant(config: str) -> VariantConfig:
    return VariantConfig.from_table(_table(config).get("model"))


def model_net(config: str) -> tuple[VariantConfig, dict]:
    t = _table(config)
    variant = VariantConfig.from_table(t.get("model"))
    sizes = net_sizes_from_table(t.get("net"))
    check_net(variant, sizes)
    return variant, sizes
