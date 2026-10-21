//! `UraniumNetConfig` (032 §6). Defaults are the 032 §4 base size (~12M parameters, 032-R2);
//! every dimension is configurable so tests can build tiny models on Flex.

use burn::config::Config;

/// Input dimensions fixed by the feature / action schemas (030 §1, 034).
#[derive(Config, Debug, PartialEq)]
pub struct InputDims {
    /// 030: unit_type vocabulary.
    #[config(default = 234)]
    pub unit_types: usize,
    /// 030: owner self / enemy / neutral.
    #[config(default = 3)]
    pub owners: usize,
    /// 030: BWAPI order vocabulary.
    #[config(default = 190)]
    pub orders: usize,
    /// 034: number of action types (A). 101 since action schema 2 (`Unload_Unit`, 2026-10-19).
    #[config(default = 101)]
    pub actions: usize,
    /// 030 §2: `entity_cont` width (F_c).
    #[config(default = 76)]
    pub entity_cont: usize,
    /// 030 §3: spatial channels (C).
    #[config(default = 26)]
    pub spatial_channels: usize,
    /// 030-R6: spatial side in build tiles.
    #[config(default = 128)]
    pub map_size: usize,
    /// 030 §4: `scalar` width (F_s).
    #[config(default = 512)]
    pub scalar: usize,
    /// 030: enemy race vocabulary (T / P / Z / unknown).
    #[config(default = 4)]
    pub races: usize,
    /// 040-R15: map families (fighting_spirit, circuit_breaker, other).
    #[config(default = 3)]
    pub map_families: usize,
    /// 030: start location vocabulary.
    #[config(default = 8)]
    pub start_locs: usize,
    /// 030 §5: z build order length.
    #[config(default = 20)]
    pub z_build_len: usize,
    /// 030 §5: z unit composition width (U).
    #[config(default = 34)]
    pub z_units: usize,
    /// 034: capability classes (K).
    #[config(default = 46)]
    pub caps: usize,
    /// 034: location classes (L).
    #[config(default = 8)]
    pub loc_classes: usize,
    /// 030-R9: `PrivilegedTensors` width.
    #[config(default = 256)]
    pub privileged: usize,
}

/// 032 §2.1.
#[derive(Config, Debug, PartialEq)]
pub struct EntityConfig {
    #[config(default = 256)]
    pub d_model: usize,
    #[config(default = 3)]
    pub layers: usize,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 1024)]
    pub ffn: usize,
    /// 0.1 for BC, 0 for RL.
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = 64)]
    pub unit_type_emb: usize,
    #[config(default = 8)]
    pub owner_emb: usize,
    #[config(default = 32)]
    pub order_emb: usize,
    #[config(default = 16)]
    pub last_action_emb: usize,
    /// Pixel distance log buckets; total buckets = 2 (same owner) x this.
    #[config(default = 16)]
    pub distance_buckets: usize,
}

/// 032 §2.2.
#[derive(Config, Debug, PartialEq)]
pub struct SpatialConfig {
    #[config(default = 32)]
    pub scatter_channels: usize,
    #[config(default = 32)]
    pub stem_channels: usize,
    /// Channels after each stride-2 stage (128² -> 64² -> 32² -> 16² -> 8²).
    #[config(default = "[64, 64, 128, 128]")]
    pub stages: [usize; 4],
    #[config(default = 256)]
    pub pool: usize,
}

/// 032 §2.3.
#[derive(Config, Debug, PartialEq)]
pub struct ScalarConfig {
    #[config(default = 256)]
    pub hidden: usize,
    #[config(default = 32)]
    pub race_emb: usize,
    #[config(default = 16)]
    pub map_family_emb: usize,
    #[config(default = 8)]
    pub start_loc_emb: usize,
    #[config(default = 32)]
    pub z_build_emb: usize,
    #[config(default = 32)]
    pub z_units_emb: usize,
    #[config(default = 64)]
    pub z_out: usize,
}

#[derive(Config, Debug, PartialEq, Copy)]
pub enum CoreKind {
    Lstm,
    Gtrxl,
}

/// 032 §2.4.
#[derive(Config, Debug, PartialEq)]
pub struct CoreConfig {
    #[config(default = "CoreKind::Lstm")]
    pub kind: CoreKind,
    #[config(default = 512)]
    pub hidden: usize,
    /// GTrXL: attention layers, heads and memory length M.
    #[config(default = 2)]
    pub gtrxl_layers: usize,
    #[config(default = 8)]
    pub gtrxl_heads: usize,
    #[config(default = 64)]
    pub gtrxl_memory: usize,
}

/// 032 §2.5.
#[derive(Config, Debug, PartialEq)]
pub struct HeadsConfig {
    #[config(default = 256)]
    pub ar: usize,
    #[config(default = 64)]
    pub key: usize,
    /// 031-R3: selection limit.
    #[config(default = 12)]
    pub max_select: usize,
    /// 031: delay buckets.
    #[config(default = 16)]
    pub delay_buckets: usize,
    /// Location head: channels of the ar map concatenated to S8.
    #[config(default = 16)]
    pub loc_ar_channels: usize,
}

/// 032 §2.6.
#[derive(Config, Debug, PartialEq)]
pub struct ValueConfig {
    #[config(default = 256)]
    pub hidden: usize,
    #[config(default = 128)]
    pub privileged: usize,
    /// 032-R18: build the `win_open` head (no privileged input, 031-R12 surrender). Off in
    /// configs that predate it, so their checkpoints load unchanged.
    #[config(default = false)]
    pub win_open: bool,
}

#[derive(Config, Debug, PartialEq)]
pub struct UraniumNetConfig {
    #[config(default = "InputDims::new()")]
    pub input: InputDims,
    #[config(default = "EntityConfig::new()")]
    pub entity: EntityConfig,
    #[config(default = "SpatialConfig::new()")]
    pub spatial: SpatialConfig,
    #[config(default = "ScalarConfig::new()")]
    pub scalar: ScalarConfig,
    #[config(default = "CoreConfig::new()")]
    pub core: CoreConfig,
    #[config(default = "HeadsConfig::new()")]
    pub heads: HeadsConfig,
    #[config(default = "ValueConfig::new()")]
    pub value: ValueConfig,
}
