//! Synthetic batches and a tiny config for model tests (Flex, 032-R1).
#![allow(dead_code)]

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use uranium_env::actions::{ActionTable, CAPABILITIES, LOC_CLASSES};
use uranium_model::batch::{ActionBatch, ActionColumns, ObsBatch, PrivBatch, TARGET_LOC, TARGET_UNIT};
use uranium_model::config::*;

pub fn columns() -> ActionColumns {
    ActionColumns::from_table(&ActionTable::load().unwrap()).unwrap()
}

pub fn tiny_config(core: CoreKind) -> UraniumNetConfig {
    let input = InputDims::new()
        .with_map_size(32)
        .with_entity_cont(6)
        .with_spatial_channels(3)
        .with_scalar(10)
        .with_z_units(5)
        .with_privileged(7)
        .with_caps(CAPABILITIES.len())
        .with_loc_classes(LOC_CLASSES.len());
    UraniumNetConfig::new()
        .with_input(input)
        .with_entity(EntityConfig::new().with_d_model(32).with_layers(1).with_heads(4).with_ffn(64).with_dropout(0.0).with_unit_type_emb(8).with_order_emb(8).with_last_action_emb(4))
        .with_spatial(SpatialConfig::new().with_scatter_channels(4).with_stem_channels(8).with_stages([8, 8, 16, 16]).with_pool(32))
        .with_scalar(ScalarConfig::new().with_hidden(32).with_race_emb(4).with_map_family_emb(4).with_start_loc_emb(4).with_z_build_emb(8).with_z_units_emb(8).with_z_out(8))
        .with_core(CoreConfig::new().with_kind(core).with_hidden(32).with_gtrxl_heads(4).with_gtrxl_memory(4))
        .with_heads(HeadsConfig::new().with_ar(32).with_key(16).with_loc_ar_channels(4))
        .with_value(ValueConfig::new().with_hidden(32).with_privileged(8))
}

/// xorshift64*: identical streams on every platform.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Host-side copy of the masks, for checking sampled actions.
#[derive(Clone, Debug)]
pub struct HostMasks {
    pub b: usize,
    pub n: usize,
    pub cells: usize,
    pub entity: Vec<bool>,
    pub owner: Vec<i32>,
    pub unit_cap: Vec<bool>,
    pub loc: Vec<bool>,
    pub type_mask: Vec<bool>,
}

impl HostMasks {
    pub fn cap(&self, b: usize, i: usize, k: usize) -> bool {
        self.unit_cap[(b * self.n + i) * CAPABILITIES.len() + k]
    }
    pub fn loc(&self, b: usize, class: usize, cell: usize) -> bool {
        self.loc[(b * LOC_CLASSES.len() + class) * self.cells + cell]
    }
    pub fn sel_ok(&self, cols: &ActionColumns, b: usize, a: usize, i: usize) -> bool {
        self.entity[b * self.n + i] && self.owner[b * self.n + i] == 0 && self.cap(b, i, cols.select_cap[a] as usize)
    }
    pub fn tgt_ok(&self, cols: &ActionColumns, b: usize, a: usize, i: usize) -> bool {
        self.entity[b * self.n + i] && self.cap(b, i, cols.target_cap[a] as usize)
    }
}

/// Random observation batch whose type_mask respects 034-R3 / R5 (an allowed action always has
/// a selectable unit, a target and a location).
pub fn random_obs(cfg: &UraniumNetConfig, cols: &ActionColumns, b: usize, n: usize, seed: u64, device: &Device) -> (ObsBatch, HostMasks) {
    let d = &cfg.input;
    let (k, l, m) = (CAPABILITIES.len(), LOC_CLASSES.len(), d.map_size);
    let cells = m * m;
    let mut r = Rng::new(seed);
    let mut entity = vec![false; b * n];
    let mut cat = vec![0i32; b * n * 4];
    let mut cell = vec![0i32; b * n];
    let mut pos = vec![0f32; b * n * 2];
    let mut owner = vec![0i32; b * n];
    for row in 0..b {
        let real = 1 + r.below(n);
        for i in 0..n {
            let idx = row * n + i;
            entity[idx] = i < real;
            let o = r.below(3) as i32;
            owner[idx] = o;
            cat[idx * 4] = r.below(d.unit_types) as i32;
            cat[idx * 4 + 1] = o;
            cat[idx * 4 + 2] = r.below(d.orders) as i32;
            cat[idx * 4 + 3] = r.below(d.actions) as i32;
            let (x, y) = (r.below(m), r.below(m));
            cell[idx] = (y * m + x) as i32;
            pos[idx * 2] = x as f32 * 32.0 + 16.0;
            pos[idx * 2 + 1] = y as f32 * 32.0 + 16.0;
        }
    }
    let unit_cap: Vec<bool> = (0..b * n * k).map(|_| r.chance(0.4)).collect();
    let loc: Vec<bool> = (0..b * l * cells).map(|_| r.chance(0.05)).collect();
    let mut hm = HostMasks { b, n, cells, entity: entity.clone(), owner, unit_cap: unit_cap.clone(), loc: loc.clone(), type_mask: vec![] };
    let mut type_mask = vec![false; b * cols.len()];
    for row in 0..b {
        for a in 0..cols.len() {
            let mut ok = a == 0 || r.chance(0.7);
            if cols.uses_select[a] == 1 {
                ok &= (0..n).any(|i| hm.sel_ok(cols, row, a, i));
            }
            if cols.target_kind[a] == TARGET_UNIT {
                ok &= (0..n).any(|i| hm.tgt_ok(cols, row, a, i));
            }
            if cols.target_kind[a] == TARGET_LOC {
                ok &= (0..cells).any(|c| hm.loc(row, cols.loc_class[a] as usize, c));
            }
            type_mask[row * cols.len() + a] = ok;
        }
    }
    hm.type_mask = type_mask.clone();

    fn f<const D: usize>(v: Vec<f32>, shape: Vec<usize>, device: &Device) -> Tensor<D> {
        Tensor::from_data(TensorData::new(v, shape), device)
    }
    let fl =|len: usize, r: &mut Rng| (0..len).map(|_| r.unit() * 2.0 - 1.0).collect::<Vec<f32>>();
    let obs = ObsBatch {
        entity_cat: Tensor::<3, Int>::from_data(TensorData::new(cat, [b, n, 4]), device),
        entity_cont: f(fl(b * n * d.entity_cont, &mut r), vec![b, n, d.entity_cont], device),
        entity_mask: Tensor::<2, Bool>::from_data(TensorData::new(entity, [b, n]), device),
        entity_cell: Tensor::<2, Int>::from_data(TensorData::new(cell, [b, n]), device),
        entity_pos: f(pos, vec![b, n, 2], device),
        spatial: f(fl(b * d.spatial_channels * cells, &mut r), vec![b, d.spatial_channels, m, m], device),
        scalar: f(fl(b * d.scalar, &mut r), vec![b, d.scalar], device),
        scalar_cat: Tensor::<2, Int>::from_data(
            TensorData::new((0..b).flat_map(|_| [r.below(d.races) as i32, r.below(d.map_families) as i32, r.below(d.start_locs) as i32]).collect::<Vec<_>>(), [b, 3]),
            device,
        ),
        z_build: Tensor::<2, Int>::from_data(TensorData::new((0..b * d.z_build_len).map(|_| r.below(d.actions) as i32).collect::<Vec<_>>(), [b, d.z_build_len]), device),
        z_units: f(fl(b * d.z_units, &mut r), vec![b, d.z_units], device),
        z_present: Tensor::<1, Bool>::from_data(TensorData::new((0..b).map(|_| r.chance(0.75)).collect::<Vec<_>>(), [b]), device),
        type_mask: Tensor::<2, Bool>::from_data(TensorData::new(type_mask, [b, cols.len()]), device),
        unit_cap: Tensor::<3, Bool>::from_data(TensorData::new(unit_cap, [b, n, k]), device),
        loc_class_mask: Tensor::<4, Bool>::from_data(TensorData::new(loc, [b, l, m, m]), device),
    };
    (obs, hm)
}

pub fn random_priv(cfg: &UraniumNetConfig, b: usize, seed: u64, device: &Device) -> PrivBatch {
    let mut r = Rng::new(seed);
    let p = cfg.input.privileged;
    PrivBatch { summary: Tensor::from_data(TensorData::new((0..b * p).map(|_| r.unit()).collect::<Vec<f32>>(), [b, p]), device) }
}

pub fn ints1(t: &Tensor<1, Int>) -> Vec<i64> {
    t.clone().into_data().convert::<i64>().try_into_vec::<i64>().unwrap()
}

pub fn ints2(t: &Tensor<2, Int>) -> Vec<i64> {
    t.clone().into_data().convert::<i64>().try_into_vec::<i64>().unwrap()
}

pub fn floats<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone().into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
}

/// Host copy of an `ActionBatch`.
pub struct HostActions {
    pub action: Vec<i64>,
    pub selected: Vec<i64>,
    pub target_unit: Vec<i64>,
    pub target_loc: Vec<i64>,
    pub s: usize,
}

pub fn host_actions(a: &ActionBatch) -> HostActions {
    HostActions {
        action: ints1(&a.action),
        selected: ints2(&a.selected),
        target_unit: ints1(&a.target_unit),
        target_loc: ints1(&a.target_loc),
        s: a.selected.dims()[1],
    }
}

/// 032-AC4 / 031-AC2 check of one sampled batch; returns the number of rows checked.
pub fn check_actions(cols: &ActionColumns, hm: &HostMasks, acts: &HostActions) -> usize {
    for row in 0..hm.b {
        let a = acts.action[row] as usize;
        assert!(hm.type_mask[row * cols.len() + a], "row {row}: masked action_type {a}");
        let sel = &acts.selected[row * acts.s..(row + 1) * acts.s];
        let picked: Vec<usize> = sel.iter().take_while(|&&i| i != hm.n as i64).map(|&i| i as usize).collect();
        if cols.uses_select[a] == 1 {
            assert!(!picked.is_empty(), "row {row}: action {a} selected nothing");
            assert!(picked.len() <= cols.max_select[a] as usize, "row {row}: too many selected");
            let mut uniq = picked.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), picked.len(), "row {row}: duplicate selection");
            for &i in &picked {
                assert!(hm.sel_ok(cols, row, a, i), "row {row}: action {a} selected ineligible entity {i}");
            }
        }
        if cols.target_kind[a] == TARGET_UNIT {
            let t = acts.target_unit[row] as usize;
            assert!(hm.tgt_ok(cols, row, a, t), "row {row}: action {a} invalid target unit {t}");
        }
        if cols.target_kind[a] == TARGET_LOC {
            let c = acts.target_loc[row] as usize;
            assert!(hm.loc(row, cols.loc_class[a] as usize, c), "row {row}: action {a} invalid location {c}");
        }
    }
    hm.b
}
