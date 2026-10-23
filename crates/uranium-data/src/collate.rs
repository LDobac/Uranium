//! Host collate (030 §1, 032 §6): observations, labels and privileged inputs of a batch packed
//! into flat row-major buffers with the shapes the model takes. Entities are padded to the largest
//! N in the batch, rounded up to a multiple of `pad` (030-R7, 032-R7); padding rows are masked.
//! The trainer and the inference server build device tensors from these buffers without another
//! copy of the layout logic (AMB-000-4).

use uranium_env::features::label::HostLabel;
use uranium_env::features::{CELLS, ENTITY_CAT, ENTITY_CONT, MAP, Observation, PrivilegedTensors, SPATIAL, Z_BUILD};

/// One batch of observations (`ObsBatch` on the host). Shapes in the field docs; `b` rows, `n`
/// padded entities.
#[derive(Debug, Clone, PartialEq)]
pub struct HostObsBatch {
    pub b: usize,
    pub n: usize,
    /// Capability classes (034).
    pub k: usize,
    /// Action types (034).
    pub a: usize,
    /// Location classes (034).
    pub l: usize,
    /// `scalar` width (030 §4).
    pub fs: usize,
    /// `z_units` width (030 §5).
    pub u: usize,
    /// [b, n, ENTITY_CAT]
    pub entity_cat: Vec<i32>,
    /// [b, n, ENTITY_CONT]
    pub entity_cont: Vec<f32>,
    /// [b, n], true for real entities.
    pub entity_mask: Vec<bool>,
    /// [b, n]
    pub entity_cell: Vec<i32>,
    /// [b, n, 2]
    pub entity_pos: Vec<f32>,
    /// [b, SPATIAL, MAP, MAP]
    pub spatial: Vec<f32>,
    /// [b, fs]
    pub scalar: Vec<f32>,
    /// [b, 3]
    pub scalar_cat: Vec<i32>,
    /// [b, Z_BUILD]
    pub z_build: Vec<i32>,
    /// [b, u]
    pub z_units: Vec<f32>,
    /// [b]
    pub z_present: Vec<bool>,
    /// [b, a]
    pub type_mask: Vec<bool>,
    /// [b, n, k]
    pub unit_cap: Vec<bool>,
    /// [b, l, MAP, MAP]
    pub loc_class_mask: Vec<bool>,
    /// [b, n]
    pub is_building: Vec<bool>,
}

/// Action labels of a batch (`ActionBatch` on the host). END = `n` (032-R12).
#[derive(Debug, Clone, PartialEq)]
pub struct HostActionBatch {
    pub b: usize,
    /// Selection steps (`max_select`).
    pub s: usize,
    /// [b]
    pub action: Vec<i32>,
    /// [b]
    pub delay: Vec<i32>,
    /// [b]
    pub queued: Vec<i32>,
    /// [b, s], END-filled after the selection.
    pub selected: Vec<i32>,
    /// [b]
    pub target_unit: Vec<i32>,
    /// [b]
    pub target_loc: Vec<i32>,
}

/// Privileged inputs of a batch (`PrivBatch` on the host, 030-R9).
#[derive(Debug, Clone, PartialEq)]
pub struct HostPrivBatch {
    pub b: usize,
    pub p: usize,
    /// [b, p]
    pub summary: Vec<f32>,
}

/// Packs `rows` with N rounded up to a multiple of `pad` (0 or 1 = the batch max N only).
pub fn obs_batch(rows: &[&Observation], pad: usize) -> HostObsBatch {
    let b = rows.len();
    let n = rows.iter().map(|o| o.obs.n).max().unwrap_or(0).max(1).next_multiple_of(pad.max(1));
    let k = rows.first().map_or(0, |o| o.masks.unit_cap.len() / o.obs.n.max(1));
    let k = if k == 0 { uranium_env::actions::CAPABILITIES.len() } else { k };
    let a = rows.first().map_or(0, |o| o.masks.type_mask.len());
    let l = rows.first().map_or(0, |o| o.masks.loc_class_mask.len() / CELLS);
    let fs = rows.first().map_or(0, |o| o.obs.scalar.len());
    let u = rows.first().map_or(0, |o| o.obs.z_units.len());

    let mut x = HostObsBatch {
        b,
        n,
        k,
        a,
        l,
        fs,
        u,
        entity_cat: vec![0i32; b * n * ENTITY_CAT],
        entity_cont: vec![0f32; b * n * ENTITY_CONT],
        entity_mask: vec![false; b * n],
        entity_cell: vec![0i32; b * n],
        entity_pos: vec![0f32; b * n * 2],
        spatial: Vec::with_capacity(b * SPATIAL * CELLS),
        scalar: Vec::with_capacity(b * fs),
        scalar_cat: Vec::with_capacity(b * 3),
        z_build: Vec::with_capacity(b * Z_BUILD),
        z_units: Vec::with_capacity(b * u),
        z_present: Vec::with_capacity(b),
        type_mask: Vec::with_capacity(b * a),
        unit_cap: vec![false; b * n * k],
        loc_class_mask: Vec::with_capacity(b * l * MAP * MAP),
        is_building: vec![false; b * n],
    };
    for (r, o) in rows.iter().enumerate() {
        let m = o.obs.n;
        let base = r * n;
        x.entity_cat[base * ENTITY_CAT..(base + m) * ENTITY_CAT].copy_from_slice(&o.obs.entity_cat);
        x.entity_cont[base * ENTITY_CONT..(base + m) * ENTITY_CONT].copy_from_slice(&o.obs.entity_cont);
        x.entity_mask[base..base + m].fill(true);
        x.entity_cell[base..base + m].copy_from_slice(&o.obs.entity_cell);
        x.entity_pos[base * 2..(base + m) * 2].copy_from_slice(&o.obs.entity_pos);
        x.unit_cap[base * k..(base + m) * k].copy_from_slice(&o.masks.unit_cap);
        x.is_building[base..base + m].copy_from_slice(&o.masks.is_building);
        x.spatial.extend_from_slice(&o.obs.spatial);
        x.scalar.extend_from_slice(&o.obs.scalar);
        x.scalar_cat.extend_from_slice(&o.obs.scalar_cat);
        x.z_build.extend_from_slice(&o.obs.z_build);
        x.z_units.extend_from_slice(&o.obs.z_units);
        x.z_present.push(o.obs.z_present);
        x.type_mask.extend_from_slice(&o.masks.type_mask);
        x.loc_class_mask.extend_from_slice(&o.masks.loc_class_mask);
    }
    x
}

/// `n` is the padded entity count of the matching [`HostObsBatch`]; END = n (032-R12).
pub fn action_batch(labels: &[&HostLabel], n: usize, max_select: usize) -> HostActionBatch {
    let b = labels.len();
    let mut selected = vec![n as i32; b * max_select];
    for (r, l) in labels.iter().enumerate() {
        for (s, &e) in l.selected.iter().take(max_select).enumerate() {
            selected[r * max_select + s] = e as i32;
        }
    }
    HostActionBatch {
        b,
        s: max_select,
        action: labels.iter().map(|l| l.action as i32).collect(),
        delay: labels.iter().map(|l| l.delay_bucket as i32).collect(),
        queued: labels.iter().map(|l| l.queued as i32).collect(),
        selected,
        target_unit: labels.iter().map(|l| l.target_unit.unwrap_or(0) as i32).collect(),
        target_loc: labels.iter().map(|l| l.target_loc.unwrap_or(0) as i32).collect(),
    }
}

pub fn priv_batch(rows: &[&PrivilegedTensors]) -> HostPrivBatch {
    let p = rows.first().map_or(0, |r| r.summary.len());
    HostPrivBatch { b: rows.len(), p, summary: rows.iter().flat_map(|r| r.summary.iter().copied()).collect() }
}
