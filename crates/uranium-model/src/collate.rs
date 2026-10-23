//! Host features (030, `uranium_env::features`) -> device batches (032 §6). Entities are padded
//! to the largest N in the batch (030-R7, 032-R7).

use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use uranium_env::features::{CELLS, ENTITY_CAT, ENTITY_CONT, MAP, Observation, PrivilegedTensors, SPATIAL, Z_BUILD};

use crate::batch::{ActionBatch, ObsBatch, PrivBatch};

/// Host form of one action label: defined next to the label check in `uranium-env` (040-R19).
pub use uranium_env::features::label::HostLabel;

pub fn obs_batch(rows: &[&Observation], device: &Device) -> ObsBatch {
    obs_batch_padded(rows, 1, device)
}

/// `obs_batch` with N rounded up to a multiple of `pad` (masked like any other padding). Training
/// sees a new N almost every update; bucketing N keeps the number of distinct shapes small.
pub fn obs_batch_padded(rows: &[&Observation], pad: usize, device: &Device) -> ObsBatch {
    let b = rows.len();
    let n = rows.iter().map(|o| o.obs.n).max().unwrap_or(0).max(1).next_multiple_of(pad.max(1));
    let k = rows.first().map_or(0, |o| o.masks.unit_cap.len() / o.obs.n.max(1));
    let k = if k == 0 { uranium_env::actions::CAPABILITIES.len() } else { k };
    let a = rows.first().map_or(0, |o| o.masks.type_mask.len());
    let l = rows.first().map_or(0, |o| o.masks.loc_class_mask.len() / CELLS);
    let fs = rows.first().map_or(0, |o| o.obs.scalar.len());
    let u = rows.first().map_or(0, |o| o.obs.z_units.len());

    let mut cat = vec![0i32; b * n * ENTITY_CAT];
    let mut cont = vec![0f32; b * n * ENTITY_CONT];
    let mut mask = vec![false; b * n];
    let mut cell = vec![0i32; b * n];
    let mut pos = vec![0f32; b * n * 2];
    let mut caps = vec![false; b * n * k];
    let mut building = vec![false; b * n];
    let mut spatial = Vec::with_capacity(b * SPATIAL * CELLS);
    let mut scalar = Vec::with_capacity(b * fs);
    let mut scalar_cat = Vec::with_capacity(b * 3);
    let mut z_build = Vec::with_capacity(b * Z_BUILD);
    let mut z_units = Vec::with_capacity(b * u);
    let mut z_present = Vec::with_capacity(b);
    let mut type_mask = Vec::with_capacity(b * a);
    let mut loc = Vec::with_capacity(b * l * CELLS);
    for (r, o) in rows.iter().enumerate() {
        let m = o.obs.n;
        let base = r * n;
        cat[base * ENTITY_CAT..(base + m) * ENTITY_CAT].copy_from_slice(&o.obs.entity_cat);
        cont[base * ENTITY_CONT..(base + m) * ENTITY_CONT].copy_from_slice(&o.obs.entity_cont);
        mask[base..base + m].fill(true);
        cell[base..base + m].copy_from_slice(&o.obs.entity_cell);
        pos[base * 2..(base + m) * 2].copy_from_slice(&o.obs.entity_pos);
        caps[base * k..(base + m) * k].copy_from_slice(&o.masks.unit_cap);
        building[base..base + m].copy_from_slice(&o.masks.is_building);
        spatial.extend_from_slice(&o.obs.spatial);
        scalar.extend_from_slice(&o.obs.scalar);
        scalar_cat.extend_from_slice(&o.obs.scalar_cat);
        z_build.extend_from_slice(&o.obs.z_build);
        z_units.extend_from_slice(&o.obs.z_units);
        z_present.push(o.obs.z_present);
        type_mask.extend_from_slice(&o.masks.type_mask);
        loc.extend_from_slice(&o.masks.loc_class_mask);
    }
    let f = |v: Vec<f32>, s: Vec<usize>| TensorData::new(v, s);
    ObsBatch {
        entity_cat: Tensor::<3, Int>::from_data(TensorData::new(cat, [b, n, ENTITY_CAT]), device),
        entity_cont: Tensor::<3>::from_data(f(cont, vec![b, n, ENTITY_CONT]), device),
        entity_mask: Tensor::<2, Bool>::from_data(TensorData::new(mask, [b, n]), device),
        entity_cell: Tensor::<2, Int>::from_data(TensorData::new(cell, [b, n]), device),
        entity_pos: Tensor::<3>::from_data(f(pos, vec![b, n, 2]), device),
        spatial: Tensor::<4>::from_data(f(spatial, vec![b, SPATIAL, MAP, MAP]), device),
        scalar: Tensor::<2>::from_data(f(scalar, vec![b, fs]), device),
        scalar_cat: Tensor::<2, Int>::from_data(TensorData::new(scalar_cat, [b, 3]), device),
        z_build: Tensor::<2, Int>::from_data(TensorData::new(z_build, [b, Z_BUILD]), device),
        z_units: Tensor::<2>::from_data(f(z_units, vec![b, u]), device),
        z_present: Tensor::<1, Bool>::from_data(TensorData::new(z_present, [b]), device),
        type_mask: Tensor::<2, Bool>::from_data(TensorData::new(type_mask, [b, a]), device),
        unit_cap: Tensor::<3, Bool>::from_data(TensorData::new(caps, [b, n, k]), device),
        loc_class_mask: Tensor::<4, Bool>::from_data(TensorData::new(loc, [b, l, MAP, MAP]), device),
        is_building: Tensor::<2, Bool>::from_data(TensorData::new(building, [b, n]), device),
    }
}

/// Device batch from the host collate of `uranium-data` (same layout, AMB-000-4): the BC trainer
/// collates lane groups there (`Batch::group`), as the Python trainer does.
pub fn obs_from_host(h: &uranium_data::collate::HostObsBatch, device: &Device) -> ObsBatch {
    let (b, n) = (h.b, h.n);
    fn ft<const D: usize>(v: &[f32], s: Vec<usize>, device: &Device) -> Tensor<D> {
        Tensor::from_data(TensorData::new(v.to_vec(), s), device)
    }
    let i2 = |v: &Vec<i32>, s: [usize; 2]| Tensor::<2, Int>::from_data(TensorData::new(v.clone(), s), device);
    let b2 = |v: &Vec<bool>, s: [usize; 2]| Tensor::<2, Bool>::from_data(TensorData::new(v.clone(), s), device);
    ObsBatch {
        entity_cat: Tensor::<3, Int>::from_data(TensorData::new(h.entity_cat.clone(), [b, n, ENTITY_CAT]), device),
        entity_cont: ft(&h.entity_cont, vec![b, n, ENTITY_CONT], device),
        entity_mask: b2(&h.entity_mask, [b, n]),
        entity_cell: i2(&h.entity_cell, [b, n]),
        entity_pos: ft(&h.entity_pos, vec![b, n, 2], device),
        spatial: ft(&h.spatial, vec![b, SPATIAL, MAP, MAP], device),
        scalar: ft(&h.scalar, vec![b, h.fs], device),
        scalar_cat: i2(&h.scalar_cat, [b, 3]),
        z_build: i2(&h.z_build, [b, Z_BUILD]),
        z_units: ft(&h.z_units, vec![b, h.u], device),
        z_present: Tensor::<1, Bool>::from_data(TensorData::new(h.z_present.clone(), [b]), device),
        type_mask: b2(&h.type_mask, [b, h.a]),
        unit_cap: Tensor::<3, Bool>::from_data(TensorData::new(h.unit_cap.clone(), [b, n, h.k]), device),
        loc_class_mask: Tensor::<4, Bool>::from_data(TensorData::new(h.loc_class_mask.clone(), [b, h.l, MAP, MAP]), device),
        is_building: b2(&h.is_building, [b, n]),
    }
}

/// Device labels from the host collate (END = the padded n of the matching batch, 032-R12).
pub fn actions_from_host(h: &uranium_data::collate::HostActionBatch, device: &Device) -> ActionBatch {
    let i = |v: &Vec<i32>| Tensor::<1, Int>::from_data(TensorData::new(v.clone(), [h.b]), device);
    ActionBatch {
        action: i(&h.action),
        delay: i(&h.delay),
        queued: i(&h.queued),
        selected: Tensor::<2, Int>::from_data(TensorData::new(h.selected.clone(), [h.b, h.s]), device),
        target_unit: i(&h.target_unit),
        target_loc: i(&h.target_loc),
    }
}

/// Device privileged inputs from the host collate (030-R9).
pub fn priv_from_host(h: &uranium_data::collate::HostPrivBatch, device: &Device) -> PrivBatch {
    PrivBatch { summary: Tensor::from_data(TensorData::new(h.summary.clone(), [h.b, h.p]), device) }
}

/// `n` is the padded entity count of the matching `ObsBatch`; END = n (032-R12).
pub fn action_batch(labels: &[&HostLabel], n: usize, max_select: usize, device: &Device) -> ActionBatch {
    let b = labels.len();
    let i = |v: Vec<i32>| Tensor::<1, Int>::from_data(TensorData::new(v, [b]), device);
    let mut sel = vec![n as i32; b * max_select];
    for (r, l) in labels.iter().enumerate() {
        for (s, &e) in l.selected.iter().take(max_select).enumerate() {
            sel[r * max_select + s] = e as i32;
        }
    }
    ActionBatch {
        action: i(labels.iter().map(|l| l.action as i32).collect()),
        delay: i(labels.iter().map(|l| l.delay_bucket as i32).collect()),
        queued: i(labels.iter().map(|l| l.queued as i32).collect()),
        selected: Tensor::<2, Int>::from_data(TensorData::new(sel, [b, max_select]), device),
        target_unit: i(labels.iter().map(|l| l.target_unit.unwrap_or(0) as i32).collect()),
        target_loc: i(labels.iter().map(|l| l.target_loc.unwrap_or(0) as i32).collect()),
    }
}

pub fn priv_batch(rows: &[&PrivilegedTensors], device: &Device) -> PrivBatch {
    let p = rows.first().map_or(0, |r| r.summary.len());
    let v: Vec<f32> = rows.iter().flat_map(|r| r.summary.iter().copied()).collect();
    PrivBatch { summary: Tensor::from_data(TensorData::new(v, [rows.len(), p]), device) }
}

/// Model input dimensions from the feature schema (030) and the action table (034).
pub fn input_dims() -> crate::config::InputDims {
    let d = uranium_env::features::extract::dims();
    crate::config::InputDims::new()
        .with_actions(d.actions)
        .with_entity_cont(ENTITY_CONT)
        .with_spatial_channels(SPATIAL)
        .with_map_size(MAP)
        .with_scalar(d.scalar())
        .with_z_units(d.z_units())
        .with_caps(d.caps())
        .with_loc_classes(d.loc_classes())
        .with_privileged(d.privileged())
        .with_orders(uranium_env::features::ORDERS)
}
