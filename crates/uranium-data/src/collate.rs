use uranium_env::features::label::HostLabel;
use uranium_env::features::{Observation, PrivilegedTensors};
pub use uranium_env::features::{CELLS, ENTITY_CAT, ENTITY_CONT, MAP, SPATIAL, Z_BUILD};

#[derive(Debug, Clone, PartialEq)]
pub struct HostObsBatch {
    pub b: usize,
    pub n: usize,
    pub k: usize,
    pub a: usize,
    pub l: usize,
    pub fs: usize,
    pub u: usize,
    pub entity_cat: Vec<i32>,
    pub entity_cont: Vec<f32>,
    pub entity_mask: Vec<bool>,
    pub entity_cell: Vec<i32>,
    pub entity_pos: Vec<f32>,
    pub spatial: Vec<f32>,
    pub scalar: Vec<f32>,
    pub scalar_cat: Vec<i32>,
    pub z_build: Vec<i32>,
    pub z_units: Vec<f32>,
    pub z_present: Vec<bool>,
    pub type_mask: Vec<bool>,
    pub unit_cap: Vec<bool>,
    pub loc_class_mask: Vec<bool>,
    pub is_building: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostActionBatch {
    pub b: usize,
    pub s: usize,
    pub action: Vec<i32>,
    pub delay: Vec<i32>,
    pub queued: Vec<i32>,
    pub selected: Vec<i32>,
    pub target_unit: Vec<i32>,
    pub target_loc: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostPrivBatch {
    pub b: usize,
    pub p: usize,
    pub summary: Vec<f32>,
}

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

pub fn obs_batch_packed(rows: &[&Observation], planes: &[&crate::pack::Planes], pad: usize, threads: usize) -> HostObsBatch {
    assert_eq!(rows.len(), planes.len(), "one plane set per row");
    let mut x = obs_batch(rows, pad);
    let (sc, lc) = planes.first().map_or((0, 0), |p| (p.spatial_len(), p.loc_len()));
    x.l = lc / CELLS;
    x.spatial = vec![0f32; rows.len() * sc];
    x.loc_class_mask = vec![false; rows.len() * lc];
    let (mut sp, mut lm): (&mut [f32], &mut [bool]) = (&mut x.spatial, &mut x.loc_class_mask);
    let mut jobs = Vec::with_capacity(planes.len());
    for p in planes {
        let (s, rest) = std::mem::take(&mut sp).split_at_mut(sc);
        sp = rest;
        let (l, rest) = std::mem::take(&mut lm).split_at_mut(lc);
        lm = rest;
        jobs.push((*p, s, l));
    }
    let per = jobs.len().div_ceil(threads.clamp(1, jobs.len().max(1))).max(1);
    std::thread::scope(|s| {
        for run in jobs.chunks_mut(per) {
            s.spawn(move || {
                for (p, sp, lm) in run.iter_mut() {
                    p.restore_into(sp, lm);
                }
            });
        }
    });
    x
}

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
