//! Device-side input batches (030 §1 tensor contract) and the 034 action table as GPU constants
//! (032 §5). `ObsBatch` has no privileged field: privileged inputs travel only in `PrivBatch`
//! (030-R9, 030-AC5, 032-R11).

use anyhow::{Result, ensure};
use burn::module::Module;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use uranium_env::actions::{ActionTable, CAPABILITIES, LOC_CLASSES, TargetKind};

/// One decision step for B games (or B·T flattened steps in training). Rows are padded to the
/// largest N in the batch (030-R7, 032-R7).
#[derive(Debug, Clone)]
pub struct ObsBatch {
    /// [B,N,4] unit_type, owner (0 self / 1 enemy / 2 neutral), order, last_action_type.
    pub entity_cat: Tensor<3, Int>,
    /// [B,N,F_c]
    pub entity_cont: Tensor<3>,
    /// [B,N] true for real entities.
    pub entity_mask: Tensor<2, Bool>,
    /// [B,N] tile index y*M+x.
    pub entity_cell: Tensor<2, Int>,
    /// [B,N,2] pixel position.
    pub entity_pos: Tensor<3>,
    /// [B,C,M,M]
    pub spatial: Tensor<4>,
    /// [B,F_s]
    pub scalar: Tensor<2>,
    /// [B,3] enemy_race, map_family, start_loc.
    pub scalar_cat: Tensor<2, Int>,
    /// [B,20] action ids, 0 = padding.
    pub z_build: Tensor<2, Int>,
    /// [B,U]
    pub z_units: Tensor<2>,
    /// [B]
    pub z_present: Tensor<1, Bool>,
    /// [B,A] (034-R3)
    pub type_mask: Tensor<2, Bool>,
    /// [B,N,K] (034)
    pub unit_cap: Tensor<3, Bool>,
    /// [B,L,M,M] (034)
    pub loc_class_mask: Tensor<4, Bool>,
    /// [B,N] entity is a building, lifted / bunker / turret included (034-R7, 032 §5, E-4):
    /// the Selected head allows a building only as the sole selection.
    pub is_building: Tensor<2, Bool>,
}

impl ObsBatch {
    pub fn batch(&self) -> usize {
        self.entity_mask.dims()[0]
    }

    pub fn entities(&self) -> usize {
        self.entity_mask.dims()[1]
    }

    /// Rows `start..start + len` of the entity encoder's inputs (entity_cat, entity_cont,
    /// entity_mask, entity_pos) with N kept; the other fields are left whole, as the entity
    /// encoder does not read them (041-R17 row chunks).
    pub fn entity_rows(&self, start: usize, len: usize) -> ObsBatch {
        ObsBatch {
            entity_cat: self.entity_cat.clone().narrow(0, start, len),
            entity_cont: self.entity_cont.clone().narrow(0, start, len),
            entity_mask: self.entity_mask.clone().narrow(0, start, len),
            entity_pos: self.entity_pos.clone().narrow(0, start, len),
            ..self.clone()
        }
    }

    /// [B,N] owner == self (034-R4).
    pub fn is_self(&self) -> Tensor<2, Bool> {
        let [b, n, _] = self.entity_cat.dims();
        self.entity_cat.clone().narrow(2, 1, 1).reshape([b, n]).equal_elem(0)
    }
}

/// Training-only privileged inputs (030-R9), consumed by the value heads only (032-R11).
#[derive(Debug, Clone)]
pub struct PrivBatch {
    /// [B,P]
    pub summary: Tensor<2>,
}

/// Action labels (teacher forcing, 032-R3) or sampled actions. Unused heads may hold any value;
/// the action table decides which heads count (032-R10).
#[derive(Debug, Clone)]
pub struct ActionBatch {
    /// [B]
    pub action: Tensor<1, Int>,
    /// [B] delay bucket index.
    pub delay: Tensor<1, Int>,
    /// [B] 0 / 1.
    pub queued: Tensor<1, Int>,
    /// [B,S] entity indices in canonical (ascending) order then END = N (032-R12); entries after
    /// END are ignored. S = max_select.
    pub selected: Tensor<2, Int>,
    /// [B] entity index.
    pub target_unit: Tensor<1, Int>,
    /// [B] tile index y*M+x.
    pub target_loc: Tensor<1, Int>,
}

/// 034 table columns as device constants for the mask gather (032 §5, 034-R4).
#[derive(Module, Debug)]
pub struct ActionTableTensors {
    /// [A] 0 / 1
    pub uses_select: Tensor<1, Int>,
    /// [A] 0 when unused.
    pub max_select: Tensor<1, Int>,
    /// [A] capability index (0 when unused).
    pub select_cap: Tensor<1, Int>,
    /// [A] 0 / 1
    pub uses_queued: Tensor<1, Int>,
    /// [A] 0 none, 1 unit, 2 loc.
    pub target_kind: Tensor<1, Int>,
    /// [A] capability index (0 when unused).
    pub target_cap: Tensor<1, Int>,
    /// [A] location class index (0 when unused).
    pub loc_class: Tensor<1, Int>,
    /// [A] 0 / 1: the target may be a selected unit (034-R9).
    pub self_target: Tensor<1, Int>,
}

pub const TARGET_NONE: i32 = 0;
pub const TARGET_UNIT: i32 = 1;
pub const TARGET_LOC: i32 = 2;

/// Host copy of the action table columns the model needs.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionColumns {
    pub uses_select: Vec<i32>,
    pub max_select: Vec<i32>,
    pub select_cap: Vec<i32>,
    pub uses_queued: Vec<i32>,
    pub target_kind: Vec<i32>,
    pub target_cap: Vec<i32>,
    pub loc_class: Vec<i32>,
    pub self_target: Vec<i32>,
}

impl ActionColumns {
    pub fn from_table(table: &ActionTable) -> Result<Self> {
        let cap = |name: &str| CAPABILITIES.iter().position(|c| *c == name).map_or(0, |i| i as i32);
        let loc = |name: &str| LOC_CLASSES.iter().position(|c| *c == name).map_or(0, |i| i as i32);
        let mut c = Self {
            uses_select: vec![],
            max_select: vec![],
            select_cap: vec![],
            uses_queued: vec![],
            target_kind: vec![],
            target_cap: vec![],
            loc_class: vec![],
            self_target: vec![],
        };
        for a in &table.actions {
            ensure!(a.max_select as usize <= 12, "action {}: max_select > 12 (031-R3)", a.name);
            c.uses_select.push(a.uses_select as i32);
            c.max_select.push(if a.uses_select { a.max_select as i32 } else { 0 });
            c.select_cap.push(cap(&a.select_cap));
            c.uses_queued.push(a.uses_queued as i32);
            c.target_kind.push(match a.target {
                TargetKind::None => TARGET_NONE,
                TargetKind::Unit => TARGET_UNIT,
                TargetKind::Loc => TARGET_LOC,
            });
            c.target_cap.push(cap(&a.target_cap));
            c.loc_class.push(loc(&a.loc_class));
            c.self_target.push(a.self_target as i32);
        }
        Ok(c)
    }

    pub fn len(&self) -> usize {
        self.uses_select.len()
    }

    pub fn is_empty(&self) -> bool {
        self.uses_select.is_empty()
    }

    pub fn to_tensors(&self, device: &Device) -> ActionTableTensors {
        let t = |v: &Vec<i32>| Tensor::<1, Int>::from_data(TensorData::new(v.clone(), [v.len()]), device);
        ActionTableTensors {
            uses_select: t(&self.uses_select),
            max_select: t(&self.max_select),
            select_cap: t(&self.select_cap),
            uses_queued: t(&self.uses_queued),
            target_kind: t(&self.target_kind),
            target_cap: t(&self.target_cap),
            loc_class: t(&self.loc_class),
            self_target: t(&self.self_target),
        }
    }
}

/// Per-row head usage and mask indices for a chosen action (gathered on device).
#[derive(Debug, Clone)]
pub struct ActionInfo {
    pub uses_select: Tensor<1, Bool>,
    pub max_select: Tensor<1, Int>,
    pub select_cap: Tensor<1, Int>,
    pub uses_queued: Tensor<1, Bool>,
    pub target_unit: Tensor<1, Bool>,
    pub target_loc: Tensor<1, Bool>,
    pub target_cap: Tensor<1, Int>,
    pub loc_class: Tensor<1, Int>,
    /// [B] the target may be a selected unit (034-R9).
    pub self_target: Tensor<1, Bool>,
}

impl ActionTableTensors {
    /// Gather the table rows for `action` [B].
    pub fn info(&self, action: Tensor<1, Int>) -> ActionInfo {
        let g = |col: &Tensor<1, Int>| col.clone().select(0, action.clone());
        let kind = g(&self.target_kind);
        ActionInfo {
            uses_select: g(&self.uses_select).equal_elem(1),
            max_select: g(&self.max_select),
            select_cap: g(&self.select_cap),
            uses_queued: g(&self.uses_queued).equal_elem(1),
            target_unit: kind.clone().equal_elem(TARGET_UNIT),
            target_loc: kind.equal_elem(TARGET_LOC),
            target_cap: g(&self.target_cap),
            loc_class: g(&self.loc_class),
            self_target: g(&self.self_target).equal_elem(1),
        }
    }
}

/// `unit_cap[b, n, cap[b]]` -> [B,N] (034-R4).
pub fn gather_cap(unit_cap: Tensor<3, Bool>, cap: Tensor<1, Int>) -> Tensor<2, Bool> {
    let [b, n, _] = unit_cap.dims();
    let idx = cap.reshape([b, 1, 1]).expand([b, n, 1]);
    unit_cap.int().gather(2, idx).reshape([b, n]).equal_elem(1)
}

/// `loc_class_mask[b, class[b]]` flattened -> [B,M*M] (034-R4).
pub fn gather_loc(loc_class_mask: Tensor<4, Bool>, class: Tensor<1, Int>) -> Tensor<2, Bool> {
    let [b, l, h, w] = loc_class_mask.dims();
    let idx = class.reshape([b, 1, 1]).expand([b, 1, h * w]);
    loc_class_mask.reshape([b, l, h * w]).int().gather(1, idx).reshape([b, h * w]).equal_elem(1)
}
