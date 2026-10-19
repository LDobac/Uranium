//! 034 action table loader (034-R1, 034-R2, 034-AC1). The table file is the single source for
//! env masks (M5), the model's mask gather (M5) and replay label inversion (040, M4).

use std::collections::HashSet;

use anyhow::{Result, bail, ensure};
use serde::Deserialize;

/// `tables/actions.toml`, compiled in so every crate sees the same table.
pub const ACTIONS_TOML: &str = include_str!("../tables/actions.toml");

/// 034-R2 / 032-R6: bump when action ids change meaning; stored in shard headers (040).
/// 2: `Unload_Unit` (id 100) appended, capability class `loaded_unit`
/// (2026-10-19, action audit F7). Ids 0..=99 keep their meaning, so schema-1 shards stay readable.
pub const ACTION_SCHEMA_VERSION: u32 = 2;

/// 034 capability classes (`unit_cap`).
pub const CAPABILITIES: &[&str] = &[
    // selection
    "can_move", "can_attack", "is_worker", "can_be_loaded", "is_transport_loaded", "can_stim", "can_siege", "can_unsiege",
    "can_cloak", "can_decloak", "can_lay_mine", "can_cast_lockdown", "can_cast_nuke", "can_heal", "can_cast_restoration",
    "can_cast_flare", "can_cast_matrix", "can_cast_emp", "can_cast_irradiate", "can_cast_yamato", "can_lift", "can_land_noncc",
    "can_land_cc", "can_rally", "addon_capable_cc", "addon_capable_factory", "addon_capable_starport", "addon_capable_scifac",
    "can_cancel_construction", "can_cancel_train", "can_cancel_research", "can_cancel_upgrade", "can_cancel_addon",
    // targets
    "attackable_enemy", "any_unit", "own_unit", "target_resource", "repairable", "lockdownable", "healable", "restorable",
    "flarable", "irradiatable", "matrixable", "yamato_target", "loadable_transport",
    // selection, appended 2026-10-17 (AMB-041-1)
    "can_right_click_unit",
    // Unload_Unit target, appended 2026-10-19 (action audit F7; the selection is is_transport_loaded)
    "loaded_unit",
    // selection, appended 2026-10-19 (액션 감사 F5, F6)
    "can_stop", "can_unload_pos",
];

/// 034 location classes (`loc_class_mask`, L = 8).
pub const LOC_CLASSES: &[&str] =
    &["any", "walkable", "buildable_4x3", "buildable_3x2", "buildable_2x2", "cc_slot", "refinery_slot", "landable_4x3"];

/// Command kinds an action maps to (010 `Command` / `CmdOrder`).
pub const COMMANDS: &[&str] = &[
    "None", "UnitMove", "UnitAttack", "UnitPatrol", "UnitHold", "UnitStop", "UnitRightClick", "UnitGather", "UnitReturnCargo",
    "UnitRepair", "Build", "BuildAddon", "Train", "Research", "Upgrade", "UnitUseTech", "UnitSiege", "UnitUnsiege", "UnitCloak",
    "UnitDecloak", "UnitLoad", "UnitUnloadAll", "UnitLift", "UnitLand", "SetRally", "Cancel", "UnitUnload",
];

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TargetKind {
    None,
    Unit,
    Loc,
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Executor {
    None,
    PickWorker,
    PickBuilding,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Action {
    pub id: u16,
    pub name: String,
    pub group: String,
    pub uses_select: bool,
    pub max_select: u8,
    pub select_cap: String,
    pub uses_queued: bool,
    pub target: TargetKind,
    pub target_cap: String,
    pub loc_class: String,
    pub executor: Executor,
    pub command: String,
    /// BWAPI UnitType for Build / BuildAddon / Train.
    pub unit_type: Option<u16>,
    /// BWAPI TechType for Research / UnitUseTech.
    pub tech: Option<u16>,
    /// BWAPI UpgradeType for Upgrade.
    pub upgrade: Option<u16>,
    /// Building type `pick_building` chooses from (031-R1).
    pub producer: Option<String>,
    /// 010 `CancelKind` for Cancel.
    pub cancel: Option<String>,
    /// 034-R9: the game accepts the selected unit itself as the target (rally on itself resets
    /// the rally). Every other unit-target action excludes the selection from its targets.
    #[serde(default)]
    pub self_target: bool,
}

#[derive(Deserialize, Debug)]
struct File {
    action: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct ActionTable {
    pub actions: Vec<Action>,
}

impl ActionTable {
    pub fn load() -> Result<Self> {
        Self::parse(ACTIONS_TOML)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let t = Self { actions: toml::from_str::<File>(text)?.action };
        t.validate()?;
        Ok(t)
    }

    pub fn by_name(&self, name: &str) -> Option<&Action> {
        self.actions.iter().find(|a| a.name == name)
    }

    /// 034-AC1: contiguous ids with NoOp = 0, unique names, existing capability / location
    /// classes and command mappings, and the arguments each command needs.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.actions.first().is_some_and(|a| a.id == 0 && a.name == "NoOp"), "action 0 must be NoOp (034-R2)");
        let mut names = HashSet::new();
        for (i, a) in self.actions.iter().enumerate() {
            let ctx = |m: &str| format!("action {} `{}`: {m}", a.id, a.name);
            ensure!(a.id as usize == i, "{}", ctx("ids must be contiguous from 0 (034-R2)"));
            ensure!(names.insert(a.name.as_str()), "{}", ctx("duplicate name"));
            ensure!(COMMANDS.contains(&a.command.as_str()), "{}", ctx("unknown command"));
            if a.uses_select {
                ensure!((1..=12).contains(&a.max_select), "{}", ctx("max_select must be 1..12 (031-R3)"));
                ensure!(CAPABILITIES.contains(&a.select_cap.as_str()), "{}", ctx("unknown select_cap"));
            } else {
                ensure!(a.select_cap.is_empty() && a.max_select == 0, "{}", ctx("select fields without uses_select"));
            }
            match a.target {
                TargetKind::Unit => ensure!(CAPABILITIES.contains(&a.target_cap.as_str()), "{}", ctx("unknown target_cap")),
                TargetKind::Loc => ensure!(LOC_CLASSES.contains(&a.loc_class.as_str()), "{}", ctx("unknown loc_class")),
                TargetKind::None => {}
            }
            if a.target != TargetKind::Unit {
                ensure!(a.target_cap.is_empty(), "{}", ctx("target_cap needs target = unit"));
            }
            if a.target != TargetKind::Loc {
                ensure!(a.loc_class.is_empty(), "{}", ctx("loc_class needs target = loc"));
            }
            let need = |ok: bool, what: &str| -> Result<()> {
                if !ok {
                    bail!("{}", ctx(&format!("{} needs {what}", a.command)));
                }
                Ok(())
            };
            match a.command.as_str() {
                "Build" | "BuildAddon" | "Train" => need(a.unit_type.is_some(), "unit_type")?,
                "Research" | "UnitUseTech" => need(a.tech.is_some(), "tech")?,
                "Upgrade" => need(a.upgrade.is_some(), "upgrade")?,
                "Cancel" => need(a.cancel.is_some(), "cancel")?,
                _ => {}
            }
            need(a.executor != Executor::PickBuilding || a.producer.is_some(), "a producer for pick_building")?;
            need(a.executor != Executor::PickWorker || a.command == "Build", "Build for pick_worker")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_loads_and_validates_034_ac1() {
        let t = ActionTable::load().unwrap();
        assert_eq!(t.actions.len(), 101);
        // F7 (2026-10-19): Unload_Unit appended last, ids 0..=99 unchanged (034-R2).
        let u = t.by_name("Unload_Unit").unwrap();
        assert_eq!(
            (u.id, u.max_select, u.select_cap.as_str(), u.target_cap.as_str(), u.command.as_str()),
            (100, 1, "is_transport_loaded", "loaded_unit", "UnitUnload")
        );
        assert_eq!(t.by_name("Cancel_Addon").unwrap().id, 99);
        assert_eq!(t.by_name("Attack_Unit").unwrap().target, TargetKind::Unit);
        assert_eq!(t.by_name("Build_Barracks").unwrap().loc_class, "buildable_4x3");
        assert_eq!(t.by_name("Train_Marine").unwrap().producer.as_deref(), Some("Barracks"));
        // 031-R8: targeted skills select exactly one caster.
        for a in t.actions.iter().filter(|a| a.group == "skill" && a.target != TargetKind::None && a.name != "Heal_Unit") {
            assert_eq!(a.max_select, 1, "{}", a.name);
        }
    }

    #[test]
    fn validation_rejects_broken_tables_034_ac1() {
        // Normalise CRLF (Windows checkouts, git autocrlf) before the textual edit.
        let gap = ACTIONS_TOML.replace("\r\n", "\n").replacen("id = 1\n", "id = 2\n", 1);
        assert!(ActionTable::parse(&gap).is_err());
        let bad_cap = ACTIONS_TOML.replacen("select_cap = \"can_move\"", "select_cap = \"can_fly\"", 1);
        assert!(ActionTable::parse(&bad_cap).is_err());
        let bad_cmd = ACTIONS_TOML.replacen("command = \"UnitMove\"", "command = \"Teleport\"", 1);
        assert!(ActionTable::parse(&bad_cmd).is_err());
    }
}
