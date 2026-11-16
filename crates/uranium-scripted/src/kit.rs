use uranium_proto::Race;

use crate::T0Build;
use crate::react::ReactParams;
use crate::tp::TpParams;
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tactic {
    Waves,
    Drop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Supply {
    Building(u16),
    Larva(u16),
}

#[derive(Clone, Debug)]
pub struct Kit {
    pub race: Race,
    pub worker: u16,
    pub max_workers: usize,
    pub supply: Supply,
    pub production: u16,
    pub production_cap: usize,
    pub production_at_workers: [usize; 3],
    pub army: u16,
    pub first_wave: usize,
    pub attacker: u16,
    pub army_cap: usize,
    pub tech: Vec<u16>,
    pub gas_workers: usize,
    pub special: Option<(u16, u16, usize)>,
    pub tactic: Tactic,
    pub scout: bool,
    pub defend: bool,
    pub react: Option<ReactParams>,
    pub tp: Option<TpParams>,
}

impl Kit {
    pub fn for_build(race: Race, build: T0Build) -> Self {
        let build = if build.race() == Some(race) { build } else { T0Build::Standard };
        match (build, race) {
            (T0Build::ZergRush, _) => Kit::zerg_rush(),
            (T0Build::ZergMuta, _) => Kit::zerg_muta(),
            (T0Build::TerranDrop, _) => Kit::terran_drop(),
            (T0Build::ProtossDt, _) => Kit::protoss_dt(),
            (T0Build::TerranReact, _) => Kit::terran_react(),
            (T0Build::TerranTp, _) => Kit::terran_tp(),
            (T0Build::Standard, Race::Protoss) => Kit::protoss(),
            (T0Build::Standard, Race::Zerg) => Kit::zerg(),
            (T0Build::Standard, _) => Kit::terran(),
        }
    }

    pub fn terran() -> Self {
        Self {
            race: Race::Terran,
            worker: SCV,
            max_workers: 22,
            supply: Supply::Building(DEPOT),
            production: BARRACKS,
            production_cap: 3,
            production_at_workers: [10, 14, 14],
            army: MARINE,
            first_wave: 10,
            attacker: MARINE,
            army_cap: usize::MAX,
            tech: Vec::new(),
            gas_workers: 0,
            special: None,
            tactic: Tactic::Waves,
            scout: false,
            defend: false,
            react: None,
            tp: None,
        }
    }
    pub fn protoss() -> Self {
        Self {
            race: Race::Protoss,
            worker: PROBE,
            max_workers: 20,
            supply: Supply::Building(PYLON),
            production: GATEWAY,
            production_cap: 3,
            production_at_workers: [10, 13, 15],
            army: ZEALOT,
            first_wave: 6,
            attacker: ZEALOT,
            army_cap: usize::MAX,
            tech: Vec::new(),
            gas_workers: 0,
            special: None,
            tactic: Tactic::Waves,
            scout: false,
            defend: false,
            react: None,
            tp: None,
        }
    }
    pub fn zerg() -> Self {
        Self {
            race: Race::Zerg,
            worker: DRONE,
            max_workers: 14,
            supply: Supply::Larva(OVERLORD),
            production: SPAWNING_POOL,
            production_cap: 1,
            production_at_workers: [9, usize::MAX, usize::MAX],
            army: ZERGLING,
            first_wave: 12,
            attacker: ZERGLING,
            army_cap: usize::MAX,
            tech: Vec::new(),
            gas_workers: 0,
            special: None,
            tactic: Tactic::Waves,
            scout: false,
            defend: false,
            react: None,
            tp: None,
        }
    }

    pub fn zerg_rush() -> Self {
        Self { max_workers: 6, production_at_workers: [5, usize::MAX, usize::MAX], first_wave: 6, ..Self::zerg() }
    }

    pub fn zerg_muta() -> Self {
        Self {
            max_workers: 16,
            first_wave: 5,
            attacker: MUTALISK,
            army_cap: 6,
            tech: vec![EXTRACTOR, LAIR, SPIRE],
            gas_workers: 3,
            special: Some((LARVA, MUTALISK, usize::MAX)),
            defend: true,
            ..Self::zerg()
        }
    }

    pub fn protoss_dt() -> Self {
        Self {
            production_cap: 2,
            first_wave: 2,
            attacker: DARK_TEMPLAR,
            army_cap: 3,
            tech: vec![ASSIMILATOR, CYBERNETICS_CORE, CITADEL, TEMPLAR_ARCHIVES],
            gas_workers: 3,
            special: Some((GATEWAY, DARK_TEMPLAR, usize::MAX)),
            defend: true,
            ..Self::protoss()
        }
    }

    pub fn terran_drop() -> Self {
        Self {
            production_cap: 2,
            production_at_workers: [10, 16, usize::MAX],
            tech: vec![REFINERY, FACTORY, STARPORT, CONTROL_TOWER],
            gas_workers: 3,
            special: Some((STARPORT, DROPSHIP, 2)),
            tactic: Tactic::Drop,
            scout: true,
            defend: true,
            ..Self::terran()
        }
    }

    pub fn terran_react() -> Self {
        Self { defend: true, react: Some(ReactParams::default()), ..Self::terran() }
    }

    pub fn terran_tp() -> Self {
        let tp = TpParams::default();
        Self { defend: true, react: Some(tp.react_params()), tp: Some(tp), ..Self::terran() }
    }
}
