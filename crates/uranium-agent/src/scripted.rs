use std::collections::BTreeMap;
use anyhow::Result;
use serde::Serialize;
use uranium_proto::*;
use uranium_store::elog;
use uranium_scripted::T0Build;
use super::Policy;

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ScriptedStats {
    pub scout_main_s: Option<f64>,
    pub scout_types_4m: u32,
    pub scout_types_6m: u32,
    pub barracks_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub scv_4m: Option<u32>,
    pub scv_min_1m: Option<u32>,
    pub army_s: Option<f64>,
    pub scv_8m: Option<u32>,
    pub react: Option<ReactRecord>,
    pub ops: uranium_env::ops::OpsMetrics,
    pub rush_combat: Option<uranium_env::ops::RushCombat>,
    pub tp: Option<TpRecord>,
}

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct TpRecord {
    pub main_s: Option<f64>,
    pub rush_s: Option<f64>,
    pub rush_rule: Option<uranium_scripted::TpRule>,
    pub defences: u32,
    pub macro_s: Option<f64>,
    pub refinery_s: Option<f64>,
    pub factory_s: Option<f64>,
    pub shop_s: Option<f64>,
    pub siege_s: Option<f64>,
    pub tank_s: Option<f64>,
    pub expand_s: Option<f64>,
    pub wave_s: Option<f64>,
    pub branch: uranium_scripted::Branch,
    pub contact_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub rules_s: BTreeMap<uranium_scripted::TpRule, f64>,
}

#[derive(Serialize, Debug, Clone, Default, PartialEq)]
pub struct ReactRecord {
    pub main_s: Option<f64>,
    pub rush_s: Option<f64>,
    pub rush_rule: Option<uranium_scripted::RushRule>,
    pub defences: u32,
    pub macro_s: Option<f64>,
    pub expand_s: Option<f64>,
    pub branch: uranium_scripted::Branch,
    pub pool_start_s: Option<f64>,
    pub pool_seen_s: Option<f64>,
    pub lings_max: u32,
    pub contact_s: Option<f64>,
    pub bunker_s: Option<f64>,
    pub rules_s: BTreeMap<uranium_scripted::RushRule, f64>,
}

pub struct ScriptedPolicy {
    bot: Box<dyn uranium_scripted::Bot>,
    scout: Option<uranium_env::scout::ScoutTracker>,
    ops: Option<uranium_env::ops::OpsTracker>,
}

impl ScriptedPolicy {
    pub fn new(info: &GameInfo, build: T0Build) -> Result<Self> {
        let race = info.players.iter().find(|p| p.is_self).map(|p| p.race);
        let terran = race == Some(Race::Terran);
        let (hints, pre) = if build.needs_map_hints() && terran {
            let family = uranium_env::maps::MapFamilies::load()?.classify(&info.map_name, "");
            let (pre, _) = uranium_env::maps::live_precompute_for_session(uranium_env::maps::maps_dir(), info, &family)?;
            if pre.is_none() {
                elog!("[agent] WARNING: no map precompute for {:?}; the teacher does not expand", info.map_name);
            }
            (pre.as_ref().map(|p| uranium_scripted::MapHints { base_locs: p.base_locs.clone(), chokes: p.chokes.clone() }).unwrap_or_default(), pre)
        } else {
            let pre = if terran { uranium_env::maps::maps_dir().ok().and_then(|d| uranium_env::maps::live_precompute(&d, info).0) } else { None };
            (uranium_scripted::MapHints::default(), pre)
        };
        let scout = terran.then(|| uranium_env::scout::ScoutTracker::for_game(info, uranium_env::scout::RewardConfig::default()));
        let ops = terran.then(|| uranium_env::ops::OpsTracker::for_game(info, pre.as_ref()));
        Ok(Self { bot: uranium_scripted::for_game_hints(info, build, hints), scout, ops })
    }
}

impl Policy for ScriptedPolicy {
    fn on_frame(&mut self, f: &RawFrame) -> Result<Act> {
        if let Some(s) = self.scout.as_mut() {
            s.step(f);
        }
        if let Some(o) = self.ops.as_mut() {
            o.step(f);
        }
        Ok(self.bot.act(f))
    }

    fn on_end(&mut self, _outcome: Option<uranium_env::reward::Outcome>, final_frame: Option<&RawFrame>) {
        if let (Some(s), Some(f)) = (self.scout.as_mut(), final_frame) {
            s.step(f);
        }
        if let (Some(o), Some(f)) = (self.ops.as_mut(), final_frame) {
            o.step(f);
        }
    }

    fn scripted_stats(&mut self) -> Option<ScriptedStats> {
        let m = self.scout.as_ref()?.metrics();
        let secs = |f: Option<u32>| f.map(uranium_env::scout::seconds);
        let react = self.bot.react_stats().map(|r| ReactRecord {
            main_s: secs(r.main_frame),
            rush_s: secs(r.rush_frame),
            rush_rule: r.rush_rule,
            defences: r.defences,
            macro_s: secs(r.macro_frame),
            expand_s: secs(r.expand_frame),
            branch: r.branch,
            pool_start_s: secs(r.pool_start_frame),
            pool_seen_s: secs(r.pool_seen_frame),
            lings_max: r.lings_max,
            contact_s: secs(r.contact_frame),
            bunker_s: secs(r.bunker_frame),
            rules_s: r.rule_frames.iter().map(|(k, f)| (*k, uranium_env::scout::seconds(*f))).collect(),
        });
        let tp = self.bot.tp_stats().map(|r| TpRecord {
            main_s: secs(r.main_frame),
            rush_s: secs(r.rush_frame),
            rush_rule: r.rush_rule,
            defences: r.defences,
            macro_s: secs(r.macro_frame),
            refinery_s: secs(r.refinery_frame),
            factory_s: secs(r.factory_frame),
            shop_s: secs(r.shop_frame),
            siege_s: secs(r.siege_frame),
            tank_s: secs(r.tank_frame),
            expand_s: secs(r.expand_frame),
            wave_s: secs(r.wave_frame),
            branch: r.branch,
            contact_s: secs(r.contact_frame),
            bunker_s: secs(r.bunker_frame),
            rules_s: r.rule_frames.iter().map(|(k, f)| (*k, uranium_env::scout::seconds(*f))).collect(),
        });
        Some(ScriptedStats {
            scout_main_s: secs(m.main_frame),
            scout_types_4m: m.types_4m,
            scout_types_6m: m.types_6m,
            barracks_s: secs(m.barracks_frame),
            bunker_s: secs(m.bunker_frame),
            scv_4m: m.scv_4m,
            scv_min_1m: m.scv_min_1m,
            army_s: secs(m.army_frame),
            scv_8m: m.scv_8m,
            react,
            ops: self.ops.as_ref().map(|o| o.ops()).unwrap_or_default(),
            rush_combat: self.ops.as_ref().and_then(|o| o.rush_combat()),
            tp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0_info(race: Race) -> GameInfo {
        let n = 64usize * 64;
        GameInfo {
            map_name: "Python".into(),
            map_hash: [0; 32],
            map_size_tiles: (64, 64),
            self_player: 0,
            players: vec![
                PlayerInfo { id: 0, name: "p0-T0".into(), race, start_tile: Some((4, 4)), is_self: true, is_enemy: false },
                PlayerInfo { id: 1, name: "p1-x".into(), race: Race::Zerg, start_tile: None, is_self: false, is_enemy: true },
            ],
            static_map: StaticMap {
                width_tiles: 64,
                height_tiles: 64,
                walkable: BitGrid::from_bytes(256, 256, &vec![1; n * 16]),
                buildable: BitGrid::from_bytes(64, 64, &vec![1; n]),
                ground_height: vec![0; n],
                start_locations: vec![(4, 4), (56, 56)],
            },
            latency_frames: 0,
        }
    }

    fn scv_frame(frame: u32, scvs: u32) -> RawFrame {
        let units = (0..scvs).map(|i| UnitState { id: 10 + i, owner: 0, unit_type: 7, pos: (200, 200), flags: flags::COMPLETED, ..Default::default() }).collect();
        RawFrame { frame, self_state: PlayerState::default(), units, visibility: BitGrid::new(64, 64), explored: BitGrid::new(64, 64), creep: BitGrid::new(64, 64), events: vec![], effects: vec![] }
    }

    #[test]
    fn scripted_row_050_r8() {
        let mut p = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::Standard).unwrap();
        p.on_frame(&scv_frame(1_440, 6)).unwrap();
        p.on_frame(&scv_frame(5_000, 3)).unwrap();
        p.on_end(None, Some(&scv_frame(11_520, 9)));
        let s = p.scripted_stats().unwrap();
        assert_eq!((s.scv_min_1m, s.scv_4m, s.scv_8m, s.react.clone()), (Some(3), Some(9), Some(9), None));
        let text = serde_json::to_string(&s).unwrap();
        assert!(text.starts_with(r#"{"scout_main_s":null,"scout_types_4m":0,"scout_types_6m":0,"barracks_s":null,"bunker_s":null,"scv_4m":9,"scv_min_1m":3,"army_s":null,"scv_8m":9,"react":null,"ops":{"end_s":480.0,"mined":0,"#), "{text}");
        assert!(text.ends_with(r#""float_gas_rs":0.0},"rush_combat":null,"tp":null}"#), "{text}");
        assert_eq!(s.ops.end_s, 480.0, "the final frame counts (052-R6, 010-R12)");

        let mut t = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::TerranReact).unwrap();
        t.on_frame(&scv_frame(100, 4)).unwrap();
        let r = t.scripted_stats().unwrap().react.expect("teacher record");
        assert_eq!((r.rush_s, r.defences, r.branch), (None, 0, uranium_scripted::Branch::Macro));
        let text = serde_json::to_string(&r).unwrap();
        assert_eq!(text, r#"{"main_s":null,"rush_s":null,"rush_rule":null,"defences":0,"macro_s":null,"expand_s":null,"branch":"macro","pool_start_s":null,"pool_seen_s":null,"lings_max":0,"contact_s":null,"bunker_s":null,"rules_s":{}}"#);

        let mut z = ScriptedPolicy::new(&t0_info(Race::Zerg), T0Build::ZergRush).unwrap();
        z.on_frame(&scv_frame(100, 0)).unwrap();
        assert!(z.scripted_stats().is_none());

        let mut tz = ScriptedPolicy::new(&t0_info(Race::Terran), T0Build::TerranTp).unwrap();
        tz.on_frame(&scv_frame(100, 4)).unwrap();
        let s = tz.scripted_stats().unwrap();
        assert!(s.react.is_some() && s.tp.is_none());
        let mut info = t0_info(Race::Terran);
        info.players[1].race = Race::Protoss;
        let mut tp = ScriptedPolicy::new(&info, T0Build::TerranTp).unwrap();
        tp.on_frame(&scv_frame(100, 4)).unwrap();
        let s = tp.scripted_stats().unwrap();
        assert!(s.react.is_none());
        let text = serde_json::to_string(&s.tp.expect("022 record")).unwrap();
        assert_eq!(
            text,
            r#"{"main_s":null,"rush_s":null,"rush_rule":null,"defences":0,"macro_s":null,"refinery_s":null,"factory_s":null,"shop_s":null,"siege_s":null,"tank_s":null,"expand_s":null,"wave_s":null,"branch":"macro","contact_s":null,"bunker_s":null,"rules_s":{}}"#
        );
    }
}
