use std::collections::HashMap;
use anyhow::Result;
use uranium_env::actions::CAPABILITIES;
use uranium_env::features::extract::action_table;
use uranium_env::features::*;
use uranium_env::features::label::{LabelWalker, UNKNOWN_REASON};
use crate::shard::ShardGame;
use uranium_env::maps::family_index;
use super::maps::MapStore;

#[derive(Debug, Default)]
pub struct LabelStats {
    pub fails: std::collections::BTreeMap<(String, &'static str), usize>,
    pub totals: std::collections::BTreeMap<String, usize>,
    pub cap_detail: std::collections::BTreeMap<String, usize>,
    pub type_detail: std::collections::BTreeMap<String, usize>,
    pub cause_detail: std::collections::BTreeMap<String, usize>,
    pub per_game: Vec<(String, String, usize, usize, usize, usize)>,
}

fn age_bucket(frames: u32) -> &'static str {
    match frames {
        0..=8 => "<=8f",
        9..=24 => "<=24f",
        25..=96 => "<=96f",
        _ => ">96f",
    }
}

fn loc_cause(w: &uranium_env::features::world::World, o: &Observation, a: &uranium_env::actions::Action, tile: Option<(u16, u16)>) -> String {
    use uranium_env::features::world::top_left;
    let Some((x, y)) = tile.map(|(x, y)| (x as i32, y as i32)) else { return "no_tile".into() };
    let tt = uranium_env::types::TypeTables::get();
    let lc = uranium_env::actions::LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
    let ok_at = |tx: i32, ty: i32| tx >= 0 && ty >= 0 && (tx as usize) < MAP && (ty as usize) < MAP && o.masks.loc_class_mask[lc * CELLS + ty as usize * MAP + tx as usize];
    let mut causes: Vec<String> = vec![];
    let built = a.unit_type.and_then(|u| tt.unit(u));
    if a.loc_class == "refinery_slot" {
        let geyser = w.entities.iter().find(|e| e.unit.unit_type == w.ids.geyser && w.ty(&e.unit).is_some_and(|t| top_left(&e.unit, t) == (x, y)));
        let refinery = w.frame.units.iter().find(|u| tt.unit(u.unit_type).is_some_and(|t| t.name.ends_with("_Refinery") || t.name.ends_with("_Assimilator") || t.name.ends_with("_Extractor")) && tt.unit(u.unit_type).is_some_and(|t| top_left(u, t) == (x, y)));
        causes.push(match (geyser, refinery) {
            (_, Some(r)) if r.owner == w.mem.self_player => format!("own_refinery_here(completed={})", r.has(uranium_proto::flags::COMPLETED)),
            (_, Some(_)) => "enemy_refinery_here".into(),
            (Some(g), None) if g.remembered => format!("geyser_out_of_vision(explored={})", w.frame.explored.get(x as u16, y as u16)),
            (Some(_), None) => "geyser_visible".into(),
            (None, None) => "no_geyser_at_tile".into(),
        });
    } else {
        let (fw, fh) = match a.loc_class.as_str() {
            "buildable_4x3" | "landable_4x3" | "cc_slot" => (4, 3),
            "buildable_3x2" => (3, 2),
            "buildable_2x2" => (2, 2),
            _ => (1, 1),
        };
        for ty in y..y + fh {
            for tx in x..x + fw {
                if !w.map.in_map(tx, ty) {
                    causes.push("out_of_map".into());
                    continue;
                }
                let i = ty as usize * MAP + tx as usize;
                if !w.map.buildable[i] {
                    causes.push("unbuildable".into());
                }
                if !w.frame.explored.get(tx as u16, ty as u16) {
                    causes.push("unexplored".into());
                }
                if w.frame.creep.get(tx as u16, ty as u16) {
                    causes.push("creep".into());
                }
                if w.blocked[i] {
                    let on = w.entities.iter().find(|e| {
                        w.ty(&e.unit).is_some_and(|t| (t.is_building() || t.is_resource()) && !e.unit.has(uranium_proto::flags::LIFTED) && {
                            let (bx, by) = top_left(&e.unit, t);
                            tx >= bx && tx < bx + t.tile_width && ty >= by && ty < by + t.tile_height
                        })
                    });
                    causes.push(match on {
                        Some(e) => {
                            let t = w.ty(&e.unit).unwrap();
                            let same = built.is_some_and(|b| b.id as u16 == e.unit.unit_type);
                            let owner = if e.owner == OWNER_SELF { "own" } else if e.owner == OWNER_NEUTRAL { "neutral" } else if e.remembered { "remembered_enemy" } else { "enemy" };
                            if same && top_left(&e.unit, t) == (x, y) {
                                format!("same_building_same_tile(completed={})", e.unit.has(uranium_proto::flags::COMPLETED))
                            } else if t.is_resource() {
                                "blocked_by_resource".into()
                            } else {
                                format!("blocked_by_{owner}_building")
                            }
                        }
                        None => "blocked_by_remembered_resource".into(),
                    });
                }
            }
        }
        if causes.is_empty() && a.loc_class == "cc_slot" {
            causes.push("near_resource".into());
        }
        if causes.is_empty() {
            causes.push("mask_other".into());
        }
    }
    causes.sort();
    causes.dedup();
    let shift = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)].into_iter().find(|&(dx, dy)| ok_at(x + dx, y + dy));
    let shift = shift.map_or(String::new(), |(dx, dy)| format!(" (passes at {dx:+},{dy:+})"));
    format!("{}{shift}", causes.join("+"))
}

fn prereq_fail(a: &uranium_env::actions::Action, w: &uranium_env::features::world::World) -> Option<&'static str> {
    let t = uranium_env::types::TypeTables::get();
    let me = &w.frame.self_state;
    let afford = |m: i32, g: i32| me.minerals >= m && me.gas >= g;
    match a.command.as_str() {
        "Train" => {
            let u = t.unit(a.unit_type?)?;
            let supply = uranium_env::features::masks::train_supply_ok(w, a);
            if !afford(u.minerals, u.gas) {
                Some("afford")
            } else if !supply {
                Some("supply")
            } else {
                None
            }
        }
        "Research" => {
            let tech = a.tech?;
            let r = &t.techs[tech as usize];
            if me.tech & (1u64 << tech) != 0 {
                Some("researched")
            } else if me.researching & (1u64 << tech) != 0 {
                Some("researching")
            } else if !afford(r.minerals, r.gas) {
                Some("afford")
            } else {
                None
            }
        }
        "Upgrade" => {
            let up = a.upgrade?;
            let u = &t.upgrades[up as usize];
            let level = me.upgrades.get(up as usize).copied().unwrap_or(0) as i32;
            if level >= u.max_repeats {
                Some("max_level")
            } else if me.upgrading & (1u64 << up) != 0 {
                Some("upgrading")
            } else if !afford(u.minerals(level + 1), u.gas(level + 1)) {
                Some("afford")
            } else {
                None
            }
        }
        "Build" => {
            let u = t.unit(a.unit_type?)?;
            if !afford(u.minerals, u.gas) { Some("afford") } else { None }
        }
        _ => None,
    }
}

pub fn label_check(game: &ShardGame, st: &mut LabelStats, maps: &MapStore) -> Result<()> {
    let mut walker = LabelWalker::new(&game.game_info, family_index(&game.meta.map_family), maps.get(&game.meta.map_hash)?.as_deref(), game.meta.enemy_random)?;
    let k = CAPABILITIES.len();
    let mut prev: Option<&uranium_proto::RawFrame> = None;
    let mut seen: HashMap<u32, (u32, u8, u16)> = HashMap::new();
    let tt = uranium_env::types::TypeTables::get();
    let type_name = |t: u16| tt.unit(t).map_or("?", |t| t.name.as_str());
    let mut g = (game.meta.src.clone(), format!("{} {} map {} {}x{}", game.meta.replay_sha256.chars().take(12).collect::<String>(), game.meta.matchup, game.meta.map_hash.chars().take(8).collect::<String>(), game.game_info.map_size_tiles.0, game.game_info.map_size_tiles.1), 0usize, 0usize, 0usize, 0usize);
    for (si, s) in game.steps.iter().enumerate() {
        let (o, label, why) = walker.step_with_memory(&s.raw_frame, &s.action_label, None, s.memory_action.as_deref());
        let a = action_table().actions.get(label.action as usize).filter(|_| !s.action_label.is_unknown());
        let name = if s.action_label.is_unknown() { "Unknown".to_string() } else { a.map_or_else(|| format!("#{}", label.action), |a| a.name.clone()) };
        *st.totals.entry(name.clone()).or_default() += 1;
        if let Some(a) = a.filter(|a| a.uses_select && matches!(why, None | Some("select_cap"))) {
            let c = CAPABILITIES.iter().position(|c| *c == a.select_cap).unwrap();
            let mut types: Vec<&str> = s
                .action_label
                .selected
                .iter()
                .filter_map(|id| o.entity_ids.iter().position(|e| e == id))
                .filter(|&i| !o.masks.unit_cap[i * k + c])
                .filter_map(|i| s.raw_frame.units.iter().find(|u| u.id == o.entity_ids[i]))
                .filter_map(|u| uranium_env::types::TypeTables::get().unit(u.unit_type).map(|t| t.name.as_str()))
                .collect();
            types.sort_unstable();
            types.dedup();
            if !types.is_empty() {
                let kind = if why.is_some() { "void" } else { "kept" };
                *st.cap_detail.entry(format!("{name} <- {} ({kind})", types.join("+"))).or_default() += 1;
            }
        }
        if let Some(w @ ("target_unit_cap" | "target_unit_missing")) = why {
            let target = s.action_label.target_unit.and_then(|id| s.raw_frame.units.iter().find(|u| u.id == id));
            let me = game.game_info.self_player;
            let desc = target.map_or_else(
                || "not_in_frame".to_string(),
                |u| {
                    let owner = if u.owner == me { "self" } else if game.game_info.players.iter().any(|p| p.id == u.owner) { "enemy" } else { "neutral" };
                    let ty = uranium_env::types::TypeTables::get().unit(u.unit_type).map_or("?", |t| t.name.as_str());
                    format!("{owner} {ty}")
                },
            );
            *st.type_detail.entry(format!("{name} [{w}] -> {desc}")).or_default() += 1;
        }
        if let (Some(_), Some(w @ ("target_unit_missing" | "target_unit_cap"))) = (a, why) {
            let id = s.action_label.target_unit;
            let in_frame = id.and_then(|id| s.raw_frame.units.iter().find(|u| u.id == id));
            let cause = match (id, in_frame) {
                (None, _) => "no_target_id".to_string(),
                (Some(_), Some(u)) if w == "target_unit_missing" => format!("in_frame_not_entity({})", type_name(u.unit_type)),
                (Some(_), Some(u)) => {
                    let done = u.has(uranium_proto::flags::COMPLETED);
                    let loaded = u.has(uranium_proto::flags::LOADED) || u.loaded_in.is_some();
                    format!("present_ineligible({}, completed={done}, loaded={loaded})", type_name(u.unit_type))
                }
                (Some(id), None) => {
                    let later = game.steps[si + 1..].iter().take(3).any(|n| n.raw_frame.units.iter().any(|u| u.id == id));
                    match seen.get(&id) {
                        Some(&(f, _, t)) => format!("seen_before({}, {}{})", type_name(t), age_bucket(s.raw_frame.frame.saturating_sub(f)), if later { ", back later" } else { "" }),
                        None if later => "first_seen_later".to_string(),
                        None => "never_seen".to_string(),
                    }
                }
            };
            *st.cause_detail.entry(format!("{name} [{w}] -> {cause}")).or_default() += 1;
        }
        if let (Some(a), Some("target_loc")) = (a, why) {
            let w = uranium_env::features::world::World::new(&s.raw_frame, &walker.mem, &walker.map);
            let ty = a.unit_type.map_or("-", type_name);
            let me = game.game_info.self_player;
            let built = s.action_label.target_tile.zip(a.unit_type).is_some_and(|((x, y), ut)| {
                game.steps[si + 1..].iter().take(200).any(|n| {
                    n.raw_frame.units.iter().any(|u| {
                        u.owner == me && u.unit_type == ut && tt.unit(ut).is_some_and(|t| uranium_env::features::world::top_left(u, t) == (x as i32, y as i32))
                    })
                })
            });
            let cause = loc_cause(&w, &o, a, s.action_label.target_tile);
            *st.cause_detail.entry(format!("{name} [target_loc] {ty} -> {cause}{}", if built { " [built there]" } else { " [not built there]" })).or_default() += 1;
        }
        if let (Some(a), Some("type_mask")) = (a, why) {
            let world = |f| uranium_env::features::world::World::new(f, &walker.mem, &walker.map);
            let cond = prereq_fail(a, &world(&s.raw_frame)).unwrap_or("producer_or_required");
            let age = match prev.and_then(|p| prereq_fail(a, &world(p))) {
                Some(c) if c == cond => "old",
                _ if cond == "producer_or_required" => "-",
                _ => "new",
            };
            *st.type_detail.entry(format!("{name} <- {cond} ({age})")).or_default() += 1;
        }
        let unknown = why == Some(UNKNOWN_REASON);
        g.2 += !unknown as usize;
        if let Some(w) = why {
            g.3 += !unknown as usize;
            g.4 += (w == "target_unit_missing") as usize;
            g.5 += (w == "target_loc") as usize;
            *st.fails.entry((name, w)).or_default() += 1;
        }
        prev = Some(&s.raw_frame);
        for u in &s.raw_frame.units {
            seen.insert(u.id, (s.raw_frame.frame, u.owner, u.unit_type));
        }
    }
    st.per_game.push(g);
    Ok(())
}
