use anyhow::Result;
use crate::relabel::MemoryCheck;
use crate::shard::ShardGame;
use crate::{Replay, header};
use super::{MIN_GAME_MS, S1_MIN_APM, S2_MIN_APM, MAX_MAP_TILES, View, race_letter};
use super::args::{BRANCH_SRC, DISTILL_SRC};

pub fn filter(rep: &Replay, family: &str, bot: bool, any_length: bool) -> std::result::Result<Vec<View>, String> {
    let active = rep.active_players();
    if active.len() != 2 {
        return Err(format!("{} active players", active.len()));
    }
    if !any_length && (rep.header.frames as u64) * crate::MS_PER_FRAME < MIN_GAME_MS {
        return Err("shorter than 5 minutes".into());
    }
    if rep.header.map_width > MAX_MAP_TILES || rep.header.map_height > MAX_MAP_TILES {
        return Err("map larger than 128 tiles".into());
    }
    let mut views = Vec::new();
    for (i, (p, s)) in active.iter().enumerate() {
        if p.race != header::RACE_TERRAN {
            continue;
        }
        let other = &active[1 - i].0;
        let stage = if bot {
            "B"
        } else if family != "other" && s.apm >= S2_MIN_APM {
            "S2"
        } else if s.apm >= S1_MIN_APM {
            "S1"
        } else {
            continue;
        };
        views.push(View { perspective: p.id, stage, apm: s.apm, matchup: format!("Tv{}", race_letter(other.race)), enemy_random: other.race == header::RACE_RANDOM });
    }
    if views.is_empty() {
        return Err("no Terran view passes the APM bar".into());
    }
    Ok(views)
}

pub fn filter_rules(src: &str, relabelled: bool) -> (bool, bool) {
    let bot = matches!(src, "bot" | "botgen" | "dagger" | BRANCH_SRC | DISTILL_SRC);
    (bot, relabelled || src == BRANCH_SRC || src == DISTILL_SRC)
}

pub fn distill_filter_reason(m: &MemoryCheck) -> String {
    let mut s = format!("{DISTILL_FILTER_PREFIX} {} (041-R28, 044-R2): {} decisions, {} checked, {} mismatched", m.status, m.decisions, m.checked, m.mismatched);
    if let Some(r) = &m.reason {
        s.push_str(": ");
        s.push_str(r);
    }
    s
}

pub const DISTILL_FILTER_PREFIX: &str = "decision_log";

pub fn invalid_label_filter(game: &ShardGame, pre: Option<&uranium_env::features::MapPrecompute>) -> Result<Option<String>> {
    let v = uranium_env::features::label::label_validity_with_memory(
        &game.game_info,
        uranium_env::maps::family_index(&game.meta.map_family),
        pre,
        game.meta.enemy_random,
        game.steps.iter().map(|s| (&s.raw_frame, &s.action_label, s.memory_action.as_deref())),
    )?;
    Ok(v.rejected().then(|| v.filter_reason()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd;

    #[test]
    fn dagger_views_have_no_length_bar_041_r22() {
        let player = |id: u8, race: u8| crate::Player { index: id, slot: id as u16, id, kind: 2, race, team: id, name: vec![b'p', b'0' + id] };
        let train = |p: u8| cmd::Cmd { frame: 10, player: p, op: cmd::op::TRAIN, data: vec![7, 0] };
        let short = Replay {
            format: crate::Format::Legacy,
            header: crate::Header { frames: 3 * 60 * 24, map_width: 128, map_height: 112, game_type: 2, map_name: "t".into(), players: vec![player(0, header::RACE_TERRAN), player(1, header::RACE_ZERG)] },
            raw_header: vec![],
            blocks: vec![cmd::Block { frame: 10, cmds: vec![train(0), train(1)] }],
            map: vec![],
            unit_limit: None,
        };
        assert_eq!(filter(&short, "fighting_spirit", true, false).unwrap_err(), "shorter than 5 minutes");
        let views = filter(&short, "fighting_spirit", true, true).unwrap();
        assert!(!views.is_empty() && views.iter().all(|v| v.stage == "B"));
    }
}
