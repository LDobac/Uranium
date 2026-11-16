use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use anyhow::{Context, Result};
use uranium_env::scout::{PseudoMode, ScoutMetrics, seconds};
use crate::league::AgentKind;
use crate::learner::{HEAD_NAMES, UpdateStats, spatial_columns};
use crate::trajectory::{GameResult, SegmentQueue};
use super::Shared;

pub(super) const GAMES_HEADER: &str = "game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,replay_reason,end_reason,r_scout,r_survive,ladder_group,scout_main_s,scout_types_4m,scout_types_6m,barracks_s,bunker_s,scv_4m,judge_scv,slot_self,slot_enemy,start_hit,scv_min_1m,army_s\n";

pub(super) fn slots_start_hit(slots: Option<[u8; 2]>) -> Option<bool> {
    slots.map(|[a, b]| uranium_runner::start_slots::start_hit(a, b))
}

pub(super) fn start_columns(slots: Option<[u8; 2]>, result: Option<&GameResult>) -> String {
    let m: Option<ScoutMetrics> = result.and_then(|r| r.scout);
    let (a, b, hit) = match slots {
        Some([a, b]) => (a.to_string(), b.to_string(), (uranium_runner::start_slots::start_hit(a, b) as u8).to_string()),
        None => (String::new(), String::new(), String::new()),
    };
    [
        a,
        b,
        hit,
        m.and_then(|m| m.scv_min_1m).map_or(String::new(), |n| n.to_string()),
        m.and_then(|m| m.army_frame).map_or(String::new(), |f| format!("{:.3}", seconds(f))),
    ]
    .join(",")
}

pub(super) fn scout_columns(mode: PseudoMode, result: Option<&GameResult>, ladder_group: Option<&str>) -> String {
    let secs = |f: Option<u32>| f.map_or(String::new(), |f| format!("{:.1}", seconds(f)));
    let (r_scout, r_survive) = match (mode, result) {
        (PseudoMode::Scout, Some(r)) => (r.scout_sum.to_string(), r.survive_sum.to_string()),
        _ => (String::new(), String::new()),
    };
    let m: Option<ScoutMetrics> = result.and_then(|r| r.scout);
    let count = |x: fn(&ScoutMetrics) -> u32| m.as_ref().map_or(String::new(), |m| x(m).to_string());
    [
        r_scout,
        r_survive,
        ladder_group.unwrap_or_default().to_string(),
        secs(m.and_then(|m| m.main_frame)),
        count(|m| m.types_4m),
        count(|m| m.types_6m),
        secs(m.and_then(|m| m.barracks_frame)),
        secs(m.and_then(|m| m.bunker_frame)),
        m.and_then(|m| m.scv_4m).map_or(String::new(), |n| n.to_string()),
    ]
    .join(",")
}

fn train_header() -> String {
    let mut h = String::from("time_s,agent,update,policy_version,agent_steps,games,steps,loss,policy,value,kl,kl_coef,rho_mean,adv_mean,lag_mean");
    for n in HEAD_NAMES {
        h += &format!(",kl_{n}");
    }
    for n in HEAD_NAMES {
        h += &format!(",entropy_{n}");
    }
    h += ",dropped_lag,dropped_full,revoked,used_aborted,winner_disagree,aborted_games,steps_per_s,stage,win_open,trained_steps,queue_wait_s";
    h += DIAG_COLUMNS;
    h += AUX_COLUMNS;
    for c in spatial_columns() {
        h += &format!(",{c}");
    }
    for r in crate::batch::REF_RACES {
        h += &format!(",kl_ref_{r}");
    }
    h += "\n";
    h
}

fn kl_ref_columns(s: &UpdateStats) -> String {
    s.kl_ref.iter().map(|x| format!(",{}", x.map_or(String::new(), |v| v.to_string()))).collect()
}

const AUX_COLUMNS: &str = ",aux_priv,aux_jepa,jepa_tgt_std,aux_jepa_slot1,aux_jepa_slot2,aux_jepa_slot3";

fn aux_columns(s: &UpdateStats) -> String {
    let opt = |x: Option<f32>| x.map_or(String::new(), |v| v.to_string());
    let mut row = format!(",{},{},{}", opt(s.aux_priv), opt(s.aux_jepa), opt(s.jepa_tgt_std));
    for v in s.aux_jepa_slots {
        row += &format!(",{}", opt(v));
    }
    let n = spatial_columns().len();
    match &s.spatial {
        Some(v) if v.len() == n => v.iter().for_each(|x| row += &format!(",{x}")),
        _ => row += &",".repeat(n),
    }
    row
}

const DIAG_COLUMNS: &str = ",adv_rms,adv_rms_win,adv_rms_build,adv_rms_units,adv_rms_value,value_loss_win,value_loss_build,value_loss_units,value_loss_value,kl_p99,kl_max,sil_buffer,sil_lanes,sil_loss,sil_pos_frac";

fn diag_columns(s: &UpdateStats, sil: Option<(usize, usize)>) -> String {
    let opt = |x: Option<f32>| x.map_or(String::new(), |v| v.to_string());
    let mut row = format!(",{}", s.adv_rms);
    for v in s.adv_rms_channels.iter().chain(&s.value_loss_channels) {
        row += &format!(",{v}");
    }
    let (buffer, lanes) = sil.map_or((String::new(), String::new()), |(b, l)| (b.to_string(), l.to_string()));
    let (loss, pos) = if sil.is_some() { (opt(s.sil_loss), opt(s.sil_pos_frac)) } else { (String::new(), String::new()) };
    row += &format!(",{},{},{buffer},{lanes},{loss},{pos}", opt(s.kl_p99), opt(s.kl_max));
    row
}

pub(super) fn open_train_csv(path: &Path) -> Result<std::fs::File> {
    open_csv(path, &train_header())
}

pub(super) fn open_csv(path: &Path, header: &str) -> Result<std::fs::File> {
    let stem = path.file_stem().map_or_else(|| "log".to_string(), |s| s.to_string_lossy().into_owned());
    let name = path.file_name().map_or_else(|| stem.clone(), |s| s.to_string_lossy().into_owned());
    let mut target = path.to_path_buf();
    if let Ok(f) = std::fs::File::open(path) {
        let mut first = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(f), &mut first)?;
        if !first.is_empty() && first.trim_end() != header.trim_end() {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let old = path.with_file_name(format!("{stem}-{secs}.csv"));
            match std::fs::rename(path, &old) {
                Ok(()) => uranium_store::elog!("[rl] {name} has other columns: kept as {}, new file started (042-R2)", old.display()),
                Err(e) => {
                    target = path.with_file_name(format!("{stem}-{secs}-new.csv"));
                    uranium_store::elog!("[rl] WARNING: {name} has other columns and cannot be moved ({e}); writing {} instead (042-R2)", target.display());
                }
            }
        }
    }
    let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(&target).with_context(|| format!("opening {}", target.display()))?;
    if csv.metadata()?.len() == 0 {
        csv.write_all(header.as_bytes())?;
    }
    Ok(csv)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_row(csv: &mut std::fs::File, started: Instant, agent: AgentKind, s: &UpdateStats, shared: &Shared, queues: &[Arc<SegmentQueue>], version: u64, steps_per_s: f64, trained_steps: u64, queue_wait_s: f64, sil: Option<(usize, usize)>) -> Result<()> {
    let q = queues[agent.index()].stats();
    let mut row = format!(
        "{:.0},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        started.elapsed().as_secs_f64(),
        agent.name(),
        s.update,
        version,
        shared.agent_steps.load(Ordering::SeqCst),
        shared.games_done.load(Ordering::SeqCst),
        s.steps,
        s.loss,
        s.policy,
        s.value,
        s.kl,
        s.kl_coef,
        s.rho_mean,
        s.adv_mean,
        s.lag_mean
    );
    for v in s.kl_heads {
        row += &format!(",{v}");
    }
    for v in s.entropy_heads {
        row += &format!(",{v}");
    }
    let stage = shared.league.lock().unwrap().stage;
    row += &format!(
        ",{},{},{},{},{},{},{:.2},{:?},{},{},{:.3}",
        q.dropped_lag,
        q.dropped_full,
        q.revoked,
        q.used_aborted,
        shared.disagree.load(Ordering::SeqCst),
        shared.aborted.load(Ordering::SeqCst),
        steps_per_s,
        stage,
        s.win_open,
        trained_steps,
        queue_wait_s
    );
    row += &diag_columns(s, sil);
    row += &aux_columns(s);
    row += &kl_ref_columns(s);
    row += "\n";
    csv.write_all(row.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::testkit::result;

    #[test]
    fn train_csv_header_columns_042_r2() {
        assert_eq!(train_header().trim().split(',').count(), 15 + 12 + 11 + 15 + 6 + 36 + 4);
        assert!(train_header().contains(
            ",win_open,trained_steps,queue_wait_s,adv_rms,adv_rms_win,adv_rms_build,adv_rms_units,adv_rms_value,value_loss_win,value_loss_build,value_loss_units,value_loss_value,kl_p99,kl_max,sil_buffer,sil_lanes,sil_loss,sil_pos_frac,aux_priv,"
        ));
    }

    #[test]
    fn train_csv_aux_and_spatial_columns_042_r21() {
        let layers = "stem,c0,c1,c2,c3,r0,r1,r2,r3,r4,r5,d0,d1,d2,d3,b0,b1";
        let pos: Vec<String> = layers.split(',').map(|l| format!("sp_pos_{l}")).collect();
        let dead: Vec<String> = layers.split(',').map(|l| format!("sp_dead_{l}")).collect();
        let tail = format!(",sil_pos_frac,aux_priv,aux_jepa,jepa_tgt_std,aux_jepa_slot1,aux_jepa_slot2,aux_jepa_slot3,{},{},sp_const_s8,sp_const_pool,kl_ref_zerg,kl_ref_terran,kl_ref_protoss,kl_ref_random", pos.join(","), dead.join(","));
        assert!(train_header().trim().ends_with(&tail));
        let off = aux_columns(&UpdateStats::default());
        assert_eq!(off, ",".repeat(6 + 36));
        let mut spatial: Vec<f32> = (0..17).map(|i| i as f32 / 32.0).collect();
        spatial.extend((0..17).map(|i| i as f32));
        spatial.extend([3.0, 5.0]);
        let s = UpdateStats { aux_priv: Some(0.5), aux_jepa: Some(0.25), jepa_tgt_std: Some(0.75), aux_jepa_slots: [Some(0.125), Some(0.375), None], spatial: Some(spatial.clone()), ..Default::default() };
        let row = aux_columns(&s);
        let want: Vec<String> = ["0.5", "0.25", "0.75", "0.125", "0.375", ""].iter().map(|x| x.to_string()).chain(spatial.iter().map(|x| x.to_string())).collect();
        assert_eq!(row, format!(",{}", want.join(",")));
        assert!(row.ends_with(",16,3,5"));
        let s = UpdateStats { spatial: None, ..s };
        assert!(aux_columns(&s).ends_with(&",".repeat(36)) && aux_columns(&s).starts_with(",0.5,0.25,0.75,0.125,0.375,,"));
        let js: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0,"aux_priv":0.5,"aux_jepa":null,"jepa_tgt_std":null,"aux_jepa_slots":[0.25,null,null],"spatial":null}"#).unwrap();
        assert_eq!((js.aux_priv, js.aux_jepa, js.aux_jepa_slots, js.spatial), (Some(0.5), None, [Some(0.25), None, None], None));
    }

    #[test]
    fn train_csv_kl_ref_columns_044_r9() {
        assert_eq!(kl_ref_columns(&UpdateStats::default()), ",,,,");
        let s = UpdateStats { kl_ref: [Some(0.5), None, Some(0.25), None], ..Default::default() };
        assert_eq!(kl_ref_columns(&s), ",0.5,,0.25,");
        let js: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0,"kl_ref":[0.125,null,null,2.0]}"#).unwrap();
        assert_eq!(js.kl_ref, [Some(0.125), None, None, Some(2.0)]);
        let old: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0}"#).unwrap();
        assert_eq!(old.kl_ref, [None; 4]);
    }

    #[test]
    fn train_csv_diag_columns_042_r2() {
        let s = UpdateStats { adv_rms: 0.5, adv_rms_channels: [0.25, 1.0, 2.0, 0.125], value_loss_channels: [0.5, 0.75, 0.0, 3.0], kl_p99: Some(0.375), kl_max: Some(4.0), sil_loss: Some(0.5), sil_pos_frac: Some(0.25), ..Default::default() };
        assert_eq!(diag_columns(&s, None), ",0.5,0.25,1,2,0.125,0.5,0.75,0,3,0.375,4,,,,");
        assert_eq!(diag_columns(&s, Some((100, 4))), ",0.5,0.25,1,2,0.125,0.5,0.75,0,3,0.375,4,100,4,0.5,0.25");
        let none = UpdateStats { sil_loss: Some(0.0), ..Default::default() };
        assert_eq!(diag_columns(&none, Some((2, 0))), ",0,0,0,0,0,0,0,0,0,,,2,0,0,");
        assert_eq!(diag_columns(&none, None).split(',').count() - 1, DIAG_COLUMNS.split(',').count() - 1);
        let old: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0}"#).unwrap();
        assert_eq!((old.kl_p99, old.sil_loss, old.adv_rms), (None, None, 0.0));
        assert_eq!((old.aux_priv, old.aux_jepa_slots, old.spatial), (None, [None; 3], None));
        let new: UpdateStats = serde_json::from_str(r#"{"update":1,"steps":2,"loss":0.1,"policy":0.1,"value":0.0,"kl":0.0,"kl_coef":0.1,"kl_heads":[0,0,0,0,0,0],"entropy_heads":[0,0,0,0,0,0],"rho_mean":1,"adv_mean":0,"lag_mean":0,"win_open":0,"adv_rms":0.5,"adv_rms_channels":[1,2,3,4],"value_loss_channels":[1,2,3,4],"kl_p99":null,"kl_max":2.5,"sil_loss":null,"sil_pos_frac":null,"grad_norm":1.0}"#).unwrap();
        assert_eq!((new.kl_p99, new.kl_max, new.adv_rms_channels), (None, Some(2.5), [1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn train_csv_with_other_columns_is_kept_aside_042_r2() {
        let dir = std::env::temp_dir().join(format!("uranium-rl-csv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train.csv");
        std::fs::write(&path, "time_s,agent\n1,main\n").unwrap();
        drop(open_train_csv(&path).unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), train_header());
        let kept: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).filter(|n| n.starts_with("train-")).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read_to_string(dir.join(&kept[0])).unwrap(), "time_s,agent\n1,main\n");
        let mut f = open_train_csv(&path).unwrap();
        f.write_all(b"row\n").unwrap();
        drop(f);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), format!("{}row\n", train_header()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn games_csv_with_other_columns_is_kept_aside_042_r14() {
        let dir = std::env::temp_dir().join(format!("uranium-rl-games-csv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("games.csv");
        let old = "game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,replay_reason\n1,main\n";
        std::fs::write(&path, old).unwrap();
        drop(open_csv(&path, GAMES_HEADER).unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), GAMES_HEADER);
        let kept: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).filter(|n| n.starts_with("games-")).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read_to_string(dir.join(&kept[0])).unwrap(), old);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn games_csv_header_matches_rows_042_r2() {
        assert_eq!(GAMES_HEADER.trim().split(',').count(), 32);
        assert!(GAMES_HEADER.trim().ends_with(",replay,replay_reason,end_reason,r_scout,r_survive,ladder_group,scout_main_s,scout_types_4m,scout_types_6m,barracks_s,bunker_s,scv_4m,judge_scv,slot_self,slot_enemy,start_hit,scv_min_1m,army_s"));
        assert!(GAMES_HEADER.starts_with("game_id,agent,opponent,class,enemy_race,map,hostd_outcome,outcome,winner_disagree,decisions,r_win,r_build,r_units,r_value,replay,"));
    }

    #[test]
    fn start_columns_042_ac9() {
        let m = ScoutMetrics { scv_min_1m: Some(3), army_frame: Some(4_452), ..Default::default() };
        let slots = |seed| Some(uranium_runner::start_slots::start_slots(seed, 0));
        assert_eq!(start_columns(slots(43_006), Some(&result(1, Some(m)))), "1,0,1,3,185.500", "c0 g6: (1, 0) is a hit");
        let late = ScoutMetrics { army_frame: Some(5_401), ..Default::default() };
        assert_eq!(start_columns(None, Some(&result(1, Some(late)))), ",,,,225.042", "the exact value select_army_before_s compares");
        assert_eq!(start_columns(slots(43_001), Some(&result(1, Some(ScoutMetrics::default())))), "2,3,0,,", "c0 g1: a miss, nothing reached");
        assert_eq!(start_columns(slots(43_001), None), "2,3,0,,", "no session report");
        assert_eq!(start_columns(None, None), ",,,,", "no seed");
    }

    #[test]
    fn scout_columns_042_r2() {
        let m = ScoutMetrics { main_frame: Some(3_000), types_4m: 2, types_6m: 5, barracks_frame: Some(2_412), bunker_frame: None, scv_4m: Some(14), scv_last: Some(9), ..Default::default() };
        let r = result(1, Some(m));
        assert_eq!(scout_columns(PseudoMode::Scout, Some(&r), Some("macro")), "0.6,0.375,macro,125.0,2,5,100.5,,14");
        assert_eq!(scout_columns(PseudoMode::Z, Some(&r), Some("rush")), ",,rush,125.0,2,5,100.5,,14", "z: r_scout / r_survive empty");
        assert_eq!(scout_columns(PseudoMode::Scout, Some(&result(2, Some(ScoutMetrics::default()))), None), "0.6,0.375,,,0,0,,,", "nothing reached");
        assert_eq!(scout_columns(PseudoMode::Scout, None, None), ",,,,,,,,", "no session report");
        assert_eq!(scout_columns(PseudoMode::Z, None, None).split(',').count(), 9);
    }
}
