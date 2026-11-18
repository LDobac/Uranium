use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::Result;
use crate::label::{Coverage, Labeler};
use crate::play::{PlayConfig, PlayResult, play_observe, play_with, slot_of};
use crate::relabel::MemoryCheck;
use crate::shard::{Step, result_source};
use crate::{Replay, cmd};
use uranium_env::maps::maps_dir;
use super::{View, Outcome};
use super::filter::distill_filter_reason;

pub(crate) fn own_labels(rep: &Replay, perspective: u8, labeler: &Labeler, units_at: &HashMap<u32, HashMap<u32, crate::label::Own>>) -> Result<(Vec<crate::label::Label>, Coverage)> {
    let cmds: Vec<cmd::Cmd> = rep
        .cmds()
        .filter(|c| c.player == perspective)
        .map(|c| cmd::to_legacy(c, rep.unit_limit))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|c| cmd::openbw(c.op) == cmd::OpenBw::Supported)
        .collect();
    let empty = HashMap::new();
    Ok(labeler.label(&cmds, rep.header.frames, |f| units_at.get(&f).unwrap_or(&empty)))
}

pub fn judge(play: &PlayResult, recorded_winner: Option<u8>, me: u8) -> (i8, &'static str) {
    match play.decided_winner().or(recorded_winner.map(|w| (w, result_source::RECORDED))) {
        Some((w, source)) => (if w == me { 1 } else { -1 }, source),
        None => (0, result_source::NONE),
    }
}

pub(super) fn recorded_slot(rep: &Replay, winner: Option<&str>) -> Option<u8> {
    let w = winner?.as_bytes();
    let active = rep.active_players();
    let exact = active.iter().find(|(p, _)| p.name == w);
    let p = exact.or_else(|| match active.iter().filter(|(p, _)| p.name.eq_ignore_ascii_case(w)).collect::<Vec<_>>()[..] {
        [one] => Some(one),
        _ => None,
    })?;
    slot_of(rep, p.0.id).ok()
}

pub struct ViewGame {
    pub steps: Vec<Step>,
    pub coverage: Coverage,
    pub result: i8,
    pub result_source: &'static str,
    pub game_info: uranium_proto::GameInfo,
    pub repeats_dropped: Option<u32>,
    pub memory: Option<crate::relabel::MemoryCheck>,
}

#[derive(Clone, Copy)]
pub(super) struct ViewOptions<'a> {
    pub(super) until_frame: Option<u32>,
    pub(super) time_limit: Option<u32>,
    pub(super) relabel: Option<uranium_scripted::T0Build>,
    pub(super) map_hash: &'a str,
    pub(super) memory: Option<&'a [crate::relabel::DecisionLogRead]>,
    pub(super) distill: bool,
}

pub(super) fn build_view(rep: &Replay, view: &View, cfg: &PlayConfig, labeler: &Labeler, recorded_winner: Option<u8>, o: ViewOptions) -> Result<std::result::Result<ViewGame, Outcome>> {
    let (until_frame, time_limit) = (o.until_frame, o.time_limit);
    let me = slot_of(rep, view.perspective)?;
    let last = until_frame.map_or(rep.header.frames, |u| u.saturating_add(32)).min(rep.header.frames);
    let mut own_at: HashMap<u32, std::collections::HashSet<u32>> = HashMap::new();
    let bound = o.memory.map(|reads| crate::relabel::bind_logs(reads, me, rep.header.frames));
    if let Some(reason) = other_player_view(o.relabel.is_some() || o.distill, bound.as_ref()) {
        return Ok(Err(Outcome::Filtered { reason }));
    }
    if o.distill {
        let reason = match &bound {
            Some(Ok(_)) => None,
            Some(Err(m)) => Some(distill_filter_reason(m)),
            None => Some(distill_filter_reason(&MemoryCheck::without_log(&crate::relabel::DecisionLogRead::Missing))),
        };
        if let Some(reason) = reason {
            return Ok(Err(Outcome::Filtered { reason }));
        }
    }
    let check_frames = match &bound {
        Some(Ok(log)) => crate::relabel::decision_frames(&log.decisions, last),
        _ => Vec::new(),
    };
    let (mut a, mut labels, coverage, repeats_dropped) = match o.relabel {
        None => {
            let mut units_at: HashMap<u32, HashMap<u32, crate::label::Own>> = HashMap::new();
            let a = play_with(rep, view.perspective, cfg, &check_frames, &mut |f, _| {
                let own: HashMap<u32, crate::label::Own> = f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, crate::label::Own::from_state(u))).collect();
                if check_frames.binary_search(&f.frame).is_ok() {
                    own_at.insert(f.frame, own.keys().copied().collect());
                }
                units_at.insert(f.frame, own);
            })?;
            let (labels, coverage) = own_labels(rep, view.perspective, labeler, &units_at)?;
            (a, labels, coverage, None)
        }
        Some(build) => {
            let mut stops = crate::relabel::teacher_frames(last);
            stops.extend(&check_frames);
            stops.sort_unstable();
            stops.dedup();
            let shadow: std::cell::RefCell<Option<crate::relabel::Shadow>> = std::cell::RefCell::new(None);
            let failed: std::cell::RefCell<Option<anyhow::Error>> = std::cell::RefCell::new(None);
            let a = play_observe(
                rep,
                view.perspective,
                cfg,
                &stops,
                &mut |gi| match write_map_precompute(o.map_hash, gi, &rep.map) {
                    Ok(pre) => *shadow.borrow_mut() = Some(crate::relabel::Shadow::new(gi, build, uranium_scripted::MapHints { base_locs: pre.base_locs, chokes: pre.chokes }, view.perspective, last)),
                    Err(e) => *failed.borrow_mut() = Some(e),
                },
                &mut |f, _| {
                    if check_frames.binary_search(&f.frame).is_ok() {
                        own_at.insert(f.frame, f.units.iter().filter(|u| u.owner == me).map(|u| u.id).collect());
                    }
                    if let Some(s) = shadow.borrow_mut().as_mut() {
                        s.observe(f);
                    }
                },
            )?;
            if let Some(e) = failed.into_inner() {
                return Err(e.context("map precompute for the relabelling teacher (041-R22)"));
            }
            let Some(shadow) = shadow.into_inner() else {
                return Ok(Err(Outcome::Error { reason: "bridge sent no GameInfo: no teacher to relabel with (041-R22)".into() }));
            };
            let (labels, coverage) = shadow.labels(labeler, rep.header.frames);
            (a, labels, coverage, Some(shadow.repeats_dropped))
        }
    };
    a.drop_time_limit_leave(time_limit);
    let reasons = a.desync_reasons(recorded_winner);
    if !reasons.is_empty() {
        return Ok(Err(Outcome::Desync { reasons }));
    }
    cut_after(&mut labels, until_frame, |l| l.frame);
    let limit = until_frame.map_or(last, |u| u.min(last));
    let decision_steps = match step_labels(o.distill, o.relabel.is_some(), bound.as_ref(), &own_at, last, limit, rep.header.frames, &mut labels) {
        Ok(n) => n,
        Err(reason) => return Ok(Err(Outcome::Filtered { reason })),
    };
    let frames: Vec<u32> = labels.iter().map(|l| l.frame).collect();
    let mut by_frame: HashMap<u32, (uranium_proto::RawFrame, Option<uranium_proto::PrivilegedSummary>)> = HashMap::new();
    let res = play_with(rep, view.perspective, cfg, &frames, &mut |f, p| {
        if frames.binary_search(&f.frame).is_ok() {
            by_frame.insert(f.frame, (f.clone(), p.cloned()));
        }
    })?;
    if a.check_digest != res.check_digest {
        return Ok(Err(Outcome::Desync { reasons: vec![format!("replays differ: {} vs {}", a.check_digest, res.check_digest)] }));
    }
    let mut steps = Vec::with_capacity(labels.len());
    for l in labels {
        let Some((raw_frame, privileged)) = by_frame.remove(&l.frame) else { break };
        steps.push(Step { raw_frame, privileged, action_label: l, memory_action: None });
    }
    if let Some(&f) = frames.first().filter(|_| steps.is_empty()) {
        return Ok(Err(Outcome::Error { reason: format!("no observation at the first step frame {f} (040-R18)") }));
    }
    let decision_steps = if o.distill { decision_steps.map(|n| n - (frames.len() - steps.len())) } else { decision_steps };
    let mut res = res;
    res.drop_time_limit_leave(time_limit);
    let (result, result_source) = judge(&res, recorded_winner, me);
    let Some(game_info) = res.game_info else {
        return Ok(Err(Outcome::Error { reason: "bridge sent no GameInfo (AMB-040-6)".into() }));
    };
    let memory = bound.map(|b| MemoryCheck { decision_steps, ..attach_memory(&mut steps, b, last, &own_at) });
    if o.distill && steps.iter().any(|s| s.memory_action.is_none()) {
        return Ok(Err(Outcome::Error { reason: "distill step without memory_action (044-R2)".into() }));
    }
    Ok(Ok(ViewGame { steps, coverage, result, result_source, game_info, repeats_dropped, memory }))
}

pub const OTHER_PLAYER_VIEW: &str = "not the logged model's view";

pub fn other_player_view(relabel_or_distill: bool, bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>) -> Option<String> {
    match bound {
        Some(Err(m)) if !relabel_or_distill && m.status == "other_player" => {
            Some(format!("{OTHER_PLAYER_VIEW} (041-R28, 053-R5: players[0] only): {}", m.reason.as_deref().unwrap_or("")))
        }
        _ => None,
    }
}

pub fn decision_steps(
    relabelled: bool,
    bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>,
    own_at: &HashMap<u32, std::collections::HashSet<u32>>,
    last: u32,
    limit: u32,
    end_frame: u32,
    labels: &mut Vec<crate::label::Label>,
) -> Option<usize> {
    let Some(Ok(log)) = bound.filter(|_| !relabelled) else { return None };
    let (checked, mismatched) = crate::relabel::unit_check(&log.decisions, last, own_at);
    MemoryCheck::checked(log.decisions.len(), checked, mismatched).ok().then(|| crate::relabel::add_decision_steps(labels, &log.decisions, limit, end_frame))
}

#[allow(clippy::too_many_arguments)]
pub fn step_labels(
    distill: bool,
    relabelled: bool,
    bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>,
    own_at: &HashMap<u32, std::collections::HashSet<u32>>,
    last: u32,
    limit: u32,
    end_frame: u32,
    labels: &mut Vec<crate::label::Label>,
) -> std::result::Result<Option<usize>, String> {
    if !distill {
        return Ok(decision_steps(relabelled, bound, own_at, last, limit, end_frame, labels));
    }
    let (slots, n) = distill_steps(bound, own_at, last, limit, end_frame)?;
    *labels = slots;
    Ok(Some(n))
}

pub fn distill_steps(
    bound: Option<&Result<&crate::relabel::ModelLog, MemoryCheck>>,
    own_at: &HashMap<u32, std::collections::HashSet<u32>>,
    last: u32,
    limit: u32,
    end_frame: u32,
) -> std::result::Result<(Vec<crate::label::Label>, usize), String> {
    let log = match bound {
        Some(Ok(log)) => log,
        Some(Err(m)) => return Err(distill_filter_reason(m)),
        None => return Err(distill_filter_reason(&MemoryCheck::without_log(&crate::relabel::DecisionLogRead::Missing))),
    };
    let (checked, mismatched) = crate::relabel::unit_check(&log.decisions, last, own_at);
    let m = MemoryCheck::checked(log.decisions.len(), checked, mismatched);
    if !m.ok() {
        return Err(distill_filter_reason(&m));
    }
    Ok(crate::relabel::distill_labels(&log.decisions, limit, end_frame))
}

pub fn attach_memory(steps: &mut [Step], bound: Result<&crate::relabel::ModelLog, crate::relabel::MemoryCheck>, last: u32, own_at: &HashMap<u32, std::collections::HashSet<u32>>) -> crate::relabel::MemoryCheck {
    use crate::relabel::{MemoryCheck, step_memory, unit_check};
    let log = match bound {
        Ok(l) => &l.decisions,
        Err(m) => return m,
    };
    let (checked, mismatched) = unit_check(log, last, own_at);
    let m = MemoryCheck::checked(log.len(), checked, mismatched);
    if m.ok() {
        let frames: Vec<u32> = steps.iter().map(|s| s.raw_frame.frame).collect();
        for (s, mem) in steps.iter_mut().zip(step_memory(log, &frames)) {
            s.memory_action = Some(mem);
        }
    }
    m
}

pub fn memory_log_path(dir: &Path, file: &Path) -> PathBuf {
    let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    dir.join(format!("{stem}.jsonl"))
}

pub fn cut_after<T>(items: &mut Vec<T>, until: Option<u32>, frame: impl Fn(&T) -> u32) {
    if let Some(n) = until {
        items.retain(|i| frame(i) <= n);
    }
}

pub fn write_map_precompute(map_hash: &str, info: &uranium_proto::GameInfo, chk: &[u8]) -> Result<uranium_env::features::MapPrecompute> {
    use uranium_env::features::mapgen;
    let dir = maps_dir();
    if let Some(p) = dir.as_deref().ok().and_then(|d| uranium_env::maps::load_precompute(d, map_hash)) {
        return Ok(p);
    }
    let pre = mapgen::precompute(&info.static_map, &mapgen::chk_resources(chk)?);
    let write = || -> Result<()> {
        let dir = dir?;
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{map_hash}.toml"));
        let tmp = dir.join(format!("{map_hash}.toml.{}", std::process::id()));
        std::fs::write(&tmp, toml::to_string(&pre)?)?;
        std::fs::rename(&tmp, &path).or_else(|_| std::fs::remove_file(&tmp))?;
        Ok(())
    };
    if let Err(e) = write() {
        eprintln!("map precompute for {map_hash}: {e:#}");
    }
    Ok(pre)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use crate::shard::{GameMeta, ShardGame};
    use uranium_env::maps::MapFamilies;
    use crate::pipeline::filter::{filter, DISTILL_FILTER_PREFIX};
    use crate::pipeline::hex;
    use crate::pipeline::args::{DISTILL_SRC, unique_stems};

    #[test]
    fn until_frame_cuts_later_steps_040_r21() {
        let mut v = vec![1u32, 11_519, 11_520, 11_521, 20_000];
        cut_after(&mut v, None, |f| *f);
        assert_eq!(v.len(), 5);
        cut_after(&mut v, Some(11_520), |f| *f);
        assert_eq!(v, vec![1, 11_519, 11_520]);
    }

    #[test]
    fn no_relabel_memory_builds_keep_the_model_view_only_041_r28_053_r5() {
        use crate::relabel::{DecisionLogRead, ModelLog};
        let other = MemoryCheck { status: "other_player", reason: Some("the log is player 0's, this view is player 1's".into()), decisions: 3, checked: 0, mismatched: 0, decision_steps: None };
        let log = ModelLog { decisions: vec![], end_frame: Some(10), player: 0 };
        let got = other_player_view(false, Some(&Err(other.clone()))).expect("the other player's view is filtered");
        assert!(got.starts_with(OTHER_PLAYER_VIEW) && got.contains("player 1's"), "{got}");
        assert_eq!(other_player_view(true, Some(&Err(other))), None, "DAgger relabel and distill builds unchanged");
        assert_eq!(other_player_view(false, Some(&Err(MemoryCheck::without_log(&DecisionLogRead::Missing)))), None, "no log: written without memory as before");
        assert_eq!(other_player_view(false, Some(&Ok(&log))), None);
        assert_eq!(other_player_view(false, None), None);
    }

    #[test]
    fn memory_action_is_all_or_none_per_game_041_r28() {
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log};
        use crate::shard::MemoryAction;
        let steps = || {
            let mut s = crate::shard::tests::game(1).steps;
            for (st, f) in s.iter_mut().zip([1u32, 11, 19]) {
                st.raw_frame.frame = f;
            }
            s
        };
        let m = |action, units: Vec<u32>| MemoryAction { action, delay_bucket: 0, units };
        let complete = |decisions, player| DecisionLogRead::Complete(ModelLog { decisions, end_frame: Some(500), player });
        let run = |read: &DecisionLogRead, own: &HashMap<u32, std::collections::HashSet<u32>>| {
            let mut s = steps();
            let r = attach_memory(&mut s, bind_log(read, 0, 502), 100, own);
            (r, s.into_iter().map(|x| x.memory_action).collect::<Vec<_>>())
        };
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (15, [5].into())]);
        let (r, got) = run(&complete(vec![(9, m(4, vec![5])), (17, m(0, vec![]))], 0), &own);
        assert_eq!((r.status, r.decisions, r.checked, r.mismatched), ("ok", 2, 1, 0));
        assert_eq!(got, vec![Some(vec![]), Some(vec![m(4, vec![5])]), Some(vec![m(0, vec![])])]);
        let (r, got) = run(&complete(vec![(9, m(4, vec![5])), (17, m(4, vec![6]))], 0), &own);
        assert_eq!((r.status, r.checked, r.mismatched), ("unit_mismatch", 2, 1));
        assert!(got.iter().all(Option::is_none));
        let other_game = DecisionLogRead::Complete(ModelLog { decisions: vec![(9, m(4, vec![5]))], end_frame: Some(9_000), player: 0 });
        for (read, status) in [
            (DecisionLogRead::Missing, "missing"),
            (DecisionLogRead::Incomplete("no end line".into()), "incomplete"),
            (complete(vec![(9, m(4, vec![5]))], 1), "other_player"),
            (other_game, "incomplete"),
        ] {
            let (x, got) = run(&read, &own);
            assert_eq!(x.status, status);
            assert!(got.iter().all(Option::is_none));
        }
        let mut s = steps();
        let r = attach_memory(&mut s, bind_log(&complete(vec![(9, m(4, vec![5])), (17, m(4, vec![6]))], 0), 0, 502), 100, &own);
        assert!(unique_stems(&[PathBuf::from("a/replays/g1.rep"), PathBuf::from("a/replays/g2.rep")]).is_ok());
        assert!(unique_stems(&[PathBuf::from("a/replays/g1.rep"), PathBuf::from("b/replays/g1.rep")]).is_err());
        let text = serde_json::to_string(&Outcome::Written { steps: 3, coverage: Default::default(), result: 0, result_source: "none".into(), repeats_dropped: Some(0), memory: Some(r) }).unwrap();
        assert!(text.contains(r#""memory":{"status":"unit_mismatch","decisions":2,"checked":2,"mismatched":1}"#), "{text}");
        assert_eq!(memory_log_path(Path::new("m"), Path::new("x/replays/g12.rep")), Path::new("m").join("g12.jsonl"));
    }

    #[test]
    fn act_log_memory_and_decision_steps_051_r13_041_r28() {
        use crate::act_log::tests::{log_text, obs_line};
        use crate::act_log::{parse_memory_log, read_memory_logs};
        use crate::label::{Label, UNKNOWN_ACTION};
        use crate::relabel::bind_logs;
        let dir = std::env::temp_dir().join(format!("uranium-act-mem-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ours = log_text(
            vec![obs_line(0, 1, Some((0, 0, vec![])), true), obs_line(1, 9, Some((4, 2, vec![5])), true), obs_line(2, 17, Some((7, 1, vec![5, 6])), false), obs_line(3, 25, Some((0, 3, vec![])), false)],
            498,
            0,
            "model:x",
        );
        std::fs::write(dir.join("g3-p0.jsonl"), &ours).unwrap();
        std::fs::write(dir.join("g3-p1.jsonl"), log_text(vec![obs_line(0, 1, None, false)], 498, 1, "t0@3:zerg_rush")).unwrap();
        let logs = read_memory_logs(&dir, Path::new("replays/g3.rep"));
        assert_eq!(logs.len(), 1);
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (15, [5, 6].into())]);
        let label = |frame, action, delay| Label { frame, action, delay, queued: false, selected: vec![], target_unit: None, target_tile: None };
        let commands = vec![label(1, 0, 10), label(11, 4, 9), label(20, 7, 480)];
        let run = |logs: &[crate::relabel::DecisionLogRead], relabelled: bool| {
            let bound = bind_logs(logs, 0, 500);
            let mut labels = commands.clone();
            let added = decision_steps(relabelled, Some(&bound), &own, 500, 500, 500, &mut labels);
            let tpl = crate::shard::tests::game(1).steps[0].clone();
            let mut s: Vec<Step> = labels.iter().map(|l| Step { raw_frame: uranium_proto::RawFrame { frame: l.frame, ..tpl.raw_frame.clone() }, action_label: l.clone(), ..tpl.clone() }).collect();
            let r = MemoryCheck { decision_steps: added, ..attach_memory(&mut s, bound, 500, &own) };
            (r, labels, s.into_iter().map(|x| x.memory_action.map(|v| v.into_iter().map(|a| a.action).collect::<Vec<_>>())).collect::<Vec<_>>())
        };
        let (r, labels, got) = run(&logs, false);
        assert_eq!((r.status, r.decisions, r.checked, r.mismatched, r.decision_steps), ("ok", 4, 2, 0, Some(3)));
        assert_eq!(labels.iter().map(|l| (l.frame, l.is_unknown(), l.delay)).collect::<Vec<_>>(), vec![(1, false, 10), (7, true, 4), (11, false, 9), (15, true, 5), (20, false, 480), (23, true, 477)]);
        assert!(labels.iter().filter(|l| l.is_unknown()).all(|l| l.action == UNKNOWN_ACTION && l.selected.is_empty()));
        assert_eq!(got, vec![Some(vec![]), Some(vec![0]), Some(vec![4]), Some(vec![]), Some(vec![7]), Some(vec![])]);
        let dl = parse_memory_log(
            &[
                r#"{"frame":1,"action":0,"delay_bucket":0,"units":[]}"#,
                r#"{"frame":9,"action":4,"delay_bucket":2,"units":[5]}"#,
                r#"{"frame":17,"action":7,"delay_bucket":1,"units":[5,6]}"#,
                r#"{"frame":25,"action":0,"delay_bucket":3,"units":[]}"#,
                r#"{"decisions":4,"end_frame":498,"player":0}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        assert_eq!(run(&[dl], false), (r, labels, got));
        let (r, labels, got) = run(&[], false);
        assert_eq!((r.status, r.decision_steps), ("missing", None));
        assert_eq!(labels, commands);
        assert!(got.iter().all(Option::is_none));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn decision_steps_leave_relabel_builds_unchanged_041_r28() {
        use crate::label::Label;
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log};
        use crate::shard::MemoryAction;
        let label = |frame, action, delay| Label { frame, action, delay, queued: false, selected: vec![5], target_unit: None, target_tile: Some((3, 4)) };
        let teacher = vec![label(1, 0, 8), label(7, 12, 8), label(15, 4, 8), label(23, 0, 8)];
        let m = |action, units: Vec<u32>| MemoryAction { action, delay_bucket: 1, units };
        let read = DecisionLogRead::Complete(ModelLog { decisions: vec![(5, m(4, vec![5])), (13, m(0, vec![])), (30, m(7, vec![5]))], end_frame: Some(500), player: 0 });
        let bound = bind_log(&read, 0, 500);
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(3, [5].into()), (28, [5].into())]);
        let bytes = |l: &Vec<Label>| postcard::to_allocvec(l).unwrap();
        let mut relabelled = teacher.clone();
        assert_eq!(decision_steps(true, Some(&bound), &own, 500, 500, 500, &mut relabelled), None);
        assert_eq!(bytes(&relabelled), bytes(&teacher), "a relabelled build gets no decision steps");
        let mut failed = teacher.clone();
        assert_eq!(decision_steps(false, Some(&bound), &HashMap::new(), 500, 500, 500, &mut failed), None);
        assert_eq!(bytes(&failed), bytes(&teacher));
        let mut cut = teacher.clone();
        assert_eq!(decision_steps(false, Some(&bound), &own, 500, 20, 500, &mut cut), Some(2));
        assert_eq!(cut.iter().map(|l| l.frame).collect::<Vec<_>>(), vec![1, 3, 7, 11, 15, 23]);
        assert_eq!(cut.iter().filter(|l| !l.is_unknown()).cloned().collect::<Vec<_>>(), teacher);
        assert_eq!(decision_steps(false, None, &own, 500, 500, 500, &mut teacher.clone()), None);
    }

    #[test]
    fn distill_steps_are_the_logged_decisions_044_r2_ac1() {
        use crate::label::{Label, UNKNOWN_ACTION};
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log, playback_frame};
        use crate::shard::MemoryAction;
        let m = |action, units: Vec<u32>| MemoryAction { action, delay_bucket: 2, units };
        let decisions = vec![(1, m(0, vec![])), (9, m(4, vec![5])), (12, m(0, vec![])), (20, m(7, vec![5, 6])), (600, m(4, vec![5]))];
        let read = DecisionLogRead::Complete(ModelLog { decisions: decisions.clone(), end_frame: Some(698), player: 0 });
        let bound = bind_log(&read, 0, 700);
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (18, [5, 6].into())]);
        let (last, limit) = (332, 300);
        let (labels, n) = distill_steps(Some(&bound), &own, last, limit, 700).unwrap();
        assert_eq!(labels.iter().map(|l| l.frame).collect::<Vec<_>>(), vec![1, 7, 10, 18]);
        assert_eq!(n, 4);
        assert!(labels.iter().all(|l| l.action == UNKNOWN_ACTION && l.is_unknown() && l.selected.is_empty() && l.target_unit.is_none() && l.target_tile.is_none() && !l.queued));
        assert_eq!(labels.iter().map(|l| l.delay).collect::<Vec<_>>(), vec![6, 3, 8, 682]);
        let cmd = |frame, action| Label { frame, action, delay: 2, queued: false, selected: vec![5], target_unit: None, target_tile: None };
        let commands = vec![cmd(1, 0), cmd(3, 4), cmd(11, 7), cmd(25, 4)];
        let mut swapped = commands.clone();
        assert_eq!(step_labels(true, false, Some(&bound), &own, last, limit, 700, &mut swapped), Ok(Some(4)));
        assert_eq!(swapped, labels);
        assert!(swapped.iter().all(|l| ![3, 11, 25].contains(&l.frame)));
        let mut branch = commands.clone();
        assert_eq!(step_labels(false, false, Some(&bound), &own, last, limit, 700, &mut branch), Ok(Some(3)));
        assert_eq!(branch.iter().map(|l| (l.frame, l.is_unknown())).collect::<Vec<_>>(), vec![(1, false), (3, false), (7, true), (10, true), (11, false), (18, true), (25, false)]);
        let mut relabelled = commands.clone();
        assert_eq!(step_labels(false, true, Some(&bound), &own, last, limit, 700, &mut relabelled), Ok(None));
        assert_eq!(relabelled, commands);
        let mut unbound = commands.clone();
        assert!(step_labels(true, false, None, &own, last, limit, 700, &mut unbound).unwrap_err().starts_with("decision_log missing"));
        let tpl = crate::shard::tests::game(1).steps[0].clone();
        let mut steps: Vec<Step> = labels.iter().map(|l| Step { raw_frame: uranium_proto::RawFrame { frame: l.frame, ..tpl.raw_frame.clone() }, action_label: l.clone(), memory_action: None, ..tpl.clone() }).collect();
        let r = attach_memory(&mut steps, bound.clone(), last, &own);
        assert_eq!((r.status, r.decisions, r.checked, r.mismatched), ("ok", 5, 2, 0));
        let mem: Vec<Vec<MemoryAction>> = steps.iter().map(|s| s.memory_action.clone().expect("every step has memory")).collect();
        assert_eq!(mem, vec![vec![], vec![m(0, vec![])], vec![m(4, vec![5])], vec![m(0, vec![])]]);
        let live: Vec<u32> = steps.iter().map(|s| crate::relabel::live_of_playback(s.raw_frame.frame)).collect();
        for i in 1..steps.len() {
            let want: Vec<MemoryAction> = decisions.iter().filter(|(l, _)| *l >= live[i - 1] && *l < live[i]).map(|(_, d)| d.clone()).collect();
            assert_eq!(mem[i], want, "step {i}");
        }
        for (i, s) in steps.iter().enumerate() {
            for (l, _) in decisions.iter().filter(|(l, _)| playback_frame(*l) == s.raw_frame.frame) {
                assert!(*l >= live[i], "step {i} would hold its own decision");
                assert!(live.get(i + 1).is_none_or(|n| l < n), "step {i}'s decision belongs to step {}", i + 1);
            }
        }
        let game = ShardGame { meta: GameMeta { src: DISTILL_SRC.into(), stage: "B".into(), ..crate::shard::tests::game(1).meta }, game_info: crate::shard::tests::game(1).game_info, steps: steps.clone() };
        let shard = crate::shard::Shard { header: Default::default(), games: vec![game.clone()] };
        let back = crate::shard::decode(&postcard::to_stdvec(&shard).unwrap()).unwrap();
        assert_eq!(back.header.schema_version, crate::shard::SCHEMA_VERSION);
        assert_eq!(back.games, vec![game]);
        let early = vec![(1, m(0, vec![])), (2, m(4, vec![5])), (3, m(0, vec![])), (11, m(0, vec![]))];
        let (labels, n) = crate::relabel::distill_labels(&early, 300, 700);
        assert_eq!((labels.iter().map(|l| (l.frame, l.delay)).collect::<Vec<_>>(), n), (vec![(1, 8), (9, 691)], 2));
        let mem = crate::relabel::step_memory(&early, &[1, 9]);
        assert_eq!(mem, vec![vec![], vec![m(0, vec![]), m(4, vec![5]), m(0, vec![])]]);
        let (labels, n) = crate::relabel::distill_labels(&[(9, m(0, vec![])), (17, m(0, vec![]))], 300, 700);
        assert_eq!((labels.iter().map(|l| l.frame).collect::<Vec<_>>(), n), (vec![1, 7, 15], 2));
    }

    #[test]
    fn distill_views_failing_the_log_checks_are_filtered_044_r2() {
        use crate::relabel::{DecisionLogRead, ModelLog, bind_log};
        use crate::shard::MemoryAction;
        let m = |units: Vec<u32>| MemoryAction { action: 4, delay_bucket: 1, units };
        let own: HashMap<u32, std::collections::HashSet<u32>> = HashMap::from([(7, [5].into()), (15, [5].into())]);
        let complete = |decisions, player| DecisionLogRead::Complete(ModelLog { decisions, end_frame: Some(500), player });
        let reason = |b: Option<&Result<&ModelLog, MemoryCheck>>| match distill_steps(b, &own, 500, 500, 500) {
            Err(reason) => reason,
            other => panic!("{other:?}"),
        };
        assert!(reason(None).starts_with("decision_log missing (041-R28, 044-R2)"));
        for (read, status) in [
            (DecisionLogRead::Missing, "missing"),
            (DecisionLogRead::Incomplete("no end line".into()), "incomplete"),
            (complete(vec![(9, m(vec![5]))], 1), "other_player"),
            (DecisionLogRead::Complete(ModelLog { decisions: vec![(9, m(vec![5]))], end_frame: Some(9_000), player: 0 }), "incomplete"),
        ] {
            let r = reason(Some(&bind_log(&read, 0, 502)));
            assert!(r.starts_with(&format!("{DISTILL_FILTER_PREFIX} {status} ")), "{r}");
        }
        let bad = complete(vec![(9, m(vec![5])), (17, m(vec![6]))], 0);
        let r = reason(Some(&bind_log(&bad, 0, 502)));
        assert_eq!(r, "decision_log unit_mismatch (041-R28, 044-R2): 2 decisions, 2 checked, 1 mismatched");
        let good = complete(vec![(9, m(vec![5])), (17, m(vec![5]))], 0);
        assert!(distill_steps(Some(&bind_log(&good, 0, 502)), &own, 500, 500, 500).is_ok());
    }

    #[test]
    #[ignore = "needs URANIUM_TEST_DISTILL_SHARD, URANIUM_TEST_DISTILL_MEMORY and URANIUM_TEST_DISTILL_REPLAYS"]
    fn distill_shard_matches_its_logs_044_r2() {
        use crate::relabel::{bind_logs, playback_frame, step_memory};
        let var = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k}")));
        let shard = crate::shard::read(&var("URANIUM_TEST_DISTILL_SHARD")).unwrap();
        let mem = crate::act_log::MemoryLogDir::open(&var("URANIUM_TEST_DISTILL_MEMORY")).unwrap();
        let mut reps = Vec::new();
        for e in std::fs::read_dir(var("URANIUM_TEST_DISTILL_REPLAYS")).unwrap().flatten() {
            let b = std::fs::read(e.path()).unwrap();
            reps.push((hex(&Sha256::digest(&b)), e.path()));
        }
        assert!(!shard.games.is_empty());
        for g in &shard.games {
            assert_eq!((g.meta.src.as_str(), g.meta.stage.as_str()), (DISTILL_SRC, "B"));
            let file = &reps.iter().find(|(s, _)| *s == g.meta.replay_sha256).expect("the game's replay").1;
            let rep = crate::parse(&std::fs::read(file).unwrap()).unwrap();
            let logs = mem.logs(file);
            let log = bind_logs(&logs, g.meta.perspective_slot, rep.header.frames).expect("a bound log");
            let frames: Vec<u32> = g.steps.iter().map(|s| s.raw_frame.frame).collect();
            assert_eq!(frames[0], crate::label::GAME_START_FRAME);
            assert!(g.steps.iter().all(|s| s.action_label.is_unknown() && s.action_label.frame == s.raw_frame.frame));
            let last = *frames.last().unwrap();
            let mut want: Vec<u32> = log.decisions.iter().map(|(l, _)| playback_frame(*l)).filter(|p| *p <= last).collect();
            want.insert(0, crate::label::GAME_START_FRAME);
            want.dedup();
            assert_eq!(frames, want, "a step at each decision frame and nowhere else");
            let got: Vec<Vec<crate::shard::MemoryAction>> = g.steps.iter().map(|s| s.memory_action.clone().expect("memory on every step")).collect();
            assert_eq!(got, step_memory(&log.decisions, &frames));
            eprintln!("{}: {} decisions, {} steps", &g.meta.replay_sha256[..12], log.decisions.len(), frames.len());
        }
    }

    #[test]
    fn result_order_bridge_leave_recorded_none_040_r8() {
        let p = PlayResult { player_slots: vec![0, 1], ..Default::default() };
        assert_eq!(judge(&PlayResult { winner: Some(0), first_leaver_slot: Some(0), ..p.clone() }, Some(1), 0), (1, result_source::BRIDGE));
        assert_eq!(judge(&PlayResult { first_leaver_slot: Some(0), ..p.clone() }, None, 0), (-1, result_source::LEAVE));
        assert_eq!(judge(&PlayResult { first_leaver_slot: Some(1), ..p.clone() }, None, 0), (1, result_source::LEAVE));
        assert_eq!(judge(&p, Some(1), 0), (-1, result_source::RECORDED));
        assert_eq!(judge(&p, None, 0), (0, result_source::NONE));
    }

    #[test]
    #[ignore = "needs OpenBW (.env) and URANIUM_TEST_DAGGER_REP"]
    fn relabel_is_deterministic_and_from_the_teacher_041_r22() {
        uranium_store::dotenv::load_env();
        let path = std::env::var("URANIUM_TEST_DAGGER_REP").expect("URANIUM_TEST_DAGGER_REP");
        let bytes = std::fs::read(&path).unwrap();
        let rep = crate::parse(&bytes).unwrap();
        let map_hash = hex(&Sha256::digest(&rep.map));
        let families = MapFamilies::load().unwrap();
        let family = families.classify(&rep.header.map_name, &map_hash);
        let view = filter(&rep, &family, true, true).unwrap().remove(0);
        let cfg = PlayConfig::from_env(crate::play::work_dir_default().join("relabel-test")).unwrap();
        let labeler = Labeler::new().unwrap();
        let run = |relabel| {
            let vo = ViewOptions { until_frame: Some(11_520), time_limit: Some(14_400), relabel, map_hash: &map_hash, memory: None, distill: false };
            match build_view(&rep, &view, &cfg, &labeler, None, vo).unwrap() {
                Ok(g) => g,
                Err(o) => panic!("{o:?}"),
            }
        };
        let (a, b) = (run(Some(uranium_scripted::T0Build::TerranReact)), run(Some(uranium_scripted::T0Build::TerranReact)));
        assert_eq!(a.steps.len(), b.steps.len());
        assert!(a.steps == b.steps, "the same replay gives the same steps");
        let own = run(None);
        let table = uranium_env::actions::ActionTable::load().unwrap();
        let name = |s: &Step| table.actions.get(s.action_label.action as usize).map_or("unknown", |a| a.name.as_str()).to_string();
        let teacher = [
            "NoOp", "Build_SupplyDepot", "Build_Barracks", "Build_CommandCenter", "Build_Bunker", "Train_SCV", "Train_Marine", "Move_Pos", "Smart_Unit", "Hold", "Attack_Pos", "Attack_Unit", "Repair_Unit",
            "Gather_Unit", "ReturnCargo",
        ];
        let off: Vec<String> = a.steps.iter().map(name).filter(|n| !teacher.contains(&n.as_str())).collect();
        assert!(off.is_empty(), "labels outside the teacher's actions: {off:?}");
        assert!(a.steps.iter().any(|s| name(s) == "Train_SCV") && a.steps.iter().any(|s| name(s) == "Move_Pos"));
        let labels = |g: &ViewGame| g.steps.iter().map(|s| s.action_label.clone()).collect::<Vec<_>>();
        assert_ne!(labels(&a), labels(&own), "not the model's own commands");
        eprintln!("teacher steps {} vs own {}; coverage {:?}", a.steps.len(), own.steps.len(), a.coverage);
    }

    #[test]
    #[ignore = "needs OpenBW (.env) and URANIUM_TEST_TEACHER_REP"]
    fn relabel_of_a_teacher_game_gives_its_botgen_labels_041_r22() {
        teacher_game_agreement("URANIUM_TEST_TEACHER_REP", uranium_scripted::T0Build::TerranReact, 11_520);
    }

    #[test]
    #[ignore = "needs OpenBW (.env) and URANIUM_TEST_TEACHER_TP_REP"]
    fn relabel_of_a_tp_teacher_game_gives_its_botgen_labels_022_ac3() {
        teacher_game_agreement("URANIUM_TEST_TEACHER_TP_REP", uranium_scripted::T0Build::TerranTp, 14_400);
    }

    fn teacher_game_agreement(var: &str, build: uranium_scripted::T0Build, until_frame: u32) {
        uranium_store::dotenv::load_env();
        let path = std::env::var(var).expect(var);
        let rep = crate::parse(&std::fs::read(&path).unwrap()).unwrap();
        let map_hash = hex(&Sha256::digest(&rep.map));
        let family = MapFamilies::load().unwrap().classify(&rep.header.map_name, &map_hash);
        let view = filter(&rep, &family, true, true).unwrap().remove(0);
        let cfg = PlayConfig::from_env(crate::play::work_dir_default().join("relabel-test")).unwrap();
        let labeler = Labeler::new().unwrap();
        let run = |relabel| {
            let vo = ViewOptions { until_frame: Some(until_frame), time_limit: Some(14_400), relabel, map_hash: &map_hash, memory: None, distill: false };
            match build_view(&rep, &view, &cfg, &labeler, None, vo).unwrap() {
                Ok(g) => g,
                Err(o) => panic!("{o:?}"),
            }
        };
        let (shadow, botgen) = (run(Some(build)), run(None));
        type Key = (u16, Vec<u32>, Option<u32>, Option<(u16, u16)>);
        let key = |s: &Step| -> Key {
            let l = &s.action_label;
            let mut sel = l.selected.clone();
            sel.sort_unstable();
            (l.action, sel, l.target_unit, l.target_tile)
        };
        let acts = |g: &ViewGame| g.steps.iter().filter(|s| s.action_label.action != 0).map(|s| (s.action_label.frame, key(s))).collect::<Vec<_>>();
        let (a, b) = (acts(&shadow), acts(&botgen));
        let mut used = vec![false; a.len()];
        let mut offsets: std::collections::BTreeMap<i64, usize> = Default::default();
        let mut matched = 0;
        for (fb, kb) in &b {
            if let Some(i) = (0..a.len()).find(|&i| !used[i] && a[i].1 == *kb && (a[i].0 as i64 - *fb as i64).abs() <= 8) {
                used[i] = true;
                matched += 1;
                *offsets.entry(*fb as i64 - a[i].0 as i64).or_default() += 1;
            }
        }
        eprintln!("botgen actions {} shadow actions {} matched {} offsets {offsets:?}", b.len(), a.len(), matched);
        if std::env::var_os("URANIUM_TEST_VERBOSE").is_some() {
            let table = uranium_env::actions::ActionTable::load().unwrap();
            let show = |(f, k): &(u32, Key)| format!("{f} {} {:?} {:?} {:?}", table.actions[k.0 as usize].name, k.1, k.2, k.3);
            for (i, x) in a.iter().enumerate().filter(|(i, _)| !used[*i]) {
                eprintln!("shadow only #{i}: {}", show(x));
            }
            let mut used_b = vec![false; a.len()];
            for x in &b {
                match (0..a.len()).find(|&i| !used_b[i] && a[i].1 == x.1 && (a[i].0 as i64 - x.0 as i64).abs() <= 8) {
                    Some(i) => used_b[i] = true,
                    None => eprintln!("botgen only: {}", show(x)),
                }
            }
        }
        assert!(matched as f64 >= 0.95 * b.len() as f64 && matched as f64 >= 0.95 * a.len() as f64);
    }
}
