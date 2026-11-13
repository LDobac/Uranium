//! 041-R28 (051-R2, 051-R13 ④) memory source: the per-observation act log of a branch-game trunk
//! as a model decision log for `uranium-replay build --memory-log`.
//!
//! Minimal adapter (U5 of `reports/jepa_planning.md` 7.14): it reads only the fields the 041-R28
//! memory needs from the 051-R2 layout (`spec/051-branch-games.md` "행동 기록 줄") through
//! `serde_json::Value`, so it does not depend on the act-log types of `uranium_env::branch` (written in
//! parallel). Observation lines `{"i", "frame", "digest", "act", "decision"?, "state"?, "forced"?}`,
//! then the end line `{"end": {"obs", "decisions", "end_frame", "player", "policy", ...}}`. A model
//! session's `decision` (`action`, `delay_bucket`, `units`: the values it recorded in memory, 041-R28 ①,
//! whether or not the executor refused it, forced prefix decisions included, 051-R5 ⑤) becomes one
//! [`LoggedDecision`] at the line's live frame. A scripted (T0) session's act log has no decisions and is
//! not a memory source.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::relabel::{DecisionLogRead, LoggedDecision, ModelLog, parse_decision_log};
use crate::shard::MemoryAction;

/// What one act log is as a memory source.
#[derive(Debug, Clone, PartialEq)]
pub enum ActLogRead {
    /// A model session's log (complete or not, 041-R28 ④).
    Model(DecisionLogRead),
    /// A scripted session's log: observation lines without `decision` (or an end line whose policy is
    /// `t0@…`, 020-R5). It has no agent memory.
    Scripted,
}

/// True when the first non-empty line is in the 051-R2 layout (an observation line with `i` or the
/// `end` object) rather than the 041-R28 decision log layout.
pub fn is_act_log(text: &str) -> bool {
    let Some(first) = text.lines().find(|l| !l.trim().is_empty()) else { return false };
    match serde_json::from_str::<Value>(first) {
        Ok(Value::Object(m)) => m.contains_key("i") || m.get("end").is_some_and(Value::is_object),
        _ => false,
    }
}

fn u64_of(v: &Value, key: &str) -> Result<u64, String> {
    v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("no integer `{key}`"))
}

fn decision_of(d: &Value) -> Result<MemoryAction, String> {
    let action = u16::try_from(u64_of(d, "action")?).map_err(|_| "`action` out of range".to_string())?;
    let delay_bucket = u32::try_from(u64_of(d, "delay_bucket")?).map_err(|_| "`delay_bucket` out of range".to_string())?;
    let units = d
        .get("units")
        .and_then(Value::as_array)
        .ok_or("no `units` list")?
        .iter()
        .map(|u| u.as_u64().and_then(|x| u32::try_from(x).ok()).ok_or("a unit id that is not a u32"))
        .collect::<Result<Vec<u32>, _>>()?;
    Ok(MemoryAction { action, delay_bucket, units })
}

/// Parses a 051-R2 act log. Incomplete (041-R28 ④): a line that does not parse, observation numbers
/// not 0, 1, 2, ..., frames going back, decisions on only some observation lines with no end line to
/// tell, no end line, lines after it, or an end line whose `obs` / `decisions` counts differ from the
/// lines.
pub fn parse_act_log(text: &str) -> ActLogRead {
    let inc = |s: String| ActLogRead::Model(DecisionLogRead::Incomplete(s));
    let mut decisions: Vec<LoggedDecision> = Vec::new();
    let (mut obs, mut last_frame) = (0u64, 0u64);
    let mut end: Option<(Option<u32>, u8, u64, u64, bool)> = None;
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let ln = n + 1;
        if end.is_some() {
            return inc(format!("act log line {ln} after the end line"));
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return inc(format!("act log line {ln}: {e}")),
        };
        if let Some(e) = v.get("end") {
            let get = || -> Result<_, String> {
                let end_frame = match e.get("end_frame") {
                    None | Some(Value::Null) => None,
                    Some(f) => Some(f.as_u64().and_then(|x| u32::try_from(x).ok()).ok_or("`end_frame` is not a frame")?),
                };
                let player = u8::try_from(u64_of(e, "player")?).map_err(|_| "`player` out of range".to_string())?;
                let scripted = e.get("policy").and_then(Value::as_str).is_some_and(|p| p.starts_with("t0"));
                Ok((end_frame, player, u64_of(e, "obs")?, u64_of(e, "decisions")?, scripted))
            };
            match get() {
                Ok(x) => end = Some(x),
                Err(r) => return inc(format!("act log end line {ln}: {r}")),
            }
            continue;
        }
        let parsed = (|| -> Result<(u64, u64, Option<MemoryAction>), String> {
            let d = match v.get("decision") {
                None | Some(Value::Null) => None,
                Some(d) => Some(decision_of(d)?),
            };
            Ok((u64_of(&v, "i")?, u64_of(&v, "frame")?, d))
        })();
        let (i, frame, d) = match parsed {
            Ok(x) => x,
            Err(r) => return inc(format!("act log line {ln}: {r}")),
        };
        if i != obs {
            return inc(format!("act log line {ln}: observation {i}, expected {obs}"));
        }
        if frame < last_frame {
            return inc(format!("act log line {ln}: frame {frame} before the previous observation's {last_frame}"));
        }
        let Ok(frame32) = u32::try_from(frame) else { return inc(format!("act log line {ln}: frame {frame} out of range")) };
        obs += 1;
        last_frame = frame;
        if let Some(d) = d {
            decisions.push((frame32, d));
        }
    }
    let Some((end_frame, player, end_obs, end_dec, scripted)) = end else {
        // Without the end line a scripted log (observations, no decisions) cannot be told from a
        // model log that broke off before its first decision; neither is a memory source.
        if obs > 0 && decisions.is_empty() {
            return ActLogRead::Scripted;
        }
        return inc("act log without its end line (the game did not end in the session)".into());
    };
    if scripted || (decisions.is_empty() && end_dec == 0 && obs > 0) {
        return ActLogRead::Scripted;
    }
    if end_obs != obs {
        return inc(format!("act log end line counts {end_obs} observations, the log has {obs}"));
    }
    if end_dec != decisions.len() as u64 {
        return inc(format!("act log end line counts {end_dec} decisions, the log has {}", decisions.len()));
    }
    ActLogRead::Model(DecisionLogRead::Complete(ModelLog { decisions, end_frame, player }))
}

/// A memory log file's text in either layout (041-R28 decision log or 051-R2 act log); a scripted
/// act log is `None`.
pub fn parse_memory_log(text: &str) -> Option<DecisionLogRead> {
    if !is_act_log(text) {
        return Some(parse_decision_log(text));
    }
    match parse_act_log(text) {
        ActLogRead::Model(r) => Some(r),
        ActLogRead::Scripted => None,
    }
}

/// 041-R28 (AMB-041-17) and 051-R12 (020 제안 R22 ①): the memory logs of replay `file` in the
/// `--memory-log` folder `dir`. `<stem>.jsonl` (the decision log of a `memory_log` plan, or an act log
/// under that name) if present, else every act log `<stem>-p<players index>.jsonl` of the game (an
/// evaluation job's `acts/` folder beside `replays/`), scripted ones left out. Empty = no log.
pub fn read_memory_logs(dir: &Path, file: &Path) -> Vec<DecisionLogRead> {
    let one = crate::pipeline::memory_log_path(dir, file);
    let read = |p: &Path| match std::fs::read_to_string(p) {
        Ok(t) => parse_memory_log(&t),
        Err(e) => Some(DecisionLogRead::Incomplete(format!("reading {}: {e}", p.display()))),
    };
    if one.exists() {
        return read(&one).map_or_else(|| vec![DecisionLogRead::Incomplete(format!("{}: a scripted session's act log", one.display()))], |r| vec![r]);
    }
    act_log_paths(dir, file).iter().filter_map(|p| read(p)).collect()
}

/// The act logs `<stem>-p<k>.jsonl` (k a players index) of replay `file` in `dir`, sorted by k.
pub fn act_log_paths(dir: &Path, file: &Path) -> Vec<PathBuf> {
    let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let prefix = format!("{stem}-p");
    let mut found: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let k = name.strip_prefix(&prefix)?.strip_suffix(".jsonl")?.parse::<u32>().ok()?;
            Some((k, e.path()))
        })
        .collect();
    found.sort();
    found.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::relabel::bind_logs;

    /// One 051-R2 observation line of a model session (`d` = (action, delay, units)) or of a scripted one.
    pub(crate) fn obs_line(i: u64, frame: u32, d: Option<(u16, u32, Vec<u32>)>, forced: bool) -> String {
        let act = r#"{"cmds":[{"kind":"Train","unit":3,"pos":null,"target":null,"arg":7}],"advance_frames":8,"surrender":null}"#;
        match d {
            Some((a, b, u)) => format!(
                r#"{{"i":{i},"frame":{frame},"digest":"9f1c00000000abcd","act":{act},"decision":{{"action":{a},"delay_bucket":{b},"units":{u:?},"rejected":{}}},"state":null,"forced":{forced}}}"#,
                a % 2 == 1
            ),
            None => format!(r#"{{"i":{i},"frame":{frame},"digest":"00000000deadbeef","act":{act}}}"#),
        }
    }

    pub(crate) fn end_line(obs: u64, decisions: u64, end_frame: u32, player: u8, policy: &str) -> String {
        format!(
            r#"{{"end":{{"obs":{obs},"decisions":{decisions},"end_frame":{end_frame},"reason":"TimeLimitDraw","player":{player},"players_index":0,"policy":"{policy}","git":"abc","feature_schema":7,"action_schema":2}}}}"#
        )
    }

    fn model_log() -> String {
        [
            obs_line(0, 1, Some((0, 3, vec![])), true),
            obs_line(1, 9, Some((37, 5, vec![1203])), true),
            obs_line(2, 17, Some((4, 1, vec![5, 6])), false),
            end_line(3, 3, 11_568, 0, "model:ckpt-39096@aa:stochastic"),
        ]
        .join("\n")
    }

    /// 041-R28 / 051-R2: a model session's act log gives its decisions (forced and executor-refused ones
    /// included) at their live frames, with the end line's frame and player: the same `ModelLog` as the
    /// 041-R28 decision log of those decisions.
    #[test]
    fn act_log_reads_as_a_decision_log_051_r2_041_r28() {
        let text = model_log();
        assert!(is_act_log(&text));
        let ActLogRead::Model(DecisionLogRead::Complete(log)) = parse_act_log(&text) else { panic!("complete model log") };
        let m = |action, delay_bucket, units| MemoryAction { action, delay_bucket, units };
        assert_eq!(log.decisions, vec![(1, m(0, 3, vec![])), (9, m(37, 5, vec![1203])), (17, m(4, 1, vec![5, 6]))]);
        assert_eq!((log.end_frame, log.player), (Some(11_568), 0));
        // The equivalent 041-R28 decision log parses to the same log.
        let dl = [
            r#"{"frame":1,"action":0,"delay_bucket":3,"units":[]}"#,
            r#"{"frame":9,"action":37,"delay_bucket":5,"units":[1203]}"#,
            r#"{"frame":17,"action":4,"delay_bucket":1,"units":[5,6]}"#,
            r#"{"decisions":3,"end_frame":11568,"player":0}"#,
        ]
        .join("\n");
        assert!(!is_act_log(&dl));
        assert_eq!(parse_memory_log(&dl), Some(DecisionLogRead::Complete(log.clone())));
        assert_eq!(parse_memory_log(&text), Some(DecisionLogRead::Complete(log)));
    }

    /// 041-R28 ④ on act logs: anything but a whole log with its end line is incomplete; a scripted
    /// session's log is no memory source.
    #[test]
    fn act_log_incomplete_and_scripted_051_r2() {
        let d = |i, f| obs_line(i, f, Some((4, 1, vec![5])), false);
        let end = |o, n| end_line(o, n, 900, 0, "model:x");
        for bad in [
            format!("{}\n{}", d(0, 1), d(1, 9)),                                   // no end line
            format!("{}\n{}\n{}", d(0, 1), d(1, 9), end(2, 1)),                    // decision count
            format!("{}\n{}\n{}", d(0, 1), d(1, 9), end(3, 2)),                    // observation count
            format!("{}\n{}\n{}", d(0, 1), d(2, 9), end(2, 2)),                    // observation numbers
            format!("{}\n{}\n{}", d(0, 9), d(1, 1), end(2, 2)),                    // frames back
            format!("{}\n{}\n{}", d(0, 1), end(1, 1), d(1, 9)),                    // after the end line
            format!("{}\n{{\"i\":1,\"frame\":9,\"decision\":{{\"action\":4}}}}\n{}", d(0, 1), end(2, 2)), // bad decision
            format!("{}\n{}", d(0, 1), r#"{"end":{"obs":1,"decisions":1}}"#),     // end without player
        ] {
            assert!(matches!(parse_act_log(&bad), ActLogRead::Model(DecisionLogRead::Incomplete(_))), "{bad}");
        }
        let t0 = format!("{}\n{}\n{}", obs_line(0, 1, None, false), obs_line(1, 9, None, false), end_line(2, 0, 900, 1, "t0@3:zerg_rush"));
        assert_eq!(parse_act_log(&t0), ActLogRead::Scripted);
        assert_eq!(parse_memory_log(&t0), None);
        // A scripted log without its end line is no memory source either.
        assert_eq!(parse_act_log(&obs_line(0, 1, None, false)), ActLogRead::Scripted);
    }

    /// 041-R28 / 051-R12: `--memory-log` finds `<stem>.jsonl` first, else the game's act logs
    /// `<stem>-p<k>.jsonl` (scripted ones left out); each view binds the log of its own player.
    #[test]
    fn memory_logs_found_by_replay_stem_051_r12() {
        let dir = std::env::temp_dir().join(format!("uranium-act-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let rep = Path::new("job/stage/replays/g7.rep");
        assert!(read_memory_logs(&dir, rep).is_empty(), "no log");
        std::fs::write(dir.join("g7-p0.jsonl"), model_log()).unwrap();
        std::fs::write(dir.join("g7-p1.jsonl"), format!("{}\n{}", obs_line(0, 1, None, false), end_line(1, 0, 11_568, 1, "t0@3:zerg_rush"))).unwrap();
        std::fs::write(dir.join("g70-p0.jsonl"), "not this game").unwrap();
        std::fs::write(dir.join("g7-px.jsonl"), "not an act log name").unwrap();
        assert_eq!(act_log_paths(&dir, rep), vec![dir.join("g7-p0.jsonl"), dir.join("g7-p1.jsonl")]);
        let logs = read_memory_logs(&dir, rep);
        assert_eq!(logs.len(), 1, "the scripted log is no memory source");
        let ours = bind_logs(&logs, 0, 11_570).expect("player 0's view binds its log");
        assert_eq!(ours.decisions.len(), 3);
        assert_eq!(bind_logs(&logs, 1, 11_570).unwrap_err().status, "other_player");
        // `<stem>.jsonl` wins over act logs.
        std::fs::write(dir.join("g7.jsonl"), r#"{"decisions":0,"end_frame":11568,"player":0}"#).unwrap();
        let logs = read_memory_logs(&dir, rep);
        assert_eq!(logs.len(), 1);
        assert!(bind_logs(&logs, 0, 11_568).unwrap().decisions.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
