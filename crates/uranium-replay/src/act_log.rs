use std::collections::HashMap;
use std::path::{Path, PathBuf};

use uranium_match::branch::ActLog;

use crate::relabel::{DecisionLogRead, LoggedDecision, ModelLog, parse_decision_log};

pub fn is_act_log(text: &str) -> bool {
    let Some(first) = text.lines().find(|l| !l.trim().is_empty()) else { return false };
    match serde_json::from_str::<serde_json::Value>(first) {
        Ok(serde_json::Value::Object(m)) => m.contains_key("i") || m.get("end").is_some_and(serde_json::Value::is_object),
        _ => false,
    }
}

pub fn decisions_of(log: &ActLog) -> Option<ModelLog> {
    if log.end.policy.starts_with("t0") || (log.end.decisions == 0 && !log.lines.is_empty()) {
        return None;
    }
    let decisions: Vec<LoggedDecision> = log.lines.iter().filter_map(|l| l.decision.as_ref().map(|d| (l.frame, d.memory_action()))).collect();
    Some(ModelLog { decisions, end_frame: Some(log.end.end_frame), player: log.end.player })
}

pub fn parse_memory_log(text: &str) -> Option<DecisionLogRead> {
    if !is_act_log(text) {
        return Some(parse_decision_log(text));
    }
    match ActLog::read(text.as_bytes()) {
        Ok(log) => decisions_of(&log).map(DecisionLogRead::Complete),
        Err(e) => Some(DecisionLogRead::Incomplete(format!("{e:#}"))),
    }
}

#[derive(Debug, Clone)]
pub struct MemoryLogDir {
    pub dir: PathBuf,
    acts: HashMap<String, Vec<(u32, PathBuf)>>,
}

impl MemoryLogDir {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let mut acts: HashMap<String, Vec<(u32, PathBuf)>> = HashMap::new();
        for e in std::fs::read_dir(dir)?.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some((stem, k)) = name.strip_suffix(".jsonl").and_then(|n| n.rsplit_once("-p")) else { continue };
            if let Ok(k) = k.parse::<u32>() {
                acts.entry(stem.to_string()).or_default().push((k, e.path()));
            }
        }
        acts.values_mut().for_each(|v| v.sort());
        Ok(Self { dir: dir.to_path_buf(), acts })
    }

    pub fn act_log_paths(&self, file: &Path) -> Vec<PathBuf> {
        let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        self.acts.get(&stem).map(|v| v.iter().map(|(_, p)| p.clone()).collect()).unwrap_or_default()
    }

    pub fn logs(&self, file: &Path) -> Vec<DecisionLogRead> {
        let one = crate::pipeline::memory_log_path(&self.dir, file);
        let read = |p: &Path| match std::fs::read_to_string(p) {
            Ok(t) => parse_memory_log(&t),
            Err(e) => Some(DecisionLogRead::Incomplete(format!("reading {}: {e}", p.display()))),
        };
        if one.exists() {
            return read(&one).map_or_else(|| vec![DecisionLogRead::Incomplete(format!("{}: a scripted session's act log", one.display()))], |r| vec![r]);
        }
        self.act_log_paths(file).iter().filter_map(|p| read(p)).collect()
    }
}

pub fn read_memory_logs(dir: &Path, file: &Path) -> Vec<DecisionLogRead> {
    MemoryLogDir::open(dir).map(|d| d.logs(file)).unwrap_or_default()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::relabel::bind_logs;
    use crate::shard::MemoryAction;
    use uranium_match::branch::{ActLogEnd, Decision, ObsLine};
    use uranium_proto::{Act, Command, EndReason};

    pub(crate) fn obs_line(i: u32, frame: u32, d: Option<(u16, u32, Vec<u32>)>, forced: bool) -> ObsLine {
        let act = Act { cmds: vec![Command::Train { building: 3, unit_type: 7 }], advance_frames: 8, valid_until_frame: None, surrender: None };
        let decision = d.map(|(action, delay_bucket, units)| Decision { action, delay_bucket, units, rejected: action % 2 == 1 });
        let model = decision.is_some();
        ObsLine { i, frame, digest: 0x9f1c_0000_0000_abcd ^ u64::from(i), act, decision, state: None, state_sha256: None, forced: forced && model }
    }

    pub(crate) fn log_text(lines: Vec<ObsLine>, end_frame: u32, player: u8, policy: &str) -> String {
        let end = ActLogEnd {
            obs: 0,
            decisions: 0,
            end_frame,
            reason: EndReason::TimeLimitDraw,
            player,
            players_index: 0,
            policy: policy.into(),
            git: "abc".into(),
            feature_schema: 7,
            action_schema: 2,
        };
        let mut b = Vec::new();
        let mut w = uranium_match::branch::ActLogWriter::new(&mut b);
        for l in &lines {
            w.push(l).unwrap();
        }
        w.finish(end).unwrap();
        String::from_utf8(b).unwrap()
    }

    fn model_log() -> String {
        log_text(vec![obs_line(0, 1, Some((0, 3, vec![])), true), obs_line(1, 9, Some((37, 5, vec![1203])), true), obs_line(2, 17, Some((4, 1, vec![5, 6])), false)], 11_568, 0, "model:ckpt-39096@aa:stochastic")
    }

    fn t0_log(end_frame: u32) -> String {
        log_text(vec![obs_line(0, 1, None, false), obs_line(1, 9, None, false)], end_frame, 1, "t0@3:zerg_rush")
    }

    #[test]
    fn act_log_reads_as_a_decision_log_051_r2_041_r28() {
        let text = model_log();
        assert!(is_act_log(&text));
        let Some(DecisionLogRead::Complete(log)) = parse_memory_log(&text) else { panic!("complete model log") };
        let m = |action, delay_bucket, units| MemoryAction { action, delay_bucket, units };
        assert_eq!(log.decisions, vec![(1, m(0, 3, vec![])), (9, m(37, 5, vec![1203])), (17, m(4, 1, vec![5, 6]))]);
        assert_eq!((log.end_frame, log.player), (Some(11_568), 0));
        let dl = [
            r#"{"frame":1,"action":0,"delay_bucket":3,"units":[]}"#,
            r#"{"frame":9,"action":37,"delay_bucket":5,"units":[1203]}"#,
            r#"{"frame":17,"action":4,"delay_bucket":1,"units":[5,6]}"#,
            r#"{"decisions":3,"end_frame":11568,"player":0}"#,
        ]
        .join("\n");
        assert!(!is_act_log(&dl));
        assert_eq!(parse_memory_log(&dl), Some(DecisionLogRead::Complete(log)));
    }

    #[test]
    fn act_log_incomplete_and_scripted_051_r2() {
        let text = model_log();
        let lines: Vec<&str> = text.lines().collect();
        for bad in [
            lines[..3].join("\n"),
            [lines[0], lines[2], lines[3]].join("\n"),
            [lines[0], lines[1], lines[3], lines[2]].join("\n"),
            format!("{}\nnot json\n{}", lines[0], lines[3]),
        ] {
            assert!(matches!(parse_memory_log(&bad), Some(DecisionLogRead::Incomplete(_))), "{bad}");
        }
        assert_eq!(parse_memory_log(&t0_log(900)), None);
    }

    #[test]
    fn memory_logs_found_by_replay_stem_051_r12() {
        let dir = std::env::temp_dir().join(format!("uranium-act-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let rep = Path::new("job/stage/replays/g7.rep");
        assert!(read_memory_logs(&dir, rep).is_empty(), "no log");
        std::fs::write(dir.join("g7-p0.jsonl"), model_log()).unwrap();
        std::fs::write(dir.join("g7-p1.jsonl"), t0_log(11_568)).unwrap();
        std::fs::write(dir.join("g70-p0.jsonl"), "not this game").unwrap();
        std::fs::write(dir.join("g7-px.jsonl"), "not an act log name").unwrap();
        assert_eq!(MemoryLogDir::open(&dir).unwrap().act_log_paths(rep), vec![dir.join("g7-p0.jsonl"), dir.join("g7-p1.jsonl")]);
        let logs = read_memory_logs(&dir, rep);
        assert_eq!(logs.len(), 1, "the scripted log is no memory source");
        let ours = bind_logs(&logs, 0, 11_570).expect("player 0's view binds its log");
        assert_eq!(ours.decisions.len(), 3);
        assert_eq!(bind_logs(&logs, 1, 11_570).unwrap_err().status, "other_player");
        std::fs::write(dir.join("g7.jsonl"), r#"{"decisions":0,"end_frame":11568,"player":0}"#).unwrap();
        let logs = read_memory_logs(&dir, rep);
        assert_eq!(logs.len(), 1);
        assert!(bind_logs(&logs, 0, 11_568).unwrap().decisions.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
