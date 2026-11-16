use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use anyhow::Result;
use serde::Serialize;
use uranium_proto::*;
use super::record::{SessionRecord, RecordSink, open_sink};

#[derive(Serialize)]
struct UnitRejectionLine<'a> {
    game_id: u64,
    slot: u8,
    frame: u32,
    cmd_index: u16,
    order: Option<&'a CmdOrder>,
    unit: UnitId,
    unit_type: Option<u16>,
    command: u8,
    error: u16,
}

#[derive(Clone, Default)]
pub struct RejectionLog(Option<Arc<(PathBuf, Mutex<Option<RecordSink>>)>>);

impl RejectionLog {
    pub fn new(path: PathBuf) -> Self {
        Self(Some(Arc::new((path, Mutex::new(None)))))
    }

    pub fn beside(records: &Path) -> Self {
        Self::new(records.with_file_name("unit_rejections.jsonl"))
    }

    pub(crate) fn append(&self, lines: &[String]) -> Result<()> {
        let Some(inner) = &self.0 else { return Ok(()) };
        if lines.is_empty() {
            return Ok(());
        }
        let mut slot = inner.1.lock().unwrap();
        if slot.is_none() {
            *slot = Some(open_sink(&inner.0)?);
        }
        let sink = slot.as_ref().expect("opened above");
        let mut w = sink.lock().unwrap();
        for l in lines {
            writeln!(w, "{l}")?;
        }
        w.flush()?;
        Ok(())
    }
}

pub(crate) fn record_unit_rejections(f: &RawFrame, last_cmds: &[Command], hello: &Hello, rec: &mut SessionRecord, lines: &mut Vec<String>) {
    for e in &f.events {
        let GameEvent::UnitRejected { cmd_index, unit, unit_type, command, error } = *e else { continue };
        rec.unit_rejections += 1;
        let ty = unit_type.map_or_else(|| "none".to_string(), |t| t.to_string());
        *rec.unit_rejections_by.entry(format!("{command}/{ty}/{error}")).or_default() += 1;
        let order = match last_cmds.get(cmd_index as usize) {
            Some(Command::Unit { order, .. }) => Some(order),
            _ => None,
        };
        let line = UnitRejectionLine { game_id: hello.game_id, slot: hello.slot, frame: f.frame, cmd_index, order, unit, unit_type, command, error };
        lines.push(serde_json::to_string(&line).expect("rejection line serializes"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn unit_rejections_are_counted_and_logged_031_r10() {
        let hello = Hello { protocol_version: PROTOCOL_VERSION, backend: Backend::OpenBw, game_version: "x".into(), mode: Mode::Lockstep, game_id: 12, slot: 1, replay: false };
        let rejected = |cmd_index, unit, unit_type| GameEvent::UnitRejected { cmd_index, unit, unit_type, command: 14, error: 10 };
        let frame = RawFrame {
            frame: 96,
            self_state: PlayerState::default(),
            units: vec![],
            visibility: BitGrid::new(1, 1),
            explored: BitGrid::new(1, 1),
            creep: BitGrid::new(1, 1),
            events: vec![GameEvent::CommandFailed { cmd_index: 1, error: 10 }, rejected(0, 5, Some(0)), rejected(0, 6, Some(0)), rejected(1, 7, None)],
            effects: vec![],
        };
        let last_cmds = vec![
            Command::Unit { units: vec![5, 6], order: CmdOrder::Attack(Target::Unit(99)), queued: false },
            Command::Build { worker: 7, unit_type: 109, tile: (3, 4) },
        ];
        let (mut rec, mut lines) = (SessionRecord::default(), Vec::new());
        record_unit_rejections(&frame, &last_cmds, &hello, &mut rec, &mut lines);
        assert_eq!(rec.unit_rejections, 3);
        assert_eq!(rec.unit_rejections_by, BTreeMap::from([("14/0/10".to_string(), 2), ("14/none/10".to_string(), 1)]));
        assert_eq!(rec.failed_commands, 0);
        let first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(first, serde_json::json!({"game_id": 12, "slot": 1, "frame": 96, "cmd_index": 0, "order": {"Attack": {"Unit": 99}}, "unit": 5, "unit_type": 0, "command": 14, "error": 10}));
        let third: serde_json::Value = serde_json::from_str(&lines[2]).unwrap();
        assert_eq!((&third["order"], &third["unit_type"]), (&serde_json::Value::Null, &serde_json::Value::Null));

        let dir = std::env::temp_dir().join(format!("uranium-agent-rejections-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = RejectionLog::beside(&dir.join("results.jsonl"));
        log.append(&[]).unwrap();
        assert!(!dir.join("unit_rejections.jsonl").exists());
        log.append(&lines[..2]).unwrap();
        log.clone().append(&lines[2..]).unwrap();
        let text = std::fs::read_to_string(dir.join("unit_rejections.jsonl")).unwrap();
        assert_eq!(text.lines().collect::<Vec<_>>(), lines);
        RejectionLog::default().append(&lines).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
