use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use serde::Serialize;
use uranium_proto::*;
use uranium_store::elog;
use crate::model_policy::ModelStats;
use super::scripted::ScriptedStats;
use super::act_log::ActLogStatus;

pub fn git_commit() -> &'static str {
    env!("URANIUM_GIT")
}

#[derive(Serialize)]
#[cfg_attr(test, derive(Default))]
pub struct SessionRecord {
    pub game_id: u64,
    pub slot: u8,
    pub map: String,
    pub map_hash: String,
    pub self_race: String,
    pub enemy_race: String,
    pub enemy_race_lobby: String,
    pub t0_build: String,
    pub winner: Option<u8>,
    pub won: Option<bool>,
    pub reason: String,
    pub end_frame: u32,
    pub wall_s: f64,
    pub policy_ms: f64,
    pub stats: Option<BridgeStats>,
    pub first_frame_minerals: usize,
    pub first_frame_units: usize,
    pub commands: u64,
    pub failed_commands: u64,
    pub unit_rejections: u64,
    pub unit_rejections_by: BTreeMap<String, u64>,
    pub frame_digest: String,
    pub privileged_msgs: u64,
    pub visibility_violations: u64,
    pub final_frame: Option<(u32, usize, usize)>,
    pub policy: String,
    pub model: Option<ModelStats>,
    pub error: Option<String>,
    pub scripted: Option<ScriptedStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub act_log: Option<ActLogStatus>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub snapshots: Vec<uranium_env::branch::SnapshotTaken>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<uranium_env::branch::BranchOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub infer_seed: Option<u64>,
}

pub(crate) fn revealed_enemy_race(f: &RawFrame, me: u8) -> Option<String> {
    let types = uranium_env::types::TypeTables::get();
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != uranium_env::types::NEUTRAL_PLAYER)
        .filter_map(|u| types.unit(u.unit_type))
        .find(|t| matches!(t.race.as_str(), "Terran" | "Protoss" | "Zerg"))
        .map(|t| t.race.clone())
}

pub(crate) fn visibility_violations(f: &RawFrame, me: u8) -> u64 {
    f.units
        .iter()
        .filter(|u| u.owner != me && u.owner != uranium_env::types::NEUTRAL_PLAYER)
        .filter(|u| {
            let (w, h) = uranium_scripted::types::tile_size(u.unit_type);
            let (x0, y0) = ((u.pos.0 as i32 - w as i32 * 16) / 32 - 1, (u.pos.1 as i32 - h as i32 * 16) / 32 - 1);
            let seen = (y0..=y0 + h as i32 + 1).any(|ty| (x0..=x0 + w as i32 + 1).any(|tx| tx >= 0 && ty >= 0 && f.visibility.get(tx as u16, ty as u16)));
            let hidden = (u.has(flags::CLOAKED) || u.has(flags::BURROWED)) && !u.has(flags::DETECTED);
            if (!seen || hidden) && std::env::var("URANIUM_AGENT_DEBUG_VIS").is_ok() {
                elog!("[vis] frame {} slot {me} unit {} type {} owner {} pos {:?} flags {:#x} seen {seen} hidden {hidden}", f.frame, u.id, u.unit_type, u.owner, u.pos, u.flags);
            }
            !seen || hidden
        })
        .count() as u64
}

pub type RecordSink = Arc<Mutex<dyn Write + Send>>;

pub fn open_sink(out: &Path) -> Result<RecordSink> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(out).with_context(|| format!("opening {}", out.display()))?;
    Ok(Arc::new(Mutex::new(BufWriter::new(SyncFile(file)))))
}

struct SyncFile(std::fs::File);

impl Write for SyncFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()?;
        self.0.sync_data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_commit_is_the_build_constant_051_r2() {
        let c = git_commit();
        assert_eq!(c, env!("URANIUM_GIT"));
        assert!(c == "unknown" || (c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit())), "{c}");
    }
}
