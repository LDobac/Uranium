//! Decision-granularity measurement helper (G0, `tools/eval/granularity.py`). Analysis only:
//! nothing is written to the datasets and no labelling rule changes.
//!
//! Per game it writes one JSON line with the action labels and a compact own/enemy state in an
//! early window:
//! - from shards (040): the stored labels and the stored decision-time observations;
//! - from replays: the replay's own commands of the Terran player through the same 040 labeler
//!   as `build` (040-R3, R4: one label per command, same-frame commands become delay-1 chains,
//!   031-R11 delay buckets), observed in one OpenBW replay pass. Unlike `build` there is no
//!   5-minute bar and no desync check, so a model game lost early to a rush is kept.
//!
//! Selected unit ids in the labels are the observation's unit ids, as in the shards.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use uranium_proto::{GameInfo, RawFrame};

use crate::cmd;
use crate::header::RACE_TERRAN;
use crate::label::{Labeler, Own};
use crate::play::{PlayConfig, play_with, slot_of};
use crate::shard::ShardGame;

/// Frames kept (labels and states) and the late check frame.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    /// Labels and states at frames `lo..=hi` are kept.
    pub lo: u32,
    pub hi: u32,
    /// A state is kept at least every `period` frames in the window (besides every command label).
    pub period: u32,
    /// Own SCVs are counted at the last observation at or before this frame (8:00 = 11,520), if
    /// the game reached it.
    pub at: u32,
}

/// Filters for shard games (replays are taken as given).
#[derive(Clone, Debug, Default)]
pub struct ShardFilter {
    pub matchup: Option<String>,
    pub stage: Option<String>,
    /// Empty = every map family.
    pub families: Vec<String>,
}

/// One own unit: (id, type, x, y, flags, hp, loaded).
type OwnUnit = (u32, u16, u16, u16, u32, u16, bool);
/// One visible enemy unit: (id, type, x, y).
type EnemyUnit = (u32, u16, u16, u16);
/// (frame, action, delay, queued, selected, target unit).
type LabelRow = (u32, u16, u32, bool, Vec<u32>, Option<u32>);

#[derive(Serialize, Debug, Clone)]
pub struct State {
    pub f: u32,
    pub own: Vec<OwnUnit>,
    pub enemy: Vec<EnemyUnit>,
}

#[derive(Serialize, Debug, Clone)]
pub struct GameDump {
    pub source: String,
    /// Replay file stem, or `<shard file>#<game index>`.
    pub game: String,
    pub replay_sha256: String,
    pub matchup: String,
    pub map_family: String,
    pub stage: String,
    pub result: i8,
    /// Last frame of the game (replay header frames / last stored step).
    pub end_frame: u32,
    /// Own start location (pixel centre of the start tile's 4x3 town hall).
    pub start: Option<(u32, u32)>,
    pub enemy_start: Option<(u32, u32)>,
    /// Own completed SCVs at `Window::at` (None if the game ended before).
    pub scv_at: Option<u32>,
    pub labels: Vec<LabelRow>,
    pub states: Vec<State>,
}

fn start_px(info: &GameInfo, enemy: bool) -> Option<(u32, u32)> {
    let p = info.players.iter().find(|p| if enemy { p.is_enemy } else { p.is_self || p.id == info.self_player })?;
    p.start_tile.map(|(x, y)| (x as u32 * 32 + 64, y as u32 * 32 + 48))
}

fn enemy_id(info: &GameInfo) -> Option<u8> {
    info.players.iter().find(|p| p.is_enemy).map(|p| p.id)
}

fn compact(f: &RawFrame, me: u8, enemy: Option<u8>) -> State {
    use uranium_proto::flags;
    let own = f
        .units
        .iter()
        .filter(|u| u.owner == me)
        .map(|u| (u.id, u.unit_type, u.pos.0, u.pos.1, u.flags, u.hp, u.loaded_in.is_some() || u.flags & flags::LOADED != 0))
        .collect();
    let enemy = f.units.iter().filter(|u| Some(u.owner) == enemy).map(|u| (u.id, u.unit_type, u.pos.0, u.pos.1)).collect();
    State { f: f.frame, own, enemy }
}

fn scv_type() -> u16 {
    uranium_env::types::TypeTables::get().unit_by_name("Terran_SCV").expect("Terran_SCV").id as u16
}

fn scvs(f: &RawFrame, me: u8) -> u32 {
    let scv = scv_type();
    f.units.iter().filter(|u| u.owner == me && u.unit_type == scv && u.flags & uranium_proto::flags::COMPLETED != 0).count() as u32
}

/// Keeps the states at command-label frames and at least one per `period` frames in the window.
fn pick_states(w: Window, frames: &BTreeMap<u32, State>, command_frames: &[u32]) -> Vec<State> {
    let mut keep: Vec<u32> = Vec::new();
    let mut next = w.lo;
    for &f in frames.keys() {
        if f >= next {
            keep.push(f);
            next = (f / w.period + 1) * w.period;
        }
    }
    for &c in command_frames {
        // The state the command was decided on: the last observation at or before it.
        if let Some((&f, _)) = frames.range(..=c).next_back() {
            keep.push(f);
        }
    }
    keep.sort_unstable();
    keep.dedup();
    keep.into_iter().filter_map(|f| frames.get(&f).cloned()).collect()
}

fn label_rows(labels: impl Iterator<Item = crate::label::Label>, w: Window) -> Vec<LabelRow> {
    labels.filter(|l| l.frame >= w.lo && l.frame <= w.hi).map(|l| (l.frame, l.action, l.delay, l.queued, l.selected, l.target_unit)).collect()
}

/// One shard game, or `None` when it fails the filter.
pub fn dump_shard_game(g: &ShardGame, name: String, w: Window, filt: &ShardFilter, noop: u16) -> Option<GameDump> {
    let m = &g.meta;
    if filt.matchup.as_ref().is_some_and(|x| *x != m.matchup) || filt.stage.as_ref().is_some_and(|x| *x != m.stage) {
        return None;
    }
    if !filt.families.is_empty() && !filt.families.contains(&m.map_family) {
        return None;
    }
    let info = &g.game_info;
    let me = info.self_player;
    let enemy = enemy_id(info);
    let mut frames = BTreeMap::new();
    let mut commands = Vec::new();
    let mut scv_at = None;
    for s in &g.steps {
        let f = s.raw_frame.frame;
        if f <= w.at {
            scv_at = Some(scvs(&s.raw_frame, me));
        }
        if f >= w.lo && f <= w.hi {
            frames.insert(f, compact(&s.raw_frame, me, enemy));
            if s.action_label.action != noop {
                commands.push(s.action_label.frame);
            }
        }
    }
    let end_frame = g.steps.last().map_or(0, |s| s.raw_frame.frame);
    // A game that ended (or was cut) well before `at` has no count there.
    if end_frame + 32 < w.at {
        scv_at = None;
    }
    Some(GameDump {
        source: m.src.clone(),
        game: name,
        replay_sha256: m.replay_sha256.clone(),
        matchup: m.matchup.clone(),
        map_family: m.map_family.clone(),
        stage: m.stage.clone(),
        result: m.result,
        end_frame,
        start: start_px(info, false),
        enemy_start: start_px(info, true),
        scv_at,
        labels: label_rows(g.steps.iter().map(|s| s.action_label.clone()), w),
        states: pick_states(w, &frames, &commands),
    })
}

/// One replay: the Terran player's own commands through the 040 labeler, observed in one OpenBW
/// replay pass (no 5-minute bar, no desync check).
pub fn dump_replay(path: &Path, cfg: &PlayConfig, labeler: &Labeler, w: Window, source: &str, noop: u16) -> Result<GameDump> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let rep = crate::parse(&bytes)?;
    let active = rep.active_players();
    let terran: Vec<_> = active.iter().filter(|(p, _)| p.race == RACE_TERRAN).collect();
    let [(view, _)] = terran[..] else { bail!("{} Terran players (want 1)", terran.len()) };
    let Some((other, _)) = active.iter().find(|(p, _)| p.id != view.id) else { bail!("no opponent") };
    let me = slot_of(&rep, view.id)?;
    let enemy = slot_of(&rep, other.id)?;
    let mut stops: Vec<u32> = (w.lo..=w.hi).step_by(w.period as usize).collect();
    stops.push(w.at);
    let mut units_at: HashMap<u32, HashMap<u32, Own>> = HashMap::new();
    let mut frames = BTreeMap::new();
    let mut scv_at = None;
    let res = play_with(&rep, view.id, cfg, &stops, &mut |f, _| {
        units_at.insert(f.frame, f.units.iter().filter(|u| u.owner == me).map(|u| (u.id, Own::from_state(u))).collect());
        if f.frame <= w.at {
            scv_at = Some(scvs(f, me));
        }
        if f.frame >= w.lo && f.frame <= w.hi {
            frames.insert(f.frame, compact(f, me, Some(enemy)));
        }
    })?;
    let end_frame = res.end_frame.unwrap_or(rep.header.frames).min(rep.header.frames);
    if end_frame + 32 < w.at {
        scv_at = None;
    }
    let cmds: Vec<cmd::Cmd> = rep
        .cmds()
        .filter(|c| c.player == view.id)
        .map(|c| cmd::to_legacy(c, rep.unit_limit))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|c| cmd::openbw(c.op) == cmd::OpenBw::Supported)
        .collect();
    let empty = HashMap::new();
    let (labels, _) = labeler.label(&cmds, rep.header.frames, |f| units_at.get(&f).unwrap_or(&empty));
    let commands: Vec<u32> = labels.iter().filter(|l| l.action != noop && l.frame >= w.lo && l.frame <= w.hi).map(|l| l.frame).collect();
    let (result, _) = crate::pipeline::judge(&res, None, me);
    let info = res.game_info.as_ref();
    use sha2::Digest;
    Ok(GameDump {
        source: source.to_string(),
        game: path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
        replay_sha256: sha2::Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect(),
        matchup: format!("Tv{}", crate::pipeline::race_letter(other.race)),
        map_family: String::new(),
        stage: String::new(),
        result,
        end_frame,
        start: info.and_then(|i| start_px(i, false)),
        enemy_start: info.and_then(|i| start_px(i, true)),
        scv_at,
        labels: label_rows(labels.into_iter(), w),
        states: pick_states(w, &frames, &commands),
    })
}

/// Expands folders into `shard_*.bin.zst` (shards) or `*.rep` (replays) files, sorted.
pub fn expand(inputs: &[PathBuf], replays: bool) -> Result<Vec<PathBuf>> {
    fn walk(d: &Path, replays: bool, out: &mut Vec<PathBuf>) -> Result<()> {
        for e in std::fs::read_dir(d).with_context(|| format!("listing {}", d.display()))? {
            let p = e?.path();
            if p.is_dir() {
                walk(&p, replays, out)?;
            } else {
                let name = p.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                let ok = if replays { name.to_ascii_lowercase().ends_with(".rep") } else { crate::shard::shard_index(&name).is_some() };
                if ok {
                    out.push(p);
                }
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    for i in inputs {
        if i.is_dir() {
            walk(i, replays, &mut out)?;
        } else {
            out.push(i.clone());
        }
    }
    out.sort();
    Ok(out)
}

/// Runs over `files` on `jobs` threads and writes gzipped JSON lines to `out`. Returns (games
/// written, files failed); a failed file is reported on stderr and skipped.
pub fn run(files: &[PathBuf], replays: bool, out: &Path, source: &str, w: Window, filt: &ShardFilter, jobs: usize) -> Result<(usize, usize)> {
    let noop = uranium_env::actions::ActionTable::load()?.by_name("NoOp").context("NoOp action")?.id;
    let cfg = if replays { Some(PlayConfig::from_env(crate::play::work_dir_default().join("granularity"))?) } else { None };
    let file = std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let writer = Mutex::new(flate2::write::GzEncoder::new(std::io::BufWriter::new(file), flate2::Compression::default()));
    let next = AtomicUsize::new(0);
    let (written, failed) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let emit = |d: &GameDump| -> Result<()> {
        let line = serde_json::to_string(d)?;
        let mut g = writer.lock().unwrap();
        g.write_all(line.as_bytes())?;
        g.write_all(b"\n")?;
        written.fetch_add(1, Ordering::Relaxed);
        Ok(())
    };
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| {
                let labeler = Labeler::new().expect("labeler tables");
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(p) = files.get(i) else { break };
                    let r = if replays {
                        dump_replay(p, cfg.as_ref().unwrap(), &labeler, w, source, noop).and_then(|d| emit(&d))
                    } else {
                        crate::shard::read(p).and_then(|sh| {
                            let stem = p.strip_prefix(p.ancestors().nth(2).unwrap_or(p)).unwrap_or(p).to_string_lossy().replace('\\', "/");
                            for (k, g) in sh.games.iter().enumerate() {
                                if let Some(d) = dump_shard_game(g, format!("{stem}#{k}"), w, filt, noop) {
                                    emit(&d)?;
                                }
                            }
                            Ok(())
                        })
                    };
                    if let Err(e) = r {
                        failed.fetch_add(1, Ordering::Relaxed);
                        eprintln!("{}: {e:#}", p.display());
                    }
                    if (i + 1).is_multiple_of(50) {
                        eprintln!("{} / {} files", i + 1, files.len());
                    }
                }
            });
        }
    });
    writer.into_inner().unwrap().finish()?.flush()?;
    Ok((written.into_inner(), failed.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(f: u32) -> State {
        State { f, own: vec![], enemy: vec![] }
    }

    /// One state per period in the window plus the state each command was decided on.
    #[test]
    fn pick_states_keeps_period_and_command_states() {
        let w = Window { lo: 100, hi: 300, period: 50, at: 1000 };
        let frames: BTreeMap<u32, State> = [100, 110, 120, 151, 160, 230, 260].into_iter().map(|f| (f, st(f))).collect();
        let kept: Vec<u32> = pick_states(w, &frames, &[125, 160]).iter().map(|s| s.f).collect();
        assert_eq!(kept, vec![100, 120, 151, 160, 230, 260]);
    }
}
