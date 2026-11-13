//! Branch games (spec 051): the engine-free formats and rules shared by the bridge sessions, the
//! plan tools and the analysis. Nothing here talks to a game or a model:
//! - observation digests (051-R2, 051-R5; the session `frame_digest` of 011-AC4 restarted per
//!   observation),
//! - the act log (051-R2) with its writer and checked reader,
//! - snapshot rules (051-R3) and the state file format (051-R3),
//! - interventions on the action-type mask (051-R7),
//! - the forced-prefix check of a branch player (051-R5, 051-R6 ①) and the prefix comparison of
//!   two logs (051-R12 `prefix`),
//! - outcome trackers and records (051-R9, 051-R10, 051-R12).

use std::collections::BTreeSet;
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};
use uranium_proto::{Act, EndReason, ProtoError, RawFrame, write_msg};

use crate::actions::ActionTable;
use crate::features::Observation;
use crate::features::label::MemoryAction;

/// Game frames per second (fastest speed): the 051-R7 / 051-R9 windows are given in seconds.
pub const FRAMES_PER_SECOND: u32 = 24;

// ---------------------------------------------------------------------------------------------
// Digests (051-R2, 051-R5, 011-AC4)

/// FNV-1a 64-bit offset basis: the start value of the session digest and of every observation
/// digest (051-R2).
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit fold of `bytes` into `h`. The agent session's `frame_digest` (011-AC4) uses this
/// function over [`frame_bytes`] of every observation; the observation digest (051-R2) is the
/// same fold restarted from [`FNV_OFFSET`] for each observation.
pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The bytes the session digest eats for one observation: the framed postcard message
/// (`u32 LE length + payload`, 010-R1) exactly as `write_msg` writes it (051-R2).
pub fn frame_bytes(f: &RawFrame) -> Result<Vec<u8>, ProtoError> {
    let mut bytes = Vec::new();
    write_msg(&mut bytes, f)?;
    Ok(bytes)
}

/// 051-R2: observation digest of already encoded [`frame_bytes`] (lets a session hash the bytes
/// it also folds into its `frame_digest`).
pub fn obs_digest_bytes(bytes: &[u8]) -> u64 {
    fnv1a(FNV_OFFSET, bytes)
}

/// 051-R2: observation digest, FNV-1a from the offset basis over [`frame_bytes`].
pub fn obs_digest(f: &RawFrame) -> Result<u64, ProtoError> {
    Ok(obs_digest_bytes(&frame_bytes(f)?))
}

/// Digest text form: 16 lowercase hex digits (as the session record's `frame_digest`).
pub fn digest_hex(d: u64) -> String {
    format!("{d:016x}")
}

pub fn parse_digest(s: &str) -> Result<u64> {
    ensure!(s.len() == 16, "digest `{s}` is not 16 hex digits");
    u64::from_str_radix(s, 16).with_context(|| format!("digest `{s}`"))
}

/// serde adapter: a `u64` digest as its 16-digit hex text.
mod hex_u64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(d: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&super::digest_hex(*d))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        let s = String::deserialize(d)?;
        super::parse_digest(&s).map_err(serde::de::Error::custom)
    }
}

/// Lowercase hex sha256 of `bytes` (state files 051-R3, checkpoints 051-R1 ④).
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------------------------
// Act log (051-R2)

/// 051-R2: the model session's decision on one observation: the same values as the 041-R28 ①
/// decision log (sampled action type, delay bucket, selected unit ids) plus whether the executor
/// refused it (031-R4).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub action: u16,
    pub delay_bucket: u32,
    pub units: Vec<u32>,
    pub rejected: bool,
}

impl Decision {
    pub fn new(m: &MemoryAction, rejected: bool) -> Self {
        Self { action: m.action, delay_bucket: m.delay_bucket, units: m.units.clone(), rejected }
    }

    /// What a forced prefix records in agent memory (051-R5 ②, `AgentMemory::record_action`).
    pub fn memory_action(&self) -> MemoryAction {
        MemoryAction { action: self.action, delay_bucket: self.delay_bucket, units: self.units.clone() }
    }
}

/// 051-R2: one observation line of an act log.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ObsLine {
    /// Observation index from 0.
    pub i: u32,
    /// `RawFrame.frame`.
    pub frame: u32,
    /// [`obs_digest`] of the observation.
    #[serde(with = "hex_u64")]
    pub digest: u64,
    /// The whole `Act` sent on this observation (every `Command` field, `advance_frames`,
    /// `valid_until_frame`, `surrender`).
    pub act: Act,
    /// Model sessions only.
    #[serde(default)]
    pub decision: Option<Decision>,
    /// Model sessions only: the state id saved before this decision (051-R3), `null` if none.
    #[serde(default)]
    pub state: Option<String>,
    /// sha256 of that state file's content (051-R3).
    #[serde(default)]
    pub state_sha256: Option<String>,
    /// 051-R5 ⑤: the decision was forced from a trunk log (branch prefix).
    #[serde(default)]
    pub forced: bool,
}

impl Serialize for ObsLine {
    /// Model lines (with `decision`) always carry `state` (`null` when none, 051-R2); T0 lines
    /// carry neither (051 Interfaces).
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(None)?;
        m.serialize_entry("i", &self.i)?;
        m.serialize_entry("frame", &self.frame)?;
        m.serialize_entry("digest", &digest_hex(self.digest))?;
        m.serialize_entry("act", &self.act)?;
        if let Some(d) = &self.decision {
            m.serialize_entry("decision", d)?;
        }
        if self.decision.is_some() || self.state.is_some() {
            m.serialize_entry("state", &self.state)?;
        }
        if let Some(h) = &self.state_sha256 {
            m.serialize_entry("state_sha256", h)?;
        }
        m.serialize_entry("forced", &self.forced)?;
        m.end()
    }
}

/// 051-R2: the end line, written only when the game ended (`GameEnd`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ActLogEnd {
    /// Observation lines written.
    pub obs: u64,
    /// Lines with a `decision`.
    pub decisions: u64,
    pub end_frame: u32,
    pub reason: EndReason,
    /// Own BWAPI player id.
    pub player: u8,
    /// Index into `GameSpec.players`.
    pub players_index: u8,
    /// 020-R5 policy version (`t0@<ver>` or `model:<checkpoint id>:<sample>`).
    pub policy: String,
    pub git: String,
    pub feature_schema: u32,
    pub action_schema: u32,
}

#[derive(Serialize, Deserialize)]
struct EndWrap<E> {
    end: E,
}

/// A complete act log (051-R2): every observation line and the end line.
#[derive(Debug, Clone, PartialEq)]
pub struct ActLog {
    pub lines: Vec<ObsLine>,
    pub end: ActLogEnd,
}

impl ActLog {
    /// Reads and checks a log. A log without its end line, with lines after it, with observation
    /// indices that do not count up from 0, with frames that do not increase, or whose end line
    /// counts other observations or decisions than it holds is incomplete and refused (051-R2).
    pub fn read(r: impl BufRead) -> Result<Self> {
        let mut lines: Vec<ObsLine> = Vec::new();
        let mut end: Option<ActLogEnd> = None;
        for (n, text) in r.lines().enumerate() {
            let text = text?;
            if text.trim().is_empty() {
                continue;
            }
            let ctx = || format!("act log line {}", n + 1);
            ensure!(end.is_none(), "{}: line after the end line", ctx());
            let v: serde_json::Value = serde_json::from_str(&text).with_context(ctx)?;
            if v.get("end").is_some() {
                end = Some(serde_json::from_value::<EndWrap<ActLogEnd>>(v).with_context(ctx)?.end);
                continue;
            }
            let line: ObsLine = serde_json::from_value(v).with_context(ctx)?;
            ensure!(line.i as usize == lines.len(), "{}: observation index {} (expected {})", ctx(), line.i, lines.len());
            if let Some(prev) = lines.last() {
                ensure!(line.frame > prev.frame, "{}: frame {} not after {}", ctx(), line.frame, prev.frame);
            }
            lines.push(line);
        }
        let Some(end) = end else { bail!("act log without its end line (incomplete, 051-R2)") };
        let decisions = lines.iter().filter(|l| l.decision.is_some()).count() as u64;
        ensure!(end.obs == lines.len() as u64, "end line counts {} observations, the log holds {}", end.obs, lines.len());
        ensure!(end.decisions == decisions, "end line counts {} decisions, the log holds {decisions}", end.decisions);
        Ok(Self { lines, end })
    }

    pub fn read_path(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        Self::read(std::io::BufReader::new(f)).with_context(|| format!("act log {}", path.display()))
    }

    pub fn write(&self, w: impl Write) -> Result<()> {
        let mut out = ActLogWriter::new(w);
        for l in &self.lines {
            out.push(l)?;
        }
        out.finish(self.end.clone())?;
        Ok(())
    }

    /// The observation line with index `i`.
    pub fn line(&self, i: u32) -> Option<&ObsLine> {
        self.lines.get(i as usize)
    }
}

/// Streaming act-log writer for a live session (051-R2): one JSON line per observation, then the
/// end line from [`finish`](Self::finish).
pub struct ActLogWriter<W: Write> {
    w: W,
    obs: u64,
    decisions: u64,
}

impl<W: Write> ActLogWriter<W> {
    pub fn new(w: W) -> Self {
        Self { w, obs: 0, decisions: 0 }
    }

    /// Appends one observation line; its index must be the next one.
    pub fn push(&mut self, line: &ObsLine) -> Result<()> {
        ensure!(line.i as u64 == self.obs, "act log: observation index {} (expected {})", line.i, self.obs);
        serde_json::to_writer(&mut self.w, line)?;
        self.w.write_all(b"\n")?;
        self.obs += 1;
        self.decisions += line.decision.is_some() as u64;
        Ok(())
    }

    /// Observation lines written so far.
    pub fn observations(&self) -> u64 {
        self.obs
    }

    /// Writes the end line with the counts of the lines written (the `obs` / `decisions` given in
    /// `end` are replaced) and flushes. Only on `GameEnd`: a log cut short stays without it.
    pub fn finish(mut self, mut end: ActLogEnd) -> Result<W> {
        end.obs = self.obs;
        end.decisions = self.decisions;
        serde_json::to_writer(&mut self.w, &EndWrap { end: &end })?;
        self.w.write_all(b"\n")?;
        self.w.flush()?;
        Ok(self.w)
    }
}

/// 020-R22 ①: act-log file name of one session (`g<plan game>-p<players index>.jsonl`).
pub fn act_log_name(plan_game: u32, players_index: u8) -> String {
    format!("g{plan_game}-p{players_index}.jsonl")
}

// ---------------------------------------------------------------------------------------------
// Forced prefix (051-R5, 051-R6 ①)

/// 051-R12 `invalid` reasons.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvalidReason {
    /// Observation digest differs from the trunk (051-R5 ①).
    Digest,
    /// Observation frame differs from the trunk (051-R5 ①).
    Frame,
    /// The state file could not be checked or restored (051-R6 ②).
    State,
    /// A trunk input (act log, state file) is missing or incomplete.
    MissingInput,
}

/// Why a branch is invalid, at which observation (051-R5 ①, 051-R12).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Invalid {
    pub reason: InvalidReason,
    pub obs: Option<u32>,
    pub frame: Option<u32>,
}

/// What the branch player does on one observation.
#[derive(Debug, Clone, PartialEq)]
pub enum PrefixStep<'a> {
    /// i < i_t: matches the trunk; send `line.act` unchanged, record `line.decision` in memory,
    /// no inference (051-R5 ②③).
    Forced(&'a ObsLine),
    /// i = i_t: matches the trunk; restore the state and play the intervention (051-R6). The
    /// trunk's own line at t is given for reference (its decision is the trunk action).
    AtT(&'a ObsLine),
    /// i > i_t: play normally (051-R6 ③).
    Live,
}

/// The branch player's prefix check against its own trunk act log (051-R5 ①, 051-R6 ①).
#[derive(Debug, Clone)]
pub struct ForcedPrefix {
    log: ActLog,
    i_t: u32,
    matched: u32,
}

impl ForcedPrefix {
    /// `log` is the trunk's act log of this model session; t is observation `i_t` at frame `f_t`.
    /// Every line up to t must hold a decision (a model session's log).
    pub fn new(log: ActLog, i_t: u32, f_t: u32) -> Result<Self> {
        let at = log.line(i_t).with_context(|| format!("trunk act log has no observation {i_t}"))?;
        ensure!(at.frame == f_t, "trunk observation {i_t} is at frame {}, not f_t {f_t}", at.frame);
        ensure!(log.lines[..=i_t as usize].iter().all(|l| l.decision.is_some()), "trunk act log is not a model session's (no decision before t)");
        Ok(Self { log, i_t, matched: 0 })
    }

    pub fn i_t(&self) -> u32 {
        self.i_t
    }

    /// Prefix observations (i <= i_t) that matched the trunk so far.
    pub fn matched(&self) -> u32 {
        self.matched
    }

    /// Checks observation `i` (`frame`, its [`obs_digest`]). Up to t a frame or digest different
    /// from the trunk makes the branch invalid (end the game with [`quit_act`]).
    pub fn step(&mut self, i: u32, frame: u32, digest: u64) -> Result<PrefixStep<'_>, Invalid> {
        if i > self.i_t {
            return Ok(PrefixStep::Live);
        }
        let Some(line) = self.log.lines.get(i as usize) else {
            return Err(Invalid { reason: InvalidReason::MissingInput, obs: Some(i), frame: Some(frame) });
        };
        let reason = if line.frame != frame {
            Some(InvalidReason::Frame)
        } else if line.digest != digest {
            Some(InvalidReason::Digest)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(Invalid { reason, obs: Some(i), frame: Some(frame) });
        }
        self.matched += 1;
        Ok(if i < self.i_t { PrefixStep::Forced(line) } else { PrefixStep::AtT(line) })
    }
}

/// 051-R5 ①: leave without chat (010-R13 `surrender` with an empty text).
pub fn quit_act() -> Act {
    Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some(String::new()) }
}

/// 051-R12 `prefix`: a branch session's log against the trunk's for the same player.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrefixCompare {
    /// Trunk observations with frame < the given frame.
    pub obs: u32,
    /// Of those, observations the branch saw identically (same index, frame and digest).
    pub matched: u32,
    /// Frame of the first observation that differs over the whole logs (the branch side's, or the
    /// trunk's when the branch log is shorter); `None` when one log is the other.
    pub first_mismatch_frame: Option<u32>,
}

/// Compares two act logs observation by observation: the prefix before `before_frame` (f_t) and
/// the first divergence (051-AC5: must come after t).
pub fn compare_prefix(trunk: &[ObsLine], branch: &[ObsLine], before_frame: u32) -> PrefixCompare {
    let same = |a: &ObsLine, b: &ObsLine| a.frame == b.frame && a.digest == b.digest;
    let prefix: Vec<&ObsLine> = trunk.iter().take_while(|l| l.frame < before_frame).collect();
    let matched = prefix.iter().zip(branch).filter(|(a, b)| same(a, b)).count() as u32;
    let first = trunk.iter().zip(branch).position(|(a, b)| !same(a, b)).or((trunk.len() != branch.len()).then(|| trunk.len().min(branch.len())));
    let first_mismatch_frame = first.map(|i| branch.get(i).or(trunk.get(i)).map_or(0, |l| l.frame));
    PrefixCompare { obs: prefix.len() as u32, matched, first_mismatch_frame }
}

// ---------------------------------------------------------------------------------------------
// Snapshots and state files (051-R3)

/// 051-R3 / 020-R22 ②: one `[[snapshots]]` rule of a plan.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRule {
    pub name: String,
    /// Window [from_frame, to_frame) over the decision observation's frame.
    pub from_frame: u32,
    pub to_frame: u32,
    /// 034 action names: the decision's `type_mask` must open at least one. Empty = no condition.
    #[serde(default)]
    pub any_open: Vec<String>,
}

/// 051-R3: checks the snapshot rules on every decision; each rule fires on the first decision in
/// its window that meets its condition, and only then.
#[derive(Debug, Clone)]
pub struct Snapshots {
    rules: Vec<(SnapshotRule, Vec<u16>)>,
    /// Observation index each rule fired at.
    fired: Vec<Option<u32>>,
}

impl Snapshots {
    pub fn new(rules: &[SnapshotRule], table: &ActionTable) -> Result<Self> {
        let mut names = BTreeSet::new();
        let mut out = Vec::with_capacity(rules.len());
        for r in rules {
            ensure!(!r.name.is_empty(), "snapshot rule without a name");
            ensure!(names.insert(r.name.as_str()), "snapshot rule `{}` given twice", r.name);
            ensure!(r.from_frame < r.to_frame, "snapshot rule `{}`: empty window [{}, {})", r.name, r.from_frame, r.to_frame);
            let ids = r.any_open.iter().map(|a| action_id(table, a).with_context(|| format!("snapshot rule `{}`", r.name))).collect::<Result<Vec<_>>>()?;
            out.push((r.clone(), ids));
        }
        Ok(Self { fired: vec![None; out.len()], rules: out })
    }

    /// Checks decision observation `obs` at `frame` (after feature extraction, before inference).
    /// Returns the names of the rules that fire on it; non-empty = save one state file for this
    /// decision, shared by every rule named (051 Interfaces).
    pub fn check(&mut self, obs: u32, frame: u32, type_mask: &[bool]) -> Vec<String> {
        let mut hit = vec![];
        for ((r, ids), fired) in self.rules.iter().zip(&mut self.fired) {
            if fired.is_some() || frame < r.from_frame || frame >= r.to_frame {
                continue;
            }
            if ids.is_empty() || ids.iter().any(|&a| type_mask.get(a as usize).copied().unwrap_or(false)) {
                *fired = Some(obs);
                hit.push(r.name.clone());
            }
        }
        hit
    }

    /// (rule name, observation index it fired at).
    pub fn fired(&self) -> impl Iterator<Item = (&str, Option<u32>)> {
        self.rules.iter().zip(&self.fired).map(|((r, _), f)| (r.name.as_str(), *f))
    }
}

/// 020-R22 ②: state id (and file stem, `.state`) of the decision at observation `obs`.
pub fn state_id(plan_game: u32, obs: u32) -> String {
    format!("g{plan_game}-o{obs}")
}

const STATE_MAGIC: &[u8; 4] = b"URST";
const STATE_VERSION: u32 = 1;

/// 051-R3 state file: one slot's Core state as `HostState` parts (032-R16, `(shape, values)`).
/// Layout (little endian): `"URST"`, version u32, part count u32, then per part: rank u32, the
/// dimensions u32 each, the f32 values. Round trips are bit-exact (NaN payloads, -0.0 included).
pub fn encode_state(parts: &[(Vec<usize>, Vec<f32>)]) -> Result<Vec<u8>> {
    let u32_of = |v: usize, what: &str| u32::try_from(v).with_context(|| format!("state {what} {v} exceeds u32"));
    let mut out = Vec::with_capacity(12 + parts.iter().map(|(s, d)| 4 + 4 * s.len() + 4 * d.len()).sum::<usize>());
    out.extend_from_slice(STATE_MAGIC);
    out.extend_from_slice(&STATE_VERSION.to_le_bytes());
    out.extend_from_slice(&u32_of(parts.len(), "part count")?.to_le_bytes());
    for (k, (shape, data)) in parts.iter().enumerate() {
        let n: usize = shape.iter().product();
        ensure!(n == data.len(), "state part {k}: shape {shape:?} holds {n} values, given {}", data.len());
        out.extend_from_slice(&u32_of(shape.len(), "rank")?.to_le_bytes());
        for &d in shape {
            out.extend_from_slice(&u32_of(d, "dimension")?.to_le_bytes());
        }
        for v in data {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    Ok(out)
}

/// Reads an [`encode_state`] file; refuses a bad header, truncation and trailing bytes.
pub fn decode_state(bytes: &[u8]) -> Result<Vec<(Vec<usize>, Vec<f32>)>> {
    struct Cursor<'a>(&'a [u8]);
    impl Cursor<'_> {
        fn word(&mut self) -> Result<u32> {
            ensure!(self.0.len() >= 4, "state file truncated");
            let (w, rest) = self.0.split_at(4);
            self.0 = rest;
            Ok(u32::from_le_bytes(w.try_into().unwrap()))
        }
    }
    let mut c = Cursor(bytes);
    ensure!(c.word()?.to_le_bytes() == *STATE_MAGIC, "not a state file");
    let version = c.word()?;
    ensure!(version == STATE_VERSION, "state file version {version}, expected {STATE_VERSION}");
    let parts = c.word()?;
    let mut out = Vec::new();
    for _ in 0..parts {
        let rank = c.word()? as usize;
        let shape = (0..rank).map(|_| c.word().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
        let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).context("state part too large")?;
        ensure!(n <= c.0.len() / 4, "state file truncated");
        let data = (0..n).map(|_| c.word().map(f32::from_bits)).collect::<Result<Vec<_>>>()?;
        out.push((shape, data));
    }
    ensure!(c.0.is_empty(), "state file has {} trailing bytes", c.0.len());
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Interventions (051-R7)

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    /// Narrow the first decision in the window where `action` is open to {action}.
    Force,
    /// Close `action` on every decision in the window.
    Ban,
}

/// 051-R7: one rule `{op, action, from_s, to_s}`; window [f_t + from_s × 24, f_t + to_s × 24).
/// `from_s = to_s = 0` is the t decision only.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionRule {
    pub op: Op,
    /// 034 action name.
    pub action: String,
    pub from_s: u32,
    pub to_s: u32,
}

impl InterventionRule {
    /// The rule acts on the t decision only (051-R7 ①).
    pub fn t_only(&self) -> bool {
        self.from_s == 0 && self.to_s == 0
    }
}

/// 051-R7: an alternative is a named rule list; no rules = no intervention (`none`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Alternative {
    pub name: String,
    #[serde(default)]
    pub rules: Vec<InterventionRule>,
}

impl Alternative {
    pub fn is_none(&self) -> bool {
        self.rules.is_empty()
    }

    /// 051-R10: actions of t-only `force` rules that are closed in the t observation's
    /// `type_mask` (the alternative does not apply at that point).
    pub fn closed_t_forces(&self, table: &ActionTable, type_mask: &[bool]) -> Result<Vec<String>> {
        let mut out = vec![];
        for r in self.rules.iter().filter(|r| r.op == Op::Force && r.t_only()) {
            let id = action_id(table, &r.action)?;
            if !type_mask.get(id as usize).copied().unwrap_or(false) {
                out.push(r.action.clone());
            }
        }
        Ok(out)
    }
}

/// 051-R7 ⑤: what one rule did in a branch game.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RuleLog {
    #[serde(flatten)]
    pub rule: InterventionRule,
    /// Decisions inside the window the rule was checked on (a `force` stops counting once it
    /// fired).
    pub in_window: u32,
    /// Decisions whose mask the rule changed (`ban`: closed an open bit; `force`: the forced
    /// decision).
    pub applied: u32,
    /// `force`: frame of the forced decision.
    pub force_frame: Option<u32>,
    /// `force`: the window ended (or the game did) without `action` open.
    pub missed: bool,
    /// `force`: the executor refused the forced decision (031-R4); `None` until known.
    pub rejected: Option<bool>,
}

#[derive(Debug, Clone)]
struct ActiveRule {
    id: u16,
    start: u32,
    end: u32,
    log: RuleLog,
}

impl ActiveRule {
    fn in_window(&self, frame: u32) -> bool {
        (self.start..self.end).contains(&frame)
    }
}

/// 051-R7 rule engine for one branch session from t on. Interventions only close open bits of
/// `type_mask`; NoOp (034-R2 id 0) is closed only by a `force`, so the mask never becomes empty.
/// Bans apply before forces, so a force fires only on an action still open after the bans.
#[derive(Debug, Clone)]
pub struct Intervention {
    rules: Vec<ActiveRule>,
}

impl Intervention {
    pub fn new(alt: &Alternative, table: &ActionTable, f_t: u32) -> Result<Self> {
        let mut rules = Vec::with_capacity(alt.rules.len());
        for r in &alt.rules {
            let ctx = || format!("alternative `{}` rule {} {}", alt.name, op_name(r.op), r.action);
            let id = action_id(table, &r.action).with_context(ctx)?;
            ensure!(r.from_s < r.to_s || r.t_only(), "{}: empty window {}..{} s", ctx(), r.from_s, r.to_s);
            let (start, end) = if r.t_only() {
                (f_t, f_t + 1)
            } else {
                (f_t + r.from_s * FRAMES_PER_SECOND, f_t + r.to_s * FRAMES_PER_SECOND)
            };
            let log = RuleLog { rule: r.clone(), in_window: 0, applied: 0, force_frame: None, missed: false, rejected: None };
            rules.push(ActiveRule { id, start, end, log });
        }
        Ok(Self { rules })
    }

    /// Narrows the decision's `type_mask` at `frame` (decisions from t on, in order). The same
    /// narrowed mask goes to the model and the executor (051-R7 ④).
    pub fn apply(&mut self, frame: u32, type_mask: &[bool]) -> Vec<bool> {
        let mut m = type_mask.to_vec();
        for r in self.rules.iter_mut().filter(|r| r.log.rule.op == Op::Ban) {
            if !r.in_window(frame) {
                continue;
            }
            r.log.in_window += 1;
            let i = r.id as usize;
            // 051-R7 ③: NoOp stays open under a ban.
            if r.id != 0 && m.get(i).copied().unwrap_or(false) {
                m[i] = false;
                r.log.applied += 1;
            }
        }
        for r in self.rules.iter_mut().filter(|r| r.log.rule.op == Op::Force) {
            if r.log.force_frame.is_some() || r.log.missed {
                continue;
            }
            if frame >= r.end {
                r.log.missed = true;
                continue;
            }
            if !r.in_window(frame) {
                continue;
            }
            r.log.in_window += 1;
            let i = r.id as usize;
            if m.get(i).copied().unwrap_or(false) {
                m.iter_mut().enumerate().for_each(|(k, b)| *b = k == i);
                r.log.force_frame = Some(frame);
                r.log.applied += 1;
            }
        }
        debug_assert!(m.iter().any(|&b| b) || !type_mask.iter().any(|&b| b), "051-R7: intervention emptied the mask");
        m
    }

    /// [`apply`](Self::apply) on an observation: only `type_mask` changes; `unit_cap`,
    /// `loc_class_mask` and the model inputs stay (051-R7 ④).
    pub fn apply_to(&mut self, frame: u32, obs: &mut Observation) {
        obs.masks.type_mask = self.apply(frame, &obs.masks.type_mask);
    }

    /// Records the executor's verdict on the decision at `frame` for a `force` that fired there
    /// (051-R7 ⑤).
    pub fn note_executor(&mut self, frame: u32, rejected: bool) {
        for r in &mut self.rules {
            if r.log.force_frame == Some(frame) {
                r.log.rejected = Some(rejected);
            }
        }
    }

    /// Game over: a `force` that never fired is `missed` even if its window had not ended.
    pub fn finish(&mut self) {
        for r in &mut self.rules {
            if r.log.rule.op == Op::Force && r.log.force_frame.is_none() {
                r.log.missed = true;
            }
        }
    }

    pub fn log(&self) -> Vec<RuleLog> {
        self.rules.iter().map(|r| r.log.clone()).collect()
    }
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Force => "force",
        Op::Ban => "ban",
    }
}

fn action_id(table: &ActionTable, name: &str) -> Result<u16> {
    Ok(table.by_name(name).with_context(|| format!("unknown action `{name}` (034)"))?.id)
}

/// Open action ids of a `type_mask` (051 `points.jsonl` `type_mask`).
pub fn open_ids(type_mask: &[bool]) -> Vec<u16> {
    type_mask.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u16).collect()
}

// ---------------------------------------------------------------------------------------------
// Outcome trackers (051-R9)

/// One time-point sample: the decision observation's frame and its whole 030 §4 scalar vector.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Sample {
    pub frame: u32,
    pub scalar: Vec<f32>,
}

/// 051-R9 ①: `None` when the game ended before the sample.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Samples {
    /// First decision with frame >= f_t + 60 / 120 / 240 s.
    pub t60: Option<Sample>,
    pub t120: Option<Sample>,
    pub t240: Option<Sample>,
    /// First decision with frame >= H.
    pub horizon: Option<Sample>,
    /// The 8th / 32nd decision after t.
    pub d8: Option<Sample>,
    pub d32: Option<Sample>,
}

/// 051-R9 ① sampler, fed every decision from t on.
#[derive(Debug, Clone)]
pub struct SampleTracker {
    f_t: u32,
    horizon: u32,
    samples: Samples,
}

impl SampleTracker {
    pub fn new(f_t: u32, horizon: u32) -> Self {
        Self { f_t, horizon, samples: Samples::default() }
    }

    /// `k`: decisions after t (0 = the t decision).
    pub fn on_decision(&mut self, k: u32, frame: u32, scalar: &[f32]) {
        let s = || Some(Sample { frame, scalar: scalar.to_vec() });
        let f_t = self.f_t;
        let x = &mut self.samples;
        for (slot, secs) in [(&mut x.t60, 60), (&mut x.t120, 120), (&mut x.t240, 240)] {
            if slot.is_none() && frame >= f_t + secs * FRAMES_PER_SECOND {
                *slot = s();
            }
        }
        if x.horizon.is_none() && frame >= self.horizon {
            x.horizon = s();
        }
        if k == 8 {
            x.d8 = s();
        }
        if k == 32 {
            x.d32 = s();
        }
    }

    pub fn samples(&self) -> &Samples {
        &self.samples
    }
}

/// 051-R9 ③: how the policy itself picked an alternative's action after t.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DilutionEntry {
    pub action: String,
    /// First decision after t (t excluded) that sampled it.
    pub first_frame: Option<u32>,
    /// Decisions in (f_t, f_t + 30 s) that sampled it.
    pub picks_30s: u32,
}

/// 051-R9 ③ dilution check over decisions after t (the t decision excluded).
#[derive(Debug, Clone)]
pub struct DilutionTracker {
    f_t: u32,
    entries: Vec<(u16, DilutionEntry)>,
}

impl DilutionTracker {
    /// The actions of `alt`'s rules, or `none_actions` (the batch's alternative actions, from the
    /// plan's configuration) when `alt` is `none`.
    pub fn new(alt: &Alternative, none_actions: &[String], table: &ActionTable, f_t: u32) -> Result<Self> {
        let names: Vec<&String> = if alt.is_none() { none_actions.iter().collect() } else { alt.rules.iter().map(|r| &r.action).collect() };
        let mut seen = BTreeSet::new();
        let mut entries = vec![];
        for n in names {
            if seen.insert(n.as_str()) {
                entries.push((action_id(table, n)?, DilutionEntry { action: n.clone(), first_frame: None, picks_30s: 0 }));
            }
        }
        Ok(Self { f_t, entries })
    }

    /// One sampled decision; decisions at or before f_t are ignored.
    pub fn on_decision(&mut self, frame: u32, action: u16) {
        if frame <= self.f_t {
            return;
        }
        for (_, e) in self.entries.iter_mut().filter(|(id, _)| *id == action) {
            e.first_frame.get_or_insert(frame);
            if frame < self.f_t + 30 * FRAMES_PER_SECOND {
                e.picks_30s += 1;
            }
        }
    }

    pub fn entries(&self) -> Vec<DilutionEntry> {
        self.entries.iter().map(|(_, e)| e.clone()).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Records (051-R9, 051-R10, 051-R12)

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GameResult {
    Win,
    Loss,
    Draw,
}

/// 051-R9 ④.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BranchEnd {
    pub result: GameResult,
    pub end_frame: u32,
    pub reason: EndReason,
    /// The game ended before the horizon H.
    pub before_horizon: bool,
}

impl BranchEnd {
    /// `winner` / `me`: BWAPI player ids (`GameEnd.winner`, the session's slot).
    pub fn new(winner: Option<u8>, me: u8, end_frame: u32, reason: EndReason, horizon: u32) -> Self {
        let result = match winner {
            None => GameResult::Draw,
            Some(w) if w == me => GameResult::Win,
            Some(_) => GameResult::Loss,
        };
        Self { result, end_frame, reason, before_horizon: end_frame < horizon }
    }
}

/// 051-R6 optional `verify_state` comparison at t.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StateCheck {
    pub bit_equal: bool,
    pub max_abs_diff: f64,
}

/// 051-R9: what a branch game's model session leaves in its session result.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BranchOutcome {
    pub valid: bool,
    pub invalid: Option<Invalid>,
    /// 051-R5: prefix observations (i <= i_t) seen and matched.
    pub prefix_obs: u32,
    pub prefix_matched: u32,
    pub state_check: Option<StateCheck>,
    pub intervention: Vec<RuleLog>,
    pub samples: Samples,
    /// 050-R7 session metrics, 042 scouting / response metrics, `held_s`, `held_8m` (051-R9 ②),
    /// as the model session computes them.
    pub metrics: serde_json::Map<String, serde_json::Value>,
    pub dilution: Vec<DilutionEntry>,
    pub end: Option<BranchEnd>,
}

/// 051-R10: one line of a branch list (`branches`, one per plan game, in plan order).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BranchPlanLine {
    /// `<job>/<stage>/g<n>` of the trunk game.
    pub trunk: String,
    /// `<trunk>#<snapshot rule>`.
    pub point: String,
    /// Trunk act logs (ours, the opponent's), relative to the shared drive (061).
    pub acts: String,
    pub opponent_acts: String,
    /// State file (relative to the shared drive) and its content sha256.
    pub state: String,
    pub state_sha256: String,
    pub i_t: u32,
    pub f_t: u32,
    /// The trunk game's seed, map and opponent (the plan game must match, 020-R22 ③).
    pub seed: u32,
    pub map: String,
    pub opponent: String,
    pub alt: Alternative,
    pub repeat: u32,
}

/// 051-R12 `trunk` of `points.jsonl`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TrunkRef {
    pub job: String,
    pub stage: String,
    pub plan_game: u32,
    pub hostd_game: u64,
    pub replay: String,
    pub acts: String,
    pub opponent_acts: String,
}

/// 030 / 034 schema versions (051-R1 ④).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schemas {
    pub feature: u32,
    pub action: u32,
}

/// 051-R12 `points.jsonl`: one line per decision point.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PointRecord {
    pub point: String,
    pub trunk: TrunkRef,
    pub seed: u32,
    pub map: String,
    pub opponent: String,
    /// 020-R21 first-target condition (`hit` / `miss`).
    pub condition: Option<String>,
    pub checkpoint_sha256: String,
    pub policy: String,
    pub t0_policy: String,
    pub git: String,
    pub schemas: Schemas,
    pub i_t: u32,
    pub f_t: u32,
    #[serde(with = "hex_u64")]
    pub digest_t: u64,
    /// Open action ids of the t observation.
    pub type_mask: Vec<u16>,
    pub scalar_t: Vec<f32>,
    pub trunk_action: Decision,
    pub state: String,
    pub state_sha256: String,
    /// The trunk's 051-R9 ② ④ values (control only, not an analysis sample, 051-R11).
    pub trunk_outcome: serde_json::Map<String, serde_json::Value>,
    pub valid: bool,
    pub skip_reason: Option<String>,
}

/// 051-R12 `game` of `branches.jsonl`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GameRef {
    pub job: String,
    pub stage: String,
    pub plan_game: u32,
    pub hostd_game: u64,
    pub attempt: u32,
    pub replay: Option<String>,
    /// Game host.
    pub host: String,
    /// Inference backend and precision (032-R8), inference server seed (051-R8).
    pub backend: String,
    pub precision: String,
    pub infer_seed: Option<u64>,
}

/// 051-R12 `prefix` of `branches.jsonl`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrefixRecord {
    pub obs: u32,
    pub matched: u32,
    /// The opponent's logs compared after the game ([`compare_prefix`]).
    pub opponent: Option<PrefixCompare>,
}

/// 051-R12 `branches.jsonl`: one line per branch game.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BranchRecord {
    pub point: String,
    pub alt: String,
    pub rules: Vec<InterventionRule>,
    pub repeat: u32,
    pub game: GameRef,
    pub checkpoint_sha256: String,
    pub git: String,
    pub valid: bool,
    pub invalid: Option<Invalid>,
    pub prefix: PrefixRecord,
    pub state_check: Option<StateCheck>,
    pub intervention: Vec<RuleLog>,
    pub samples: Samples,
    pub metrics: serde_json::Map<String, serde_json::Value>,
    pub dilution: Vec<DilutionEntry>,
    pub end: Option<BranchEnd>,
}

/// Reads a JSON-lines file of `T` (blank lines skipped).
pub fn read_jsonl<T: serde::de::DeserializeOwned>(r: impl BufRead) -> Result<Vec<T>> {
    let mut out = vec![];
    for (n, line) in r.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).with_context(|| format!("line {}", n + 1))?);
    }
    Ok(out)
}

/// Writes `rows` as JSON lines.
pub fn write_jsonl<T: Serialize>(mut w: impl Write, rows: &[T]) -> Result<()> {
    for r in rows {
        serde_json::to_writer(&mut w, r)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests;
