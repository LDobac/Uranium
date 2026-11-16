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

pub use crate::types::FRAMES_PER_SECOND;

pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

pub fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn frame_bytes(f: &RawFrame) -> Result<Vec<u8>, ProtoError> {
    let mut bytes = Vec::new();
    write_msg(&mut bytes, f)?;
    Ok(bytes)
}

pub fn obs_digest_bytes(bytes: &[u8]) -> u64 {
    fnv1a(FNV_OFFSET, bytes)
}

pub fn obs_digest(f: &RawFrame) -> Result<u64, ProtoError> {
    Ok(obs_digest_bytes(&frame_bytes(f)?))
}

pub fn digest_hex(d: u64) -> String {
    format!("{d:016x}")
}

pub fn parse_digest(s: &str) -> Result<u64> {
    ensure!(s.len() == 16, "digest `{s}` is not 16 hex digits");
    u64::from_str_radix(s, 16).with_context(|| format!("digest `{s}`"))
}

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

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

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

    pub fn memory_action(&self) -> MemoryAction {
        MemoryAction { action: self.action, delay_bucket: self.delay_bucket, units: self.units.clone() }
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct ObsLine {
    pub i: u32,
    pub frame: u32,
    #[serde(with = "hex_u64")]
    pub digest: u64,
    pub act: Act,
    #[serde(default)]
    pub decision: Option<Decision>,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub state_sha256: Option<String>,
    #[serde(default)]
    pub forced: bool,
}

impl ObsLine {
    pub fn check_state(&self) -> Result<()> {
        ensure!(self.state.is_some() == self.state_sha256.is_some(), "observation {}: `state` and `state_sha256` must be given together (051-R3)", self.i);
        Ok(())
    }
}

impl Serialize for ObsLine {
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ActLogEnd {
    pub obs: u64,
    pub decisions: u64,
    pub end_frame: u32,
    pub reason: EndReason,
    pub player: u8,
    pub players_index: u8,
    pub policy: String,
    pub git: String,
    pub feature_schema: u32,
    pub action_schema: u32,
}

#[derive(Serialize, Deserialize)]
struct EndWrap<E> {
    end: E,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActLog {
    pub lines: Vec<ObsLine>,
    pub end: ActLogEnd,
}

impl ActLog {
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
            line.check_state().with_context(ctx)?;
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
        if let Some(last) = lines.last() {
            ensure!(end.end_frame >= last.frame, "end frame {} before the last observation's frame {}", end.end_frame, last.frame);
        }
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

    pub fn line(&self, i: u32) -> Option<&ObsLine> {
        self.lines.get(i as usize)
    }
}

pub struct ActLogWriter<W: Write> {
    w: W,
    obs: u64,
    decisions: u64,
}

impl<W: Write> ActLogWriter<W> {
    pub fn new(w: W) -> Self {
        Self { w, obs: 0, decisions: 0 }
    }

    pub fn push(&mut self, line: &ObsLine) -> Result<()> {
        ensure!(line.i as u64 == self.obs, "act log: observation index {} (expected {})", line.i, self.obs);
        line.check_state()?;
        serde_json::to_writer(&mut self.w, line)?;
        self.w.write_all(b"\n")?;
        self.obs += 1;
        self.decisions += line.decision.is_some() as u64;
        Ok(())
    }

    pub fn observations(&self) -> u64 {
        self.obs
    }

    pub fn finish(mut self, mut end: ActLogEnd) -> Result<W> {
        end.obs = self.obs;
        end.decisions = self.decisions;
        serde_json::to_writer(&mut self.w, &EndWrap { end: &end })?;
        self.w.write_all(b"\n")?;
        self.w.flush()?;
        Ok(self.w)
    }
}

pub fn act_log_name(plan_game: u32, players_index: u8) -> String {
    format!("g{plan_game}-p{players_index}.jsonl")
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InvalidReason {
    Digest,
    Frame,
    State,
    MissingInput,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Invalid {
    pub reason: InvalidReason,
    pub obs: Option<u32>,
    pub frame: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrefixStep<'a> {
    Forced(&'a ObsLine),
    AtT(&'a ObsLine),
    Live,
}

#[derive(Debug, Clone)]
pub struct ForcedPrefix {
    log: ActLog,
    i_t: u32,
    matched: u32,
}

impl ForcedPrefix {
    pub fn new(log: ActLog, i_t: u32, f_t: u32) -> Result<Self> {
        let at = log.line(i_t).with_context(|| format!("trunk act log has no observation {i_t}"))?;
        ensure!(at.frame == f_t, "trunk observation {i_t} is at frame {}, not f_t {f_t}", at.frame);
        ensure!(log.lines[..=i_t as usize].iter().all(|l| l.decision.is_some()), "trunk act log is not a model session's (no decision before t)");
        Ok(Self { log, i_t, matched: 0 })
    }

    pub fn i_t(&self) -> u32 {
        self.i_t
    }

    pub fn matched(&self) -> u32 {
        self.matched
    }

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

pub fn quit_act() -> Act {
    Act { cmds: vec![], advance_frames: 1, valid_until_frame: None, surrender: Some(String::new()) }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrefixCompare {
    pub obs: u32,
    pub matched: u32,
    pub first_mismatch_frame: Option<u32>,
}

pub fn compare_prefix(trunk: &[ObsLine], branch: &[ObsLine], before_frame: u32) -> PrefixCompare {
    let same = |a: &ObsLine, b: &ObsLine| a.frame == b.frame && a.digest == b.digest;
    let prefix: Vec<&ObsLine> = trunk.iter().take_while(|l| l.frame < before_frame).collect();
    let matched = prefix.iter().zip(branch).filter(|(a, b)| same(a, b)).count() as u32;
    let first = trunk.iter().zip(branch).position(|(a, b)| !same(a, b)).or((trunk.len() != branch.len()).then(|| trunk.len().min(branch.len())));
    let first_mismatch_frame = first.map(|i| branch.get(i).or(trunk.get(i)).map_or(0, |l| l.frame));
    PrefixCompare { obs: prefix.len() as u32, matched, first_mismatch_frame }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRule {
    pub name: String,
    pub from_frame: u32,
    pub to_frame: u32,
    #[serde(default)]
    pub any_open: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_rule: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_s: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Snapshots {
    rules: Vec<(SnapshotRule, Vec<u16>, Option<(usize, u32)>)>,
    fired: Vec<Option<(u32, u32)>>,
    not_before: Vec<u32>,
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
            let after = match (&r.after_rule, r.after_s) {
                (None, None) => None,
                (Some(a), Some(s)) => {
                    let i = out.iter().position(|(x, _, _): &(SnapshotRule, Vec<u16>, _)| &x.name == a).with_context(|| format!("snapshot rule `{}`: after_rule `{a}` is not an earlier rule (051-R3)", r.name))?;
                    let delay = s.checked_mul(FRAMES_PER_SECOND).with_context(|| format!("snapshot rule `{}`: after_s {s} overflows the frame count (051-R3)", r.name))?;
                    Some((i, delay))
                }
                _ => anyhow::bail!("snapshot rule `{}`: after_rule and after_s go together (051-R3)", r.name),
            };
            out.push((r.clone(), ids, after));
        }
        Ok(Self { fired: vec![None; out.len()], not_before: vec![0; out.len()], rules: out })
    }

    pub fn check(&mut self, obs: u32, frame: u32, type_mask: &[bool]) -> Vec<String> {
        let mut hit = vec![];
        for k in 0..self.rules.len() {
            let (r, ids, after) = &self.rules[k];
            if self.fired[k].is_some() || frame < r.from_frame.max(self.not_before[k]) || frame >= r.to_frame {
                continue;
            }
            if !(ids.is_empty() || ids.iter().any(|&a| type_mask.get(a as usize).copied().unwrap_or(false))) {
                continue;
            }
            if let Some((i, delay)) = *after
                && let Some((o, f)) = self.fired[i]
                && o == obs
            {
                self.not_before[k] = f.saturating_add(delay);
                continue;
            }
            self.fired[k] = Some((obs, frame));
            hit.push(r.name.clone());
        }
        hit
    }

    pub fn fired(&self) -> impl Iterator<Item = (&str, Option<u32>)> {
        self.rules.iter().zip(&self.fired).map(|((r, _, _), f)| (r.name.as_str(), f.map(|(o, _)| o)))
    }
}

pub fn state_id(plan_game: u32, obs: u32) -> String {
    format!("g{plan_game}-o{obs}")
}

const STATE_MAGIC: &[u8; 4] = b"URST";
const STATE_VERSION: u32 = 1;

pub fn encode_state(parts: &[(Vec<usize>, Vec<f32>)]) -> Result<Vec<u8>> {
    let u32_of = |v: usize, what: &str| u32::try_from(v).with_context(|| format!("state {what} {v} exceeds u32"));
    let mut out = Vec::with_capacity(12 + parts.iter().map(|(s, d)| 4 + 4 * s.len() + 4 * d.len()).sum::<usize>());
    out.extend_from_slice(STATE_MAGIC);
    out.extend_from_slice(&STATE_VERSION.to_le_bytes());
    out.extend_from_slice(&u32_of(parts.len(), "part count")?.to_le_bytes());
    for (k, (shape, data)) in parts.iter().enumerate() {
        let n: usize = shape.iter().product();
        ensure!(!shape.is_empty(), "state part {k}: rank 0 (every Core state part has a batch dimension, 032-R16)");
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
        ensure!(rank > 0, "state part of rank 0");
        let shape = (0..rank).map(|_| c.word().map(|d| d as usize)).collect::<Result<Vec<_>>>()?;
        let n = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).context("state part too large")?;
        ensure!(n <= c.0.len() / 4, "state file truncated");
        let data = (0..n).map(|_| c.word().map(f32::from_bits)).collect::<Result<Vec<_>>>()?;
        out.push((shape, data));
    }
    ensure!(c.0.is_empty(), "state file has {} trailing bytes", c.0.len());
    Ok(out)
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Force,
    Ban,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionRule {
    pub op: Op,
    pub action: String,
    pub from_s: u32,
    pub to_s: u32,
}

impl InterventionRule {
    pub fn t_only(&self) -> bool {
        self.from_s == 0 && self.to_s == 0
    }
}

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RuleLog {
    #[serde(flatten)]
    pub rule: InterventionRule,
    pub in_window: u32,
    pub applied: u32,
    pub force_frame: Option<u32>,
    pub missed: bool,
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
            ensure!(!(r.op == Op::Ban && id == 0), "{}: NoOp cannot be banned (051-R7 ③)", ctx());
            let at = |s: u32| s.checked_mul(FRAMES_PER_SECOND).and_then(|d| f_t.checked_add(d)).with_context(|| format!("{}: window frame overflows u32", ctx()));
            let (start, end) = if r.t_only() {
                (f_t, f_t.checked_add(1).with_context(|| format!("{}: f_t {f_t} overflows u32", ctx()))?)
            } else {
                (at(r.from_s)?, at(r.to_s)?)
            };
            let log = RuleLog { rule: r.clone(), in_window: 0, applied: 0, force_frame: None, missed: false, rejected: None };
            rules.push(ActiveRule { id, start, end, log });
        }
        Ok(Self { rules })
    }

    pub fn apply(&mut self, frame: u32, type_mask: &[bool]) -> Vec<bool> {
        let mut m = type_mask.to_vec();
        for r in self.rules.iter_mut().filter(|r| r.log.rule.op == Op::Ban) {
            if !r.in_window(frame) {
                continue;
            }
            r.log.in_window += 1;
            let i = r.id as usize;
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
            } else if r.log.rule.t_only() {
                r.log.missed = true;
            }
        }
        debug_assert!(m.iter().any(|&b| b) || !type_mask.iter().any(|&b| b), "051-R7: intervention emptied the mask");
        m
    }

    pub fn apply_to(&mut self, frame: u32, obs: &mut Observation) {
        obs.masks.type_mask = self.apply(frame, &obs.masks.type_mask);
    }

    pub fn note_executor(&mut self, frame: u32, rejected: bool) {
        for r in &mut self.rules {
            if r.log.force_frame == Some(frame) {
                r.log.rejected = Some(rejected);
            }
        }
    }

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

pub fn open_ids(type_mask: &[bool]) -> Vec<u16> {
    type_mask.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u16).collect()
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Sample {
    pub frame: u32,
    pub scalar: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Samples {
    pub t60: Option<Sample>,
    pub t120: Option<Sample>,
    pub t240: Option<Sample>,
    pub horizon: Option<Sample>,
    pub d8: Option<Sample>,
    pub d32: Option<Sample>,
}

#[derive(Debug, Clone)]
pub struct SampleTracker {
    f_t: u32,
    horizon: u32,
    samples: Samples,
    started: bool,
    commands: u32,
}

impl SampleTracker {
    pub fn new(f_t: u32, horizon: u32) -> Self {
        Self { f_t, horizon, samples: Samples::default(), started: false, commands: 0 }
    }

    pub fn on_decision(&mut self, frame: u32, scalar: &[f32], sent_command: bool) {
        let s = || Some(Sample { frame, scalar: scalar.to_vec() });
        let f_t = self.f_t;
        let x = &mut self.samples;
        for (slot, secs) in [(&mut x.t60, 60), (&mut x.t120, 120), (&mut x.t240, 240)] {
            if slot.is_none() && frame >= f_t.saturating_add(secs * FRAMES_PER_SECOND) {
                *slot = s();
            }
        }
        if x.horizon.is_none() && frame >= self.horizon {
            x.horizon = s();
        }
        let after_t = std::mem::replace(&mut self.started, true);
        if after_t && sent_command {
            self.commands += 1;
            match self.commands {
                8 => x.d8 = s(),
                32 => x.d32 = s(),
                _ => {}
            }
        }
    }

    pub fn samples(&self) -> &Samples {
        &self.samples
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DilutionEntry {
    pub action: String,
    pub first_frame: Option<u32>,
    pub picks_30s: u32,
}

#[derive(Debug, Clone)]
pub struct DilutionTracker {
    f_t: u32,
    entries: Vec<(u16, DilutionEntry)>,
}

impl DilutionTracker {
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

    pub fn on_decision(&mut self, frame: u32, action: u16) {
        if frame <= self.f_t {
            return;
        }
        for (_, e) in self.entries.iter_mut().filter(|(id, _)| *id == action) {
            e.first_frame.get_or_insert(frame);
            if frame < self.f_t.saturating_add(30 * FRAMES_PER_SECOND) {
                e.picks_30s += 1;
            }
        }
    }

    pub fn entries(&self) -> Vec<DilutionEntry> {
        self.entries.iter().map(|(_, e)| e.clone()).collect()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GameResult {
    Win,
    Loss,
    Draw,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BranchEnd {
    pub result: GameResult,
    pub end_frame: u32,
    pub reason: EndReason,
    pub before_horizon: bool,
}

impl BranchEnd {
    pub fn new(winner: Option<u8>, me: u8, end_frame: u32, reason: EndReason, horizon: u32) -> Self {
        let result = match winner {
            None => GameResult::Draw,
            Some(w) if w == me => GameResult::Win,
            Some(_) => GameResult::Loss,
        };
        Self { result, end_frame, reason, before_horizon: end_frame < horizon }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StateCheck {
    pub bit_equal: bool,
    pub max_abs_diff: Option<f64>,
}

impl StateCheck {
    pub fn new(bit_equal: bool, max_abs_diff: f64) -> Self {
        Self { bit_equal, max_abs_diff: max_abs_diff.is_finite().then_some(max_abs_diff) }
    }
}

pub const OUTCOME_METRICS: &[&str] = &["held_8m", "held_s", "scv_8m", "scv_min_1m", "bunker_s", "barracks_s", "army_s"];

pub trait FiniteRecord {
    fn check_finite(&self) -> Result<()>;
}

fn finite(what: &str, v: &[f32]) -> Result<()> {
    if let Some(k) = v.iter().position(|x| !x.is_finite()) {
        bail!("{what}[{k}] = {} is not finite", v[k]);
    }
    Ok(())
}

impl FiniteRecord for Samples {
    fn check_finite(&self) -> Result<()> {
        for (name, s) in [("t60", &self.t60), ("t120", &self.t120), ("t240", &self.t240), ("horizon", &self.horizon), ("d8", &self.d8), ("d32", &self.d32)] {
            if let Some(s) = s {
                finite(&format!("samples.{name}.scalar"), &s.scalar)?;
            }
        }
        Ok(())
    }
}

fn finite_state_check(c: &Option<StateCheck>) -> Result<()> {
    if let Some(d) = c.as_ref().and_then(|c| c.max_abs_diff) {
        ensure!(d.is_finite(), "state_check.max_abs_diff = {d} is not finite");
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BranchOutcome {
    pub valid: bool,
    pub invalid: Option<Invalid>,
    pub prefix_obs: u32,
    pub prefix_matched: u32,
    pub state_check: Option<StateCheck>,
    pub intervention: Vec<RuleLog>,
    pub samples: Samples,
    pub metrics: serde_json::Map<String, serde_json::Value>,
    pub dilution: Vec<DilutionEntry>,
    pub end: Option<BranchEnd>,
}

impl FiniteRecord for BranchOutcome {
    fn check_finite(&self) -> Result<()> {
        finite_state_check(&self.state_check)?;
        self.samples.check_finite()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BranchPlanLine {
    pub trunk: String,
    pub point: String,
    pub acts: String,
    pub opponent_acts: String,
    pub state: String,
    pub state_sha256: String,
    pub i_t: u32,
    pub f_t: u32,
    pub seed: u32,
    pub map: String,
    pub opponent: String,
    pub alt: Alternative,
    pub repeat: u32,
}

impl FiniteRecord for BranchPlanLine {
    fn check_finite(&self) -> Result<()> {
        Ok(())
    }
}

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

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schemas {
    pub feature: u32,
    pub action: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PointRecord {
    pub point: String,
    pub trunk: TrunkRef,
    pub seed: u32,
    pub map: String,
    pub opponent: String,
    pub condition: Option<String>,
    pub checkpoint_sha256: String,
    pub policy: String,
    pub t0_policy: String,
    pub git: String,
    pub schemas: Schemas,
    pub backend: String,
    pub precision: String,
    pub infer_seed: Option<u64>,
    pub i_t: u32,
    pub f_t: u32,
    #[serde(with = "hex_u64")]
    pub digest_t: u64,
    pub type_mask: Vec<u16>,
    pub scalar_t: Vec<f32>,
    pub trunk_action: Decision,
    pub state: String,
    pub state_sha256: String,
    pub trunk_outcome: serde_json::Map<String, serde_json::Value>,
    pub valid: bool,
    pub skip_reason: Option<String>,
}

impl FiniteRecord for PointRecord {
    fn check_finite(&self) -> Result<()> {
        finite("scalar_t", &self.scalar_t)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct GameRef {
    pub job: String,
    pub stage: String,
    pub plan_game: u32,
    pub hostd_game: u64,
    pub attempt: u32,
    pub replay: Option<String>,
    pub host: String,
    pub backend: String,
    pub precision: String,
    pub infer_seed: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrefixRecord {
    pub obs: u32,
    pub matched: u32,
    pub opponent: Option<PrefixCompare>,
}

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

impl FiniteRecord for BranchRecord {
    fn check_finite(&self) -> Result<()> {
        finite_state_check(&self.state_check)?;
        self.samples.check_finite()
    }
}

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

pub fn write_jsonl<T: Serialize + FiniteRecord>(mut w: impl Write, rows: &[T]) -> Result<()> {
    for (n, r) in rows.iter().enumerate() {
        r.check_finite().with_context(|| format!("row {}", n + 1))?;
    }
    for r in rows {
        serde_json::to_writer(&mut w, r)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    Ok(())
}

pub const HORIZON_MARGIN: u32 = 48;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BranchPlayer {
    pub line: BranchPlanLine,
    pub acts: std::path::PathBuf,
    pub state: std::path::PathBuf,
    pub horizon: u32,
    pub none_actions: Vec<String>,
    pub verify_state: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct GameBranch {
    pub plan_game: u32,
    pub acts_dir: Option<std::path::PathBuf>,
    pub snapshots: Vec<SnapshotRule>,
    pub states_dir: Option<std::path::PathBuf>,
    pub branch: Option<BranchPlayer>,
}

impl GameBranch {
    pub fn act_log_path(&self, players_index: u8) -> Option<std::path::PathBuf> {
        self.acts_dir.as_ref().map(|d| d.join(act_log_name(self.plan_game, players_index)))
    }

    pub fn state_path(&self, obs: u32) -> Option<std::path::PathBuf> {
        self.states_dir.as_ref().map(|d| d.join(format!("{}.state", state_id(self.plan_game, obs))))
    }
}

pub type BranchBoard = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<GameBranch>>>>;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SnapshotTaken {
    pub rules: Vec<String>,
    pub state: String,
    pub state_sha256: String,
    pub i: u32,
    pub frame: u32,
    #[serde(with = "hex_u64")]
    pub digest: u64,
    pub type_mask: Vec<u16>,
    pub scalar: Vec<f32>,
}

impl FiniteRecord for SnapshotTaken {
    fn check_finite(&self) -> Result<()> {
        finite("scalar", &self.scalar)
    }
}

#[cfg(test)]
mod tests;
