use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use uranium_env::host::collate::{HostObsBatch, obs_batch};
use uranium_env::host::table::ActionColumns;
use uranium_env::features::Observation;
use uranium_store::elog;

use crate::infer::{ActReply, HEADS, HostState, InferOptions, InferenceHandle, Req, SampleMode, host_action};

const SLOW_BATCH: Duration = Duration::from_secs(5);

struct Pending {
    slot: usize,
    obs: Box<Observation>,
    reply: Sender<Result<ActReply>>,
    since: Instant,
}

pub struct Work {
    pub resets: Vec<usize>,
    pub restores: Vec<(usize, HostState)>,
    pub states: Vec<usize>,
    pub batch: Option<WorkBatch>,
}

pub struct WorkBatch {
    pub slots: Vec<usize>,
    pub obs: HostObsBatch,
    pub wait_mean: Duration,
    pub wait_max: Duration,
    pub formed_unix_ns: u64,
}

pub fn unix_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

fn restore_shape_error(state: &HostState) -> Option<String> {
    if state.parts.is_empty() {
        return Some("the state has no parts (051-R4)".into());
    }
    state.parts.iter().enumerate().find_map(|(i, (shape, v))| {
        let n = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        (shape.first() != Some(&1) || n != Some(v.len())).then(|| format!("part {i}: {} values for shape {shape:?}, batch 1 expected (051-R4)", v.len()))
    })
}

pub struct BatchOut<'a> {
    pub action: &'a [i64],
    pub delay: &'a [i64],
    pub queued: &'a [i64],
    pub selected: &'a [i64],
    pub target_unit: &'a [i64],
    pub target_loc: &'a [i64],
    pub log_prob: &'a [f32],
    pub head_log_prob: &'a [f32],
    pub value: &'a [f32],
    pub win_open: Option<&'a [f32]>,
}

struct InFlight {
    batch: Vec<Pending>,
    n: usize,
    started: Instant,
}

pub struct Broker {
    rx: Receiver<Req>,
    opts: InferOptions,
    cols: ActionColumns,
    used: Vec<bool>,
    pending: Vec<Pending>,
    resets: Vec<usize>,
    states: Vec<(usize, Sender<Result<HostState>>)>,
    asked: Vec<Sender<Result<HostState>>>,
    restores: Vec<(usize, HostState, Sender<Result<()>>)>,
    restoring: Vec<Sender<Result<()>>>,
    inflight: Option<InFlight>,
    version: u64,
    closed: bool,
}

impl Broker {
    pub fn new(checkpoint_id: String, sample: SampleMode, opts: InferOptions) -> Result<(Self, InferenceHandle)> {
        let (tx, rx) = mpsc::channel();
        let cols = ActionColumns::current()?;
        let b = Self { rx, opts, cols, used: vec![false; opts.capacity], pending: Vec::new(), resets: Vec::new(), states: Vec::new(), asked: Vec::new(), restores: Vec::new(), restoring: Vec::new(), inflight: None, version: 0, closed: false };
        Ok((b, InferenceHandle::new(tx, checkpoint_id, sample, &opts)))
    }

    pub fn options(&self) -> &InferOptions {
        &self.opts
    }

    pub fn closed(&self) -> bool {
        self.closed && self.pending.is_empty() && self.inflight.is_none()
    }

    pub fn set_version(&mut self, v: u64) {
        self.version = v;
    }

    fn handle(&mut self, r: Req) {
        match r {
            Req::Open(reply) => {
                let res = match self.used.iter().position(|u| !u) {
                    Some(slot) => {
                        self.used[slot] = true;
                        self.resets.push(slot);
                        Ok(slot)
                    }
                    None => Err(anyhow!("all {} inference slots in use", self.opts.capacity)),
                };
                let _ = reply.send(res);
            }
            Req::Close(slot) => {
                self.used[slot] = false;
                self.pending.retain(|p| p.slot != slot);
                self.resets.retain(|s| *s != slot);
                self.restores.retain(|(s, _, _)| *s != slot);
            }
            Req::Act { slot, obs, reply } => self.pending.push(Pending { slot, obs, reply, since: Instant::now() }),
            Req::Swap { reply, .. } => {
                let _ = reply.send(Err(anyhow!("this server swaps weights in its model thread (032-R16), not through the handle")));
            }
            Req::State { slot, reply } => self.states.push((slot, reply)),
            Req::SetState { slot, state, reply } => match restore_shape_error(&state) {
                Some(e) => {
                    let _ = reply.send(Err(anyhow!("state restore refused: {e}")));
                }
                None => self.restores.push((slot, state, reply)),
            },
        }
    }

    fn due(&self) -> bool {
        let active = self.used.iter().filter(|u| **u).count();
        !self.pending.is_empty() && (self.pending.len() >= self.opts.max_batch || self.pending.len() >= active || self.pending[0].since.elapsed() >= self.opts.max_wait)
    }

    pub fn next(&mut self, timeout: Duration) -> Result<Option<Work>> {
        ensure!(self.inflight.is_none(), "the previous batch has no reply yet");
        ensure!(self.asked.is_empty(), "the previous state requests have no reply yet");
        ensure!(self.restoring.is_empty(), "the previous restores have no reply yet");
        let deadline = Instant::now() + timeout;
        loop {
            loop {
                match self.rx.try_recv() {
                    Ok(r) => self.handle(r),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.closed = true;
                        return Ok(self.take());
                    }
                }
            }
            if !self.resets.is_empty() || !self.restores.is_empty() || !self.states.is_empty() || self.due() {
                return Ok(self.take());
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            let mut wait = deadline - now;
            if let Some(p) = self.pending.first() {
                wait = wait.min(self.opts.max_wait.saturating_sub(p.since.elapsed()));
            }
            match self.rx.recv_timeout(wait) {
                Ok(r) => self.handle(r),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    self.closed = true;
                    return Ok(self.take());
                }
            }
        }
    }

    fn take(&mut self) -> Option<Work> {
        let resets = std::mem::take(&mut self.resets);
        let (restores, restoring): (Vec<_>, Vec<_>) = std::mem::take(&mut self.restores).into_iter().map(|(slot, st, reply)| ((slot, st), reply)).unzip();
        self.restoring = restoring;
        let (states, asked): (Vec<usize>, Vec<_>) = std::mem::take(&mut self.states).into_iter().unzip();
        self.asked = asked;
        let batch = self.due().then(|| {
            let k = self.pending.len().min(self.opts.max_batch);
            let batch: Vec<Pending> = self.pending.drain(..k).collect();
            let refs: Vec<&Observation> = batch.iter().map(|p| &*p.obs).collect();
            let obs = obs_batch(&refs, self.opts.pad_entities);
            let slots = batch.iter().map(|p| p.slot).collect();
            let waits: Vec<Duration> = batch.iter().map(|p| p.since.elapsed()).collect();
            let wait_mean = waits.iter().sum::<Duration>() / waits.len().max(1) as u32;
            let wait_max = waits.iter().copied().max().unwrap_or_default();
            self.inflight = Some(InFlight { n: obs.n, batch, started: Instant::now() });
            WorkBatch { slots, obs, wait_mean, wait_max, formed_unix_ns: unix_ns() }
        });
        (!resets.is_empty() || !restores.is_empty() || !states.is_empty() || batch.is_some()).then_some(Work { resets, restores, states, batch })
    }

    pub fn reply_restores(&mut self, errors: Vec<Option<String>>) -> Result<()> {
        let restoring = std::mem::take(&mut self.restoring);
        ensure!(errors.len() == restoring.len(), "{} restore results for {} restores", errors.len(), restoring.len());
        for (e, reply) in errors.into_iter().zip(restoring) {
            let _ = reply.send(match e {
                None => Ok(()),
                Some(msg) => Err(anyhow!("state restore refused: {msg}")),
            });
        }
        Ok(())
    }

    pub fn reply_states(&mut self, states: Vec<HostState>) -> Result<()> {
        let asked = std::mem::take(&mut self.asked);
        ensure!(states.len() == asked.len(), "{} states for {} requests", states.len(), asked.len());
        for (s, reply) in states.into_iter().zip(asked) {
            let _ = reply.send(Ok(s));
        }
        Ok(())
    }

    pub fn reply(&mut self, out: BatchOut<'_>) -> Result<()> {
        let f = self.inflight.take().ok_or_else(|| anyhow!("no batch to answer"))?;
        let b = f.batch.len();
        ensure!(out.action.len() == b && out.log_prob.len() == b && out.head_log_prob.len() == b * HEADS && out.value.len() == b * 4, "batch output of {} rows has the wrong shape", b);
        let s = out.selected.len() / b.max(1);
        let run = f.started.elapsed();
        if run >= SLOW_BATCH {
            let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
            elog!("[infer] slow batch: {b} requests took {:.1}s (unix {unix})", run.as_secs_f64());
        }
        for (r, p) in f.batch.into_iter().enumerate() {
            let action = host_action(&self.cols, out.action[r], out.delay[r], out.queued[r], &out.selected[r * s..(r + 1) * s], out.target_unit[r], out.target_loc[r], f.n);
            let reply = ActReply {
                action,
                log_prob: out.log_prob[r],
                head_log_prob: std::array::from_fn(|h| out.head_log_prob[r * HEADS + h]),
                value: std::array::from_fn(|v| out.value[r * 4 + v]),
                win_open: out.win_open.map(|w| w[r]),
                obs: *p.obs,
                batch: b,
                run,
                version: self.version,
            };
            let _ = p.reply.send(Ok(reply));
        }
        Ok(())
    }

    pub fn fail(&mut self, msg: &str) {
        if let Some(f) = self.inflight.take() {
            for p in f.batch {
                let _ = p.reply.send(Err(anyhow!("inference failed: {msg}")));
            }
        }
        for reply in std::mem::take(&mut self.asked) {
            let _ = reply.send(Err(anyhow!("inference failed: {msg}")));
        }
        for reply in std::mem::take(&mut self.restoring) {
            let _ = reply.send(Err(anyhow!("inference failed: {msg}")));
        }
    }

    pub fn shutdown(&mut self, msg: &str) -> Result<()> {
        self.fail(msg);
        while let Ok(r) = self.rx.try_recv() {
            self.handle(r);
        }
        for p in self.pending.drain(..) {
            let _ = p.reply.send(Err(anyhow!("inference server stopped: {msg}")));
        }
        for (_, reply) in self.states.drain(..) {
            let _ = reply.send(Err(anyhow!("inference server stopped: {msg}")));
        }
        for (_, _, reply) in self.restores.drain(..) {
            let _ = reply.send(Err(anyhow!("inference server stopped: {msg}")));
        }
        let open = self.used.iter().filter(|u| **u).count();
        if open > 0 {
            elog!("[infer] server stopped with {open} session(s) open: {msg}");
        }
        Ok(())
    }
}
