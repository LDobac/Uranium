//! The server side of [`InferenceHandle`] for a model that runs outside Rust (the PyTorch
//! inference thread, AMB-000-4, 001-R6). Sessions talk to it exactly as to any inference server;
//! the model thread pulls work with [`Broker::next`] and answers with [`Broker::reply`]:
//! - `resets`: slots opened since the last work; their Core state starts at zero (032-R5)
//! - `states`: slots whose Core state a session asked for (032-R16), answered with
//!   [`Broker::reply_states`] before the batch
//! - `batch`: the waiting sessions' observations, collated on the host (030 §1 layout), run once
//!   when the batch is full, every active session waits, or the oldest has waited `max_wait`
//!   (032-R14; lockstep sessions wait for their reply, so batching changes throughput only)
//!
//! The model thread owns the `[capacity, …]` slot state and the weights (032-R16 swaps happen
//! there); this side builds the executor's actions (034 columns, END = padded n).

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use uranium_data::collate::{HostObsBatch, obs_batch};
use uranium_data::table::ActionColumns;
use uranium_env::features::Observation;

use crate::infer::{ActReply, HEADS, HostState, InferOptions, InferenceHandle, Req, SampleMode, host_action};

/// Batches slower than this are logged (normal p95 is well under a second).
const SLOW_BATCH: Duration = Duration::from_secs(5);

struct Pending {
    slot: usize,
    obs: Box<Observation>,
    reply: Sender<Result<ActReply>>,
    since: Instant,
}

/// One pull of the model thread.
pub struct Work {
    pub resets: Vec<usize>,
    pub states: Vec<usize>,
    pub batch: Option<WorkBatch>,
}

pub struct WorkBatch {
    /// Slot of every row.
    pub slots: Vec<usize>,
    pub obs: HostObsBatch,
    /// Time the requests waited before the batch was formed (mean, max), and the batch's
    /// forming time (UNIX ns): latency accounting (AMB-042-10).
    pub wait_mean: Duration,
    pub wait_max: Duration,
    pub formed_unix_ns: u64,
}

/// Wall clock in UNIX nanoseconds (comparable with Python's `time.time_ns`).
pub fn unix_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

/// The model's output for a batch, rows in `WorkBatch::slots` order.
pub struct BatchOut<'a> {
    pub action: &'a [i64],
    pub delay: &'a [i64],
    pub queued: &'a [i64],
    /// [rows, s] selection steps, END = the batch's padded n.
    pub selected: &'a [i64],
    pub target_unit: &'a [i64],
    pub target_loc: &'a [i64],
    pub log_prob: &'a [f32],
    /// [rows, HEADS]
    pub head_log_prob: &'a [f32],
    /// [rows, 4]
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
    inflight: Option<InFlight>,
    version: u64,
    /// Every handle is gone: no more work will come.
    closed: bool,
}

impl Broker {
    /// A broker and the handle sessions use (`checkpoint_id`: 020-R5 policy version).
    pub fn new(checkpoint_id: String, sample: SampleMode, opts: InferOptions) -> Result<(Self, InferenceHandle)> {
        let (tx, rx) = mpsc::channel();
        let cols = ActionColumns::current()?;
        let b = Self { rx, opts, cols, used: vec![false; opts.capacity], pending: Vec::new(), resets: Vec::new(), states: Vec::new(), asked: Vec::new(), inflight: None, version: 0, closed: false };
        Ok((b, InferenceHandle::new(tx, checkpoint_id, sample, &opts)))
    }

    pub fn options(&self) -> &InferOptions {
        &self.opts
    }

    /// Every handle (and session) is gone; the model thread can stop.
    pub fn closed(&self) -> bool {
        self.closed && self.pending.is_empty() && self.inflight.is_none()
    }

    /// 032-R16: weights version reported with every later decision (the model thread swapped).
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
            }
            Req::Act { slot, obs, reply } => self.pending.push(Pending { slot, obs, reply, since: Instant::now() }),
            Req::Swap { reply, .. } => {
                let _ = reply.send(Err(anyhow!("this server swaps weights in its model thread (032-R16), not through the handle")));
            }
            Req::State { slot, reply } => self.states.push((slot, reply)),
        }
    }

    fn due(&self) -> bool {
        let active = self.used.iter().filter(|u| **u).count();
        !self.pending.is_empty() && (self.pending.len() >= self.opts.max_batch || self.pending.len() >= active || self.pending[0].since.elapsed() >= self.opts.max_wait)
    }

    /// The next work, waiting at most `timeout`; `None` when there is none yet (or every handle is
    /// gone). The previous batch must have been answered.
    pub fn next(&mut self, timeout: Duration) -> Result<Option<Work>> {
        ensure!(self.inflight.is_none(), "the previous batch has no reply yet");
        ensure!(self.asked.is_empty(), "the previous state requests have no reply yet");
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
            if !self.resets.is_empty() || !self.states.is_empty() || self.due() {
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
        (!resets.is_empty() || !states.is_empty() || batch.is_some()).then_some(Work { resets, states, batch })
    }

    /// Answers the state requests of the last work, in its `states` order.
    pub fn reply_states(&mut self, states: Vec<HostState>) -> Result<()> {
        let asked = std::mem::take(&mut self.asked);
        ensure!(states.len() == asked.len(), "{} states for {} requests", states.len(), asked.len());
        for (s, reply) in states.into_iter().zip(asked) {
            let _ = reply.send(Ok(s));
        }
        Ok(())
    }

    /// Answers the last batch.
    pub fn reply(&mut self, out: BatchOut<'_>) -> Result<()> {
        let f = self.inflight.take().ok_or_else(|| anyhow!("no batch to answer"))?;
        let b = f.batch.len();
        ensure!(out.action.len() == b && out.log_prob.len() == b && out.head_log_prob.len() == b * HEADS && out.value.len() == b * 4, "batch output of {} rows has the wrong shape", b);
        let s = out.selected.len() / b.max(1);
        let run = f.started.elapsed();
        if run >= SLOW_BATCH {
            // 011-AC7: OpenBW drops a LAN client after 60 s without sync, so a stall here ends the
            // game early with both sides claiming the win. Log it with wall time to line it up.
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

    /// The model failed on the last batch (or state requests): every waiting session gets the error.
    pub fn fail(&mut self, msg: &str) {
        if let Some(f) = self.inflight.take() {
            for p in f.batch {
                let _ = p.reply.send(Err(anyhow!("inference failed: {msg}")));
            }
        }
        for reply in std::mem::take(&mut self.asked) {
            let _ = reply.send(Err(anyhow!("inference failed: {msg}")));
        }
    }

    /// Refuses everything still queued (the model thread is stopping).
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
        let open = self.used.iter().filter(|u| **u).count();
        if open > 0 {
            elog!("[infer] server stopped with {open} session(s) open: {msg}");
        }
        Ok(())
    }
}
