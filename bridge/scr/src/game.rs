//! One game on the game thread (012-R1): per-frame bookkeeping, observations on the realtime
//! schedule (012-R8), Acts within their deadline (012-R7), surrender (012-R11) and the end
//! (012-R10, 012-R15). No I/O here: the caller hands messages to the I/O thread.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use uranium_proto::*;

use crate::commands::{self, Ctx};
use crate::end::EndTracker;
use crate::events::EventTracker;
use crate::observe::{NukeState, ObserveState, derive, observe_units, raw_frame};
use crate::schedule::Schedule;
use crate::snapshot::Snapshot;
use crate::tables::Tables;

/// BWAPI neutral player.
const NEUTRAL: u8 = 11;

#[derive(Default, Debug)]
pub struct Outgoing {
    /// Network commands for `send_command`, in order.
    pub packets: Vec<Vec<u8>>,
    pub to_agent: Vec<BridgeMsg>,
    /// Lines for the bridge log.
    pub log: Vec<String>,
}

pub struct Game {
    t: &'static Tables,
    me: u8,
    observe: ObserveState,
    events: EventTracker,
    nukes: NukeState,
    end: EndTracker,
    pub schedule: Schedule,
    /// Position of every map resource seen alive (011-R20).
    remembered: HashMap<u32, Pos>,
    /// Surrendered or ended: nothing more is sent (012-R11).
    pub done: bool,
    pub observations: u32,
    started: Instant,
    serialize_us: Vec<f64>,
    obs_sent: Option<Instant>,
    agent_wait: Duration,
}

fn p99(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * 0.99) as usize).min(v.len() - 1)]
}

impl Game {
    pub fn new(t: &'static Tables, me: u8, enemy: Option<u8>) -> Self {
        Self {
            t,
            me,
            observe: ObserveState::default(),
            events: EventTracker::default(),
            nukes: NukeState::default(),
            end: EndTracker::new(me, enemy),
            // Unit tags are ready from frame 1, like the OpenBW bridge's first observation (011-R15).
            schedule: Schedule::new(1),
            remembered: HashMap::new(),
            done: false,
            observations: 0,
            started: Instant::now(),
            serialize_us: vec![],
            obs_sent: None,
            agent_wait: Duration::ZERO,
        }
    }

    fn end_msg(&mut self, s: &Snapshot, winner: Option<u8>, reason: EndReason, frame: RawFrame) -> BridgeMsg {
        self.done = true;
        BridgeMsg::End(GameEnd {
            winner,
            frame: s.frame,
            reason,
            stats: Some(BridgeStats {
                game_frames: s.frame,
                game_time_ms: self.started.elapsed().as_secs_f64() * 1e3,
                agent_wait_ms: self.agent_wait.as_secs_f64() * 1e3,
                serialize_us_p99: p99(&mut self.serialize_us),
                observations: self.observations,
            }),
            final_frame: Some(Box::new(frame)),
        })
    }

    /// One game frame. `act`: the agent's reply read since the last frame, if any.
    pub fn frame(&mut self, s: &Snapshot, act: Option<AgentMsg>) -> Outgoing {
        let t = self.t;
        let mut out = Outgoing::default();
        if self.done {
            return out;
        }
        let d = derive(s, t);
        self.events.update(s, &d);
        let (dots, launches) = self.nukes.update(s, t, t.units.ghost);
        for p in launches {
            self.events.push(GameEvent::NukeDetect(p));
        }
        let mut resources = HashMap::new();
        for (i, u) in s.units.iter().enumerate() {
            if u.player == NEUTRAL && (t.is_mineral_field(u.unit_type) || u.unit_type == t.units.vespene_geyser) {
                resources.insert(u.tag, i);
                self.remembered.insert(u.tag, (d[i].pos.0.max(0) as u16, d[i].pos.1.max(0) as u16));
            }
        }
        let snapshot_frame = |game: &mut Self| {
            let units = observe_units(s, &d, t, &game.observe);
            raw_frame(s, t, units, &dots, game.events.take())
        };
        if let Some((winner, reason)) = self.end.update(s, &d) {
            out.log.push(format!("game over at frame {}: winner {winner:?}, {reason:?}", s.frame));
            let last = snapshot_frame(self);
            out.to_agent.push(self.end_msg(s, winner, reason, last));
            return out;
        }
        match act {
            Some(AgentMsg::Act(a)) => {
                if let Some(sent) = self.obs_sent.take() {
                    self.agent_wait += sent.elapsed();
                }
                let ok = self.schedule.on_act(s.frame, a.advance_frames, a.valid_until_frame);
                if let Some(text) = &a.surrender {
                    // 012-R11: chat, stop, report the loss; the game window stays open.
                    if !a.cmds.is_empty() || text.len() > MAX_SURRENDER_CHAT {
                        out.log.push(format!("010-R13: bad surrender Act ({} commands, {} bytes) ignored", a.cmds.len(), text.len()));
                    } else {
                        if !text.is_empty() {
                            out.packets.push(commands::chat(t, self.me, text));
                        }
                        out.log.push(format!("012-R11: surrender at frame {} (chat {text:?})", s.frame));
                        let last = snapshot_frame(self);
                        let winner = self.end.enemy();
                        out.to_agent.push(self.end_msg(s, winner, EndReason::Surrender, last));
                        return out;
                    }
                } else if ok {
                    let observed: HashMap<u32, usize> = s.units.iter().enumerate().filter(|(i, _)| d[*i].observed).map(|(i, u)| (u.tag, i)).collect();
                    let ctx = Ctx { s, d: &d, t, observed: &observed, resources: &resources, remembered: &self.remembered };
                    out.packets.extend(commands::encode(&ctx, &a.cmds));
                } else {
                    out.log.push(format!("012-R7: Act for an observation expired at frame {} (valid until {:?})", s.frame, a.valid_until_frame));
                }
            }
            Some(AgentMsg::Reject(why)) => {
                out.log.push(format!("agent rejected the session: {why}"));
                self.done = true;
                return out;
            }
            None => {}
        }
        if self.schedule.due(s.frame) {
            let t0 = Instant::now();
            let frame = snapshot_frame(self);
            self.serialize_us.push(t0.elapsed().as_secs_f64() * 1e6);
            out.to_agent.push(BridgeMsg::Frame(Box::new(frame)));
            self.schedule.observed(s.frame);
            self.observations += 1;
            self.obs_sent = Some(Instant::now());
        }
        self.observe.end_frame(s, &d);
        out
    }

    /// 012-AC6 summary for the log: observation -> applied delay percentiles and expired Acts.
    pub fn summary(&self) -> String {
        let mut d = self.schedule.delays.clone();
        d.sort_unstable();
        let q = |p: f64| d.get(((d.len().max(1) - 1) as f64 * p) as usize).copied().unwrap_or(0);
        format!(
            "observations {} acts {} expired {} delay frames p50 {} p90 {} p99 {} max {}",
            self.observations,
            d.len(),
            self.schedule.expired,
            q(0.5),
            q(0.9),
            q(0.99),
            d.last().copied().unwrap_or(0)
        )
    }
}
