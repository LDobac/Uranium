use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use uranium_agent::infer::InferenceHandle;
use uranium_agent::model_policy::{LearnerHook, SessionObserver};
use uranium_data::data::ZEntry;
use uranium_env::features::Strategy;
use uranium_env::scout::RewardConfig;
use uranium_proto::{GameInfo, Race};
use crate::league::{AgentKind, Class};
use crate::trajectory::{GameResult, Segment, TrajectoryObserver};
use super::Shared;
use super::seed::splitmix;
use super::csv::slots_start_hit;

pub(super) struct Hook {
    pub(super) agent: AgentKind,
    pub(super) handle: InferenceHandle,
    pub(super) seq_len: usize,
    pub(super) shared: Arc<Shared>,
    pub(super) tx: mpsc::Sender<GameResult>,
    pub(super) z: Vec<ZEntry>,
    pub(super) z_drop: f64,
    pub(super) rng: Mutex<u64>,
    pub(super) reward: RewardConfig,
}

impl LearnerHook for Hook {
    fn handle(&self) -> &InferenceHandle {
        &self.handle
    }

    fn session(&self, game_id: u64, info: &GameInfo) -> (Option<Strategy>, Box<dyn SessionObserver>) {
        let mut class = None;
        let mut ladder_group = None;
        let mut key = String::new();
        let mut start_hit = None;
        if let Some(m) = self.shared.games.lock().unwrap().get_mut(&game_id) {
            m.self_player = Some(info.self_player);
            class = Some(m.pick.class);
            ladder_group = m.pick.ladder_group.clone();
            key = m.pick.key.clone();
            start_hit = slots_start_hit(m.slots);
            if let Some(e) = info.players.iter().find(|p| p.is_enemy) {
                if m.enemy_race == Race::Random || m.enemy_race == Race::Unknown {
                    m.enemy_race = e.race;
                }
            }
        }
        let lobby_race = info.players.iter().find(|p| p.is_enemy).map_or(Race::Unknown, |p| p.race);
        let r_value_policy = class.is_none_or(|c| self.shared.league.lock().unwrap().r_value_in_policy(c));
        let pseudo_scale = self.reward.pseudo_scale(class == Some(Class::Model));
        let z = {
            let mut r = self.rng.lock().unwrap();
            let u = splitmix(&mut r);
            if self.z.is_empty() || u < self.z_drop {
                None
            } else {
                let e = &self.z[(splitmix(&mut r) * self.z.len() as f64) as usize % self.z.len()];
                Some(Strategy { build: e.build.clone(), units: e.units.clone() })
            }
        };
        let mut obs = TrajectoryObserver::new(self.agent.index(), game_id, self.seq_len, self.shared.queues[self.agent.index()].clone(), self.tx.clone()).hold_terminal(held_sink(self.shared.clone())).r_value_policy(r_value_policy).pseudo(self.reward.pseudo, pseudo_scale).enemy_lobby_race(lobby_race);
        if let Some(sil) = &self.shared.sil
            && self.agent == AgentKind::Main
            && ladder_group.as_deref() == Some(sil.cfg.select_group.as_str())
            && sil.cfg.may_select(&key, start_hit)
        {
            let s = self.shared.clone();
            obs = obs.copy_segments(Box::new(move |seg: &Segment| {
                let games = s.games.lock().unwrap();
                if games.contains_key(&seg.game_id)
                    && let Some(sil) = &s.sil
                {
                    sil.keep(seg);
                }
                drop(games);
            }));
        }
        (z, Box::new(obs))
    }

    fn reward_config(&self) -> RewardConfig {
        self.reward.clone()
    }
}

fn held_sink(s: Arc<Shared>) -> Box<dyn Fn(Segment) + Send> {
    Box::new(move |seg: Segment| {
        s.held.lock().unwrap().insert(seg.game_id, seg);
    })
}
