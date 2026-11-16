use anyhow::Result;
use uranium_env::features::augment::{Sym, transform_target_loc, transform_with_starts};
use uranium_env::features::extract::dims;
use uranium_env::features::*;
use uranium_env::features::label::{LabelWalker, UNKNOWN_REASON};
use uranium_replay::shard::ShardGame;
use uranium_env::maps::family_index;
use super::{TrainStep, GameSeq, value_weight, replay_z};
use super::maps::MapStore;

pub fn enemy_race_masked(replay_sha256: &str, seed: u64, ratio: f64) -> bool {
    if ratio <= 0.0 {
        return false;
    }
    game_unit(replay_sha256, seed, RACE_MASK_SALT) < ratio
}

const RACE_MASK_SALT: u64 = 0x5241_4345_4D41_534B;
pub(crate) const SUBSAMPLE_SALT: u64 = 0x5355_4253_414D_504C;

pub(crate) fn game_unit(replay_sha256: &str, seed: u64, salt: u64) -> f64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in replay_sha256.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut z = h ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

pub struct GameStream {
    game: ShardGame,
    walker: LabelWalker,
    z: Option<Strategy>,
    sym: Sym,
    pos: usize,
    zero_priv: PrivilegedTensors,
    pub result: f32,
    pub value_weight: f32,
    chain_start: bool,
    chain_from: usize,
    pending_events: Vec<uranium_proto::GameEvent>,
}

pub fn chained_step(steps: &[uranium_replay::shard::Step], i: usize) -> bool {
    i > 0 && i < steps.len() && steps[i].raw_frame.frame <= steps[i - 1].raw_frame.frame.saturating_add(1) && steps[i - 1].action_label.action != NOOP_ACTION
}

const NOOP_ACTION: u16 = 0;

pub const CHAIN_START_SOURCES: [&str; 2] = ["botgen", "dagger"];

pub fn check_chain_start_sources(sources: &[String]) -> Result<()> {
    anyhow::ensure!(!sources.is_empty(), "[data] chain_start_sources is empty: remove the key (041-R30)");
    for s in sources {
        anyhow::ensure!(CHAIN_START_SOURCES.contains(&s.as_str()), "[data] chain_start_sources: {s:?} is not one of {CHAIN_START_SOURCES:?} (041-R30)");
    }
    Ok(())
}

impl GameStream {
    pub fn new(game: ShardGame, z_present: bool, sym: Sym, maps: &MapStore) -> Result<Self> {
        let walker = LabelWalker::new(&game.game_info, family_index(&game.meta.map_family), maps.get(&game.meta.map_hash)?.as_deref(), game.meta.enemy_random)?;
        let z = z_present.then(|| replay_z(&game));
        let zero_priv = PrivilegedTensors { summary: vec![0.0; dims().privileged()] };
        let result = game.meta.result as f32;
        let value_weight = value_weight(&game.meta);
        Ok(Self { game, walker, z, sym, pos: 0, zero_priv, result, value_weight, chain_start: false, chain_from: 0, pending_events: Vec::new() })
    }

    pub fn with_chain_start(mut self, sources: &[String]) -> Self {
        debug_assert_eq!(self.pos, 0, "chain-start rule after the first step");
        self.chain_start = sources.contains(&self.game.meta.src);
        self
    }

    pub fn with_race_mask(mut self, ratio: f64, seed: u64) -> Self {
        debug_assert_eq!(self.pos, 0, "race mask after the first step");
        if enemy_race_masked(&self.game.meta.replay_sha256, seed, ratio) {
            self.walker.mem.enemy_race = uranium_proto::Race::Unknown;
        }
        self
    }

    pub fn enemy_race(&self) -> &'static str {
        use uranium_proto::Race;
        match self.game.game_info.players.iter().find(|p| p.is_enemy).map(|p| p.race) {
            Some(Race::Terran) => "terran",
            Some(Race::Protoss) => "protoss",
            Some(Race::Zerg) => "zerg",
            Some(Race::Random) => "random",
            _ => "unknown",
        }
    }

    pub fn replay_sha256(&self) -> &str {
        &self.game.meta.replay_sha256
    }

    pub fn perspective_slot(&self) -> u8 {
        self.game.meta.perspective_slot
    }

    pub fn remaining(&self) -> usize {
        self.game.steps.len() - self.pos
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn sym(&self) -> Sym {
        self.sym
    }

    pub fn z_present(&self) -> bool {
        self.z.is_some()
    }

    pub fn advance_to(&mut self, pos: usize) {
        while self.pos < pos.min(self.game.steps.len()) {
            let n = (pos - self.pos).min(64);
            self.next_chunk(n);
        }
    }

    pub fn next_chunk(&mut self, n: usize) -> (Vec<TrainStep>, usize) {
        let end = self.pos.saturating_add(n).min(self.game.steps.len());
        let mut steps = Vec::with_capacity(end - self.pos);
        let mut invalid = 0;
        for i in self.pos..end {
            let s = &self.game.steps[i];
            let chained = self.chain_start && chained_step(&self.game.steps, i);
            let owned: Option<uranium_proto::RawFrame>;
            let obs_step = if chained {
                self.pending_events.extend(s.raw_frame.events.iter().cloned());
                let c = &self.game.steps[self.chain_from];
                owned = Some(uranium_proto::RawFrame { events: Vec::new(), ..c.raw_frame.clone() });
                c
            } else {
                self.chain_from = i;
                owned = (!self.pending_events.is_empty()).then(|| {
                    let mut events = std::mem::take(&mut self.pending_events);
                    events.extend(s.raw_frame.events.iter().cloned());
                    uranium_proto::RawFrame { events, ..s.raw_frame.clone() }
                });
                s
            };
            let frame = owned.as_ref().unwrap_or(&s.raw_frame);
            let (o, mut label, why) = self.walker.step_with_memory(frame, &s.action_label, self.z.as_ref(), s.memory_action.as_deref());
            let surrender_rule = uranium_env::surrender::rule_holds(frame, &self.walker.mem);
            let ok = why.is_none();
            invalid += (!ok && why != Some(UNKNOWN_REASON)) as usize;
            let o = if self.sym == Sym::IDENTITY {
                o
            } else {
                if let Some(t) = label.target_loc {
                    label.target_loc = Some(transform_target_loc(label.action, t, o.map_size, o.transposed, self.sym));
                }
                transform_with_starts(&o, self.sym, &self.walker.map.start_locations)
            };
            let privileged = obs_step.privileged.as_ref().map_or_else(|| self.zero_priv.clone(), privileged);
            let has_privileged = obs_step.privileged.is_some();
            steps.push(TrainStep { obs: o, label, privileged, has_privileged, weight: if ok { 1.0 } else { 0.0 }, frame: frame.frame, surrender_rule });
        }
        self.pos = end;
        (steps, invalid)
    }
}

pub fn prepare(game: &ShardGame, z_present: bool, sym: Sym, maps: &MapStore) -> Result<GameSeq> {
    let mut stream = GameStream::new(game.clone(), z_present, sym, maps)?;
    let (steps, invalid_labels) = stream.next_chunk(usize::MAX);
    Ok(GameSeq { steps, result: stream.result, invalid_labels })
}
