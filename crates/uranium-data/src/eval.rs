//! The data side of `bc-eval` (041 offline metrics, 041-AC1, AC4): the split's games (all, or N
//! by the fixed-seed shuffle of 032-R17 ②), read without z (041-AC2 condition), symmetry or the
//! enemy race mask, lanes drained to the last game; and the replay of the 031-R12 surrender check
//! over every game with the model's win_open values (041-AC4) plus the win_open MSE (032-R18).
//! The metrics from the head outputs are computed by the PyTorch evaluator (AMB-000-4).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;
use uranium_env::surrender::SurrenderJudge;

use crate::bc::{Batch, BcConfig, Feeder, eval_games};
use crate::data::{ChosenGames, MapStore, ShardSource, StepSource};
use crate::split::{Selection, Split};

/// 041-AC4: the 031-R12 surrender check replayed over the evaluated games (Shadow mode).
#[derive(Serialize, Debug, Default, Clone)]
pub struct SurrenderEval {
    pub games: u64,
    /// Games where the check held at some decision.
    pub held: u64,
    /// Of those, games with a decided result, and the ones the player won (false surrenders).
    pub held_decided: u64,
    pub held_won: u64,
    /// held_won / held_decided.
    pub false_surrender_rate: f64,
    /// First frame it held: min / median / max over the games where it held.
    pub held_frame_min: Option<u32>,
    pub held_frame_median: Option<u32>,
    pub held_frame_max: Option<u32>,
}

/// How the games are picked.
#[derive(Clone, Copy, Debug)]
pub struct EvalSelection {
    pub split: Split,
    /// `Some(n)`: n games by the fixed-seed shuffle (032-R17 ②); `None`: every game.
    pub games: Option<usize>,
    pub seed: u64,
}

enum Source {
    All(ShardSource),
    Chosen(ChosenGames),
}

impl Source {
    fn get(&mut self) -> &mut dyn StepSource {
        match self {
            Source::All(s) => s,
            Source::Chosen(s) => s,
        }
    }
}

pub struct EvalStream {
    feeder: Feeder,
    source: Source,
    has_open: bool,
    /// Per lane, the surrender check of its current game and the game's result (041-AC4).
    judges: Vec<Option<(SurrenderJudge, f32)>>,
    sur: SurrenderEval,
    held_frames: Vec<u32>,
    open_se: f64,
    open_n: u64,
}

impl EvalStream {
    /// `has_open`: the model has a win_open head (032-R18), so the surrender check is replayed.
    pub fn new(dirs: &[PathBuf], cfg: &BcConfig, sel: EvalSelection, maps: Option<Arc<MapStore>>, has_open: bool) -> Result<Self> {
        // B-2: every game's map precompute is checked before the model runs; the scan also gives
        // the split's games for the selection.
        let maps = maps.unwrap_or_else(|| Arc::new(MapStore::from_env()));
        let src = ShardSource::from_dirs(dirs)?.select(Selection::Only(sel.split));
        let scan = crate::data::check_maps(&src, Some(Selection::Only(sel.split)), &maps)?;
        let mut ecfg = cfg.clone();
        ecfg.data.z_drop = 1.0;
        ecfg.data.augment = false;
        // 030-R11: the race mask is a training-only exception; evaluation sees the real input.
        ecfg.data.enemy_race_mask = 0.0;
        let mut feeder = Feeder::new(&ecfg).with_maps(maps);
        feeder.drain = true;
        let source = match sel.games {
            Some(n) => {
                let ids = eval_games(&src, &scan, n, sel.seed);
                Source::Chosen(ChosenGames::new(src, &ids))
            }
            None => Source::All(src),
        };
        Ok(Self { feeder, source, has_open, judges: vec![None; cfg.lanes], sur: SurrenderEval::default(), held_frames: Vec::new(), open_se: 0.0, open_n: 0 })
    }

    /// The next batch; `None` once every lane is drained.
    pub fn next(&mut self) -> Result<Option<Batch>> {
        let Some(batch) = self.feeder.next_batch(self.source.get(), None)? else { return Ok(None) };
        if self.has_open {
            for l in 0..batch.chunks.len() {
                if batch.fresh[l] || batch.chunks[l].is_empty() {
                    let j = self.judges[l].take();
                    self.finish_game(j);
                }
                if !batch.chunks[l].is_empty() && self.judges[l].is_none() {
                    self.judges[l] = Some((SurrenderJudge::new(), batch.results[l]));
                }
            }
        }
        if batch.chunks.iter().all(|c| c.is_empty()) {
            return Ok(None);
        }
        Ok(Some(batch))
    }

    /// 032-R18 MSE and the 041-AC4 replay for lanes `lanes` of `batch`: `open` is the group's
    /// win_open output, rows as in [`Batch::group`] (lane * t + step); every real step in game
    /// order is a decision.
    pub fn decide(&mut self, batch: &Batch, lanes: std::ops::Range<usize>, t: usize, open: &[f32]) {
        for (i, l) in lanes.enumerate() {
            let result = batch.results[l];
            for (k, s) in batch.chunks[l].iter().enumerate() {
                let x = open[i * t + k];
                if result != 0.0 {
                    self.open_se += ((x - result) as f64).powi(2);
                    self.open_n += 1;
                }
                if let Some((j, _)) = self.judges[l].as_mut() {
                    j.decide(s.frame, Some(x), || s.surrender_rule);
                }
            }
        }
    }

    fn finish_game(&mut self, judge: Option<(SurrenderJudge, f32)>) {
        let Some((j, result)) = judge else { return };
        self.sur.games += 1;
        if let Some(f) = j.first() {
            self.sur.held += 1;
            self.held_frames.push(f);
            if result != 0.0 {
                self.sur.held_decided += 1;
                self.sur.held_won += (result > 0.0) as u64;
            }
        }
    }

    /// Games read so far.
    pub fn games(&self) -> usize {
        self.feeder.state(None).games
    }

    /// Ends the open games; (win_open MSE, surrender replay), both `None` without the head.
    pub fn finish(&mut self) -> (Option<f64>, Option<SurrenderEval>) {
        if !self.has_open {
            return (None, None);
        }
        for l in 0..self.judges.len() {
            let j = self.judges[l].take();
            self.finish_game(j);
        }
        let mut s = self.sur.clone();
        let mut f = self.held_frames.clone();
        f.sort_unstable();
        s.false_surrender_rate = s.held_won as f64 / s.held_decided.max(1) as f64;
        s.held_frame_min = f.first().copied();
        s.held_frame_median = f.get(f.len() / 2).copied();
        s.held_frame_max = f.last().copied();
        (Some(self.open_se / self.open_n.max(1) as f64), Some(s))
    }
}
