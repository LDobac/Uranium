use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;
use uranium_match::surrender::SurrenderJudge;
use crate::shard::ShardGame;

use crate::bc::{Batch, BcConfig, Feeder, eval_games};
use crate::data::{ChosenGames, GameId, MapStore, ShardSource, StepSource, cut_game};
use crate::split::{Selection, Split};

#[derive(Serialize, Debug, Default, Clone)]
pub struct SurrenderEval {
    pub games: u64,
    pub held: u64,
    pub held_decided: u64,
    pub held_won: u64,
    pub false_surrender_rate: f64,
    pub held_frame_min: Option<u32>,
    pub held_frame_median: Option<u32>,
    pub held_frame_max: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct EvalSelection {
    pub split: Split,
    pub games: Option<usize>,
    pub seed: u64,
    pub until_frame: Option<u32>,
}

struct Until<'a> {
    src: &'a mut dyn StepSource,
    frame: Option<u32>,
}

impl StepSource for Until<'_> {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        let mut g = self.src.next_game()?;
        if let (Some(g), Some(f)) = (g.as_mut(), self.frame) {
            cut_game(g, f);
        }
        Ok(g)
    }

    fn last_id(&self) -> Option<GameId> {
        self.src.last_id()
    }
}

#[allow(clippy::large_enum_variant)]
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
    until_frame: Option<u32>,
    has_open: bool,
    judges: Vec<Option<(SurrenderJudge, f32)>>,
    sur: SurrenderEval,
    held_frames: Vec<u32>,
    open_se: f64,
    open_n: u64,
}

impl EvalStream {
    pub fn new(dirs: &[PathBuf], cfg: &BcConfig, sel: EvalSelection, maps: Option<Arc<MapStore>>, has_open: bool) -> Result<Self> {
        let maps = maps.unwrap_or_else(|| Arc::new(MapStore::from_env()));
        let src = ShardSource::from_dirs(dirs)?.select(Selection::Only(sel.split));
        let scan = crate::data::check_maps(&src, Some(Selection::Only(sel.split)), &maps)?;
        let mut ecfg = cfg.clone();
        ecfg.data.z_drop = 1.0;
        ecfg.data.augment = false;
        ecfg.data.enemy_race_mask = 0.0;
        ecfg.data.chain_start_sources = None;
        let mut feeder = Feeder::new(&ecfg).with_maps(maps);
        feeder.drain = true;
        let source = match sel.games {
            Some(n) => {
                let ids = eval_games(&src, &scan, n, sel.seed);
                Source::Chosen(ChosenGames::with_until(src, &ids, sel.until_frame))
            }
            None => Source::All(src.skip_empty(&scan)),
        };
        Ok(Self { feeder, source, until_frame: sel.until_frame, has_open, judges: vec![None; cfg.lanes], sur: SurrenderEval::default(), held_frames: Vec::new(), open_se: 0.0, open_n: 0 })
    }

    pub fn with_chain_start(mut self, sources: Vec<String>) -> Result<Self> {
        crate::data::check_chain_start_sources(&sources)?;
        self.feeder.set_chain_start(sources);
        Ok(self)
    }

    pub fn next(&mut self) -> Result<Option<Batch>> {
        let mut src = Until { src: self.source.get(), frame: self.until_frame };
        let Some(batch) = self.feeder.next_batch(&mut src, None)? else { return Ok(None) };
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

    pub fn games(&self) -> usize {
        self.feeder.state(None).games
    }

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
