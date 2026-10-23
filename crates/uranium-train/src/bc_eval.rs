//! 041 offline metrics and one-epoch lengths.
//!
//! `evaluate`: the 041 offline indicators on one split (default the target-map test set, 041-AC1),
//! with the labels fed to the autoregressive heads (teacher forcing, as in training): action_type
//! top-1/top-5, delay MAE (frames, E-9), target_loc tile error, selected F1 (argmax of every
//! selection step against the label set), and the per-head NLL averaged over the steps that use
//! the head (E-9). Games are read without z (041-AC2 condition) or symmetry. With `--games N`
//! the N games are chosen by a fixed-seed shuffle over the whole split (032-R17 ②, G-D8), so the
//! two cores of the core ablation are scored on the same games. With a win_open head (032-R18)
//! it also reports the head's MSE and replays the 031-R12 surrender check over every game
//! (041-AC4: how often it would have held, and how many of those games were won).
//!
//! `epoch_table`: updates of one epoch per stage (041-R13, 2026-10-18), from the step counts of
//! every game in the order the trainer reads them.

use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use serde::Serialize;
use uranium_env::features::DELAY_BUCKETS;
use uranium_env::surrender::SurrenderJudge;
use uranium_model::CoreInit;
use uranium_model::collate::{HostLabel, action_batch, obs_batch_padded, priv_batch};

use crate::bc::{BcConfig, Feeder};
use crate::data::{ChosenGames, MapStore, ShardSource, StepSource, TrainStep};
use crate::split::{Selection, Split};

pub use uranium_data::bc::{EpochCount, EpochTable, epoch_table, eval_games};

/// Output fields and their units (E-9), printed with every report.
pub const UNITS: [(&str, &str); 8] = [
    ("action_top1 / action_top5", "share of scored steps"),
    ("delay_mae_frames", "mean |predicted - label| delay in game frames (bucket values, 031-R11)"),
    ("delay_mae_buckets", "the same in delay bucket indices"),
    ("target_loc_err_*", "Euclidean distance in target_loc cells (map tiles)"),
    ("nll.<head>", "mean -log p of the label over the scored steps that use the head (nll_steps.<head>)"),
    ("steps", "steps with a usable label (041-R11 weight 1); padding and void labels are not scored"),
    ("win_open_mse", "mean (win_open - result)^2 over the steps of decided games (032-R18, result +1 / -1)"),
    ("surrender.*", "031-R12 check replayed per game (041-AC4): games, games where it held, the share of those won, first frame it held"),
];

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

#[derive(Serialize, Debug, Default)]
pub struct EvalReport {
    pub checkpoint: String,
    pub split: String,
    pub games: usize,
    /// `--games` selection seed (032-R17 ②); `None` = every game of the split.
    pub selection_seed: Option<u64>,
    /// Steps with a usable label (041-R11 weight 1).
    pub steps: u64,
    pub action_top1: f64,
    pub action_top5: f64,
    /// E-9: mean |predicted - label| delay in frames (the bucket values of `DELAY_BUCKETS`).
    pub delay_mae_frames: f64,
    /// Mean |predicted - label| delay bucket index.
    pub delay_mae_buckets: f64,
    pub target_loc_steps: u64,
    /// Mean / median Euclidean distance in target_loc tiles.
    pub target_loc_err_mean: f64,
    pub target_loc_err_median: f64,
    pub target_loc_exact: f64,
    pub selected_steps: u64,
    /// Micro F1 over selected units.
    pub selected_f1: f64,
    /// E-9: mean -log p of the label per head (041-R1 terms, unweighted), over the steps where
    /// the head is used (action and delay: every step; queued, selected, target_unit, target_loc:
    /// steps whose action has that argument).
    pub nll: std::collections::BTreeMap<String, f64>,
    /// Steps each `nll` entry is averaged over.
    pub nll_steps: std::collections::BTreeMap<String, u64>,
    /// 041-AC1: target-map test action_type top-1 >= 50%.
    pub ac1_pass: Option<bool>,
    /// 032-R18: win_open MSE over the steps of decided games; `None` without the head.
    pub win_open_mse: Option<f64>,
    /// 041-AC4; `None` without the head.
    pub surrender: Option<SurrenderEval>,
    /// Field -> unit (E-9).
    pub units: std::collections::BTreeMap<String, String>,
}

#[derive(Default)]
struct Acc {
    steps: u64,
    top1: u64,
    top5: u64,
    delay_abs: f64,
    delay_abs_frames: f64,
    loc_n: u64,
    loc_err: Vec<f32>,
    loc_exact: u64,
    sel_n: u64,
    sel_tp: u64,
    sel_fp: u64,
    sel_fn: u64,
    nll: [f64; 6],
    nll_n: [u64; 6],
    open_se: f64,
    open_n: u64,
    sur: SurrenderEval,
    held_frames: Vec<u32>,
}

impl Acc {
    /// 041-AC4: a lane's game is over (or the evaluation ended).
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
}

/// How `evaluate` picks games.
#[derive(Clone, Copy, Debug)]
pub struct EvalSelection {
    pub split: Split,
    /// `Some(n)`: n games by the fixed-seed shuffle (032-R17 ②); `None`: every game.
    pub games: Option<usize>,
    pub seed: u64,
}

pub fn evaluate<C: CoreInit>(dirs: &[PathBuf], ckpt: &Path, sel: EvalSelection, cfg: &BcConfig, dev: Device) -> Result<EvalReport> {
    let split = sel.split;
    // B-2: every game's map precompute is checked before the model is loaded; the scan also
    // gives the split's games for the selection.
    let maps = std::sync::Arc::new(MapStore::from_env());
    let src = ShardSource::from_dirs(dirs)?.select(Selection::Only(split));
    let scan = crate::data::check_maps(&src, Some(Selection::Only(split)), &maps)?;
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table())?;
    let (net, meta) = uranium_model::checkpoint::load::<C>(ckpt, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &dev)?;
    let net_cfg = meta.net_config()?;
    let mut ecfg = cfg.clone();
    ecfg.data.z_drop = 1.0;
    ecfg.data.augment = false;
    // 030-R11: the race mask is a training-only exception; evaluation sees the real input.
    ecfg.data.enemy_race_mask = 0.0;
    let mut feeder = Feeder::new(&ecfg).with_maps(maps);
    feeder.drain = true;
    let mut chosen;
    let mut all;
    let source: &mut dyn StepSource = match sel.games {
        Some(n) => {
            let ids = eval_games(&src, &scan, n, sel.seed);
            chosen = ChosenGames::new(src, &ids);
            &mut chosen
        }
        None => {
            all = src;
            &mut all
        }
    };
    let t = cfg.seq_len;
    // Lanes in groups of `micro_lanes` per forward, each group keeping its core state (as in
    // training: GPU memory follows the group).
    let g = match cfg.micro_lanes {
        0 => cfg.lanes,
        m => m.min(cfg.lanes),
    };
    let mut states: Vec<Option<C::State>> = (0..cfg.lanes.div_ceil(g)).map(|_| None).collect();
    let mut acc = Acc::default();
    let has_open = net.has_win_open();
    // 041-AC4: per lane, the surrender check of its current game and the game's result.
    let mut judges: Vec<Option<(SurrenderJudge, f32)>> = vec![None; cfg.lanes];
    while let Some(batch) = feeder.next_batch(source, None)? {
        let (chunks, fresh, results) = batch.parts();
        if has_open {
            for l in 0..chunks.len() {
                if fresh[l] || chunks[l].is_empty() {
                    acc.finish_game(judges[l].take());
                }
                if !chunks[l].is_empty() && judges[l].is_none() {
                    judges[l] = Some((SurrenderJudge::new(), results[l]));
                }
            }
        }
        // Drained lanes (no game left) are filled with another lane's step at weight 0.
        let Some(filler) = chunks.iter().find_map(|c| c.first()) else { break };
        for (gi, lanes) in (0..chunks.len()).collect::<Vec<_>>().chunks(g).enumerate() {
            if lanes.iter().all(|&l| chunks[l].is_empty()) {
                states[gi] = None;
                continue;
            }
            let b = lanes.len();
            let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
            let mut weights = Vec::with_capacity(b * t);
            for &l in lanes {
                let c = &chunks[l];
                for k in 0..t {
                    rows.push(c.get(k.min(c.len().saturating_sub(1))).unwrap_or(filler));
                    weights.push(if k < c.len() { c[k].weight } else { 0.0 });
                }
            }
            let obs: Vec<_> = rows.iter().map(|s| &s.obs).collect();
            let labels: Vec<&HostLabel> = rows.iter().map(|s| &s.label).collect();
            let privs: Vec<_> = rows.iter().map(|s| &s.privileged).collect();
            let ob = obs_batch_padded(&obs, cfg.pad_entities, &dev);
            let ab = action_batch(&labels, ob.entities(), net_cfg.heads.max_select, &dev);
            let pb = priv_batch(&privs, &dev);
            let keep: Vec<bool> = lanes.iter().map(|&l| !fresh[l]).collect();
            let keep_t = Tensor::<1, Bool>::from_data(TensorData::new(keep, [b]), &dev);
            let s0 = match states[gi].take() {
                Some(s) => C::mask_state(s, keep_t),
                None => net.init_state(b, &dev),
            };
            let out = net.forward_seq(&ob, &ab, Some(&pb), s0, t);
            states[gi] = Some(out.state);
            score(&mut acc, &out.heads, &labels, &weights);
            // 032-R18 MSE and the 041-AC4 replay, every real step in game order (a decision).
            if let Some(open) = out.win_open {
                let v = host_f32(open);
                for (i, &l) in lanes.iter().enumerate() {
                    for (k, s) in chunks[l].iter().enumerate() {
                        let x = v[i * t + k];
                        if results[l] != 0.0 {
                            acc.open_se += ((x - results[l]) as f64).powi(2);
                            acc.open_n += 1;
                        }
                        if let Some((j, _)) = judges[l].as_mut() {
                            j.decide(s.frame, Some(x), || s.surrender_rule);
                        }
                    }
                }
            }
        }
    }
    for j in judges.iter_mut() {
        acc.finish_game(j.take());
    }
    let mut rep = report(&acc, ckpt, split, feeder.state(None).games, sel.games.map(|_| sel.seed));
    if has_open {
        rep.win_open_mse = Some(acc.open_se / acc.open_n.max(1) as f64);
        let mut s = acc.sur.clone();
        let mut f = acc.held_frames.clone();
        f.sort_unstable();
        s.false_surrender_rate = s.held_won as f64 / s.held_decided.max(1) as f64;
        s.held_frame_min = f.first().copied();
        s.held_frame_median = f.get(f.len() / 2).copied();
        s.held_frame_max = f.last().copied();
        rep.surrender = Some(s);
    }
    Ok(rep)
}

const HEADS: [&str; 6] = ["action", "delay", "queued", "selected", "target_unit", "target_loc"];

fn report(acc: &Acc, ckpt: &Path, split: Split, games: usize, selection_seed: Option<u64>) -> EvalReport {
    let n = acc.steps.max(1) as f64;
    let mut errs = acc.loc_err.clone();
    errs.sort_by(|a, b| a.total_cmp(b));
    let prec = acc.sel_tp as f64 / (acc.sel_tp + acc.sel_fp).max(1) as f64;
    let rec = acc.sel_tp as f64 / (acc.sel_tp + acc.sel_fn).max(1) as f64;
    let top1 = acc.top1 as f64 / n;
    EvalReport {
        checkpoint: ckpt.display().to_string(),
        split: format!("{split:?}").to_lowercase(),
        games,
        selection_seed,
        steps: acc.steps,
        action_top1: top1,
        action_top5: acc.top5 as f64 / n,
        delay_mae_frames: acc.delay_abs_frames / n,
        delay_mae_buckets: acc.delay_abs / n,
        target_loc_steps: acc.loc_n,
        target_loc_err_mean: if errs.is_empty() { 0.0 } else { errs.iter().map(|&e| e as f64).sum::<f64>() / errs.len() as f64 },
        target_loc_err_median: errs.get(errs.len() / 2).map_or(0.0, |&e| e as f64),
        target_loc_exact: acc.loc_exact as f64 / acc.loc_n.max(1) as f64,
        selected_steps: acc.sel_n,
        selected_f1: if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 },
        nll: HEADS.iter().enumerate().map(|(h, k)| (k.to_string(), acc.nll[h] / acc.nll_n[h].max(1) as f64)).collect(),
        nll_steps: HEADS.iter().enumerate().map(|(h, k)| (k.to_string(), acc.nll_n[h])).collect(),
        ac1_pass: (split == Split::Test).then_some(top1 >= 0.5),
        win_open_mse: None,
        surrender: None,
        units: UNITS.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    }
}

fn host_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
}

fn host_i64<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_into_vec::<i64>().unwrap()
}

fn host_bool(t: Tensor<1, Bool>) -> Vec<bool> {
    host_i64(t.int()).into_iter().map(|v| v != 0).collect()
}

/// Delay of bucket `b` in frames (031-R11 bucket values).
fn bucket_frames(b: i64) -> f64 {
    DELAY_BUCKETS[(b.max(0) as usize).min(DELAY_BUCKETS.len() - 1)] as f64
}

/// Adds one batch (rows with weight 0 are skipped: padding and 041-R11 invalid labels).
fn score(acc: &mut Acc, heads: &uranium_model::heads::HeadsOut, labels: &[&HostLabel], weights: &[f32]) {
    let d = &heads.dists;
    let rows = labels.len();
    let a = d.action.dims()[1];
    let action = host_f32(d.action.clone());
    let delay = host_i64(d.delay.clone().argmax(1));
    let loc = host_i64(d.target_loc.clone().argmax(1));
    let m = (d.target_loc.dims()[1] as f64).sqrt().round() as i64;
    let [_, s, _] = d.selected.dims();
    let sel = host_i64(d.selected.clone().argmax(2));
    let end = d.selected.dims()[2] as i64 - 1;
    let lp = &heads.logp;
    let nll: Vec<Vec<f32>> = [&lp.action, &lp.delay, &lp.queued, &lp.selected, &lp.target_unit, &lp.target_loc].iter().map(|t| host_f32((*t).clone())).collect();
    // E-9: which heads each row's (teacher-forced) action uses.
    let uses = [host_bool(d.uses_queued.clone()), host_bool(d.uses_select.clone()), host_bool(d.uses_target_unit.clone()), host_bool(d.uses_target_loc.clone())];
    for r in 0..rows {
        if weights[r] <= 0.0 {
            continue;
        }
        let l = labels[r];
        acc.steps += 1;
        let row = &action[r * a..(r + 1) * a];
        let want = row[l.action as usize];
        let better = row.iter().filter(|&&x| x > want).count();
        acc.top1 += (better == 0) as u64;
        acc.top5 += (better < 5) as u64;
        acc.delay_abs += (delay[r] - l.delay_bucket as i64).abs() as f64;
        acc.delay_abs_frames += (bucket_frames(delay[r]) - bucket_frames(l.delay_bucket as i64)).abs();
        if let Some(t) = l.target_loc {
            let (p, t) = (loc[r], t as i64);
            let (dy, dx) = ((p / m - t / m) as f32, (p % m - t % m) as f32);
            acc.loc_err.push((dx * dx + dy * dy).sqrt());
            acc.loc_exact += (p == t) as u64;
            acc.loc_n += 1;
        }
        if !l.selected.is_empty() {
            acc.sel_n += 1;
            // Teacher-forced steps 0..=len(label): argmax per step until END.
            let mut pred = Vec::new();
            for k in 0..s.min(l.selected.len() + 1) {
                let e = sel[r * s + k];
                if e == end {
                    break;
                }
                pred.push(e as usize);
            }
            let tp = pred.iter().filter(|e| l.selected.contains(e)).count() as u64;
            acc.sel_tp += tp;
            acc.sel_fp += pred.len() as u64 - tp;
            acc.sel_fn += l.selected.len() as u64 - tp;
        }
        for (h, v) in nll.iter().enumerate() {
            let used = match h {
                0 | 1 => true,
                _ => uses[h - 2][r],
            };
            if used {
                acc.nll[h] -= v[r] as f64;
                acc.nll_n[h] += 1;
            }
        }
    }
}
