//! 041 offline metrics and one-epoch lengths.
//!
//! `evaluate`: the 041 offline indicators on one split (default the target-map test set, 041-AC1),
//! with the labels fed to the autoregressive heads (teacher forcing, as in training): action_type
//! top-1/top-5, delay bucket MAE, target_loc tile error, selected F1 (argmax of every selection
//! step against the label set), and the loss terms. Games are read in file order without z
//! (041-AC2 condition) or symmetry.
//!
//! `epoch_table`: updates of one epoch per stage (041-R13, 2026-10-18), from the step counts of
//! every game in the order the trainer reads them.

use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};
use serde::Serialize;
use uranium_model::collate::{HostLabel, action_batch, obs_batch_padded, priv_batch};
use uranium_model::CoreInit;
use uranium_replay::shard::{GameMeta, ShardGame, read};

use crate::bc::{BcConfig, Feeder};
use crate::data::{HasMeta, ShardSource, StepSource, TrainStep, epoch_updates};
use crate::split::{Selection, Split};

/// One game's selection data and length (the epoch count does not keep the frames).
#[derive(Clone)]
struct LenItem {
    meta: GameMeta,
    len: usize,
}

impl HasMeta for LenItem {
    fn meta(&self) -> &GameMeta {
        &self.meta
    }
}

#[derive(Serialize, Debug, Clone, Copy, Default)]
pub struct EpochCount {
    pub updates: u64,
    pub games: usize,
    pub steps: u64,
}

#[derive(Serialize, Debug)]
pub struct EpochTable {
    pub lanes: usize,
    pub seq_len: usize,
    pub seed: u64,
    pub s1: EpochCount,
    pub s2: EpochCount,
    /// Ablation (A) reads the S2 games in the same order.
    pub ablation_a: EpochCount,
}

/// Reads every shard once (in parallel) and counts the updates of one epoch per stage.
pub fn epoch_table(dirs: &[PathBuf], cfg: &BcConfig) -> Result<EpochTable> {
    let base = ShardSource::from_dirs(dirs)?;
    let files: Vec<PathBuf> = base.files().to_vec();
    // A decoded shard is close to 1 GB: a few readers keep a 16 GB host (BC250) out of swap.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(4);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut per_file: Vec<Option<Vec<LenItem>>> = vec![None; files.len()];
    let results = std::sync::Mutex::new(&mut per_file);
    std::thread::scope(|s| -> Result<()> {
        let mut hs = Vec::new();
        for _ in 0..threads {
            hs.push(s.spawn(|| -> Result<()> {
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if k >= files.len() {
                        return Ok(());
                    }
                    let games = read(&files[k]).map_err(|e| anyhow::anyhow!("{}: {e:#}", files[k].display()))?.games;
                    let items = games.into_iter().map(|g: ShardGame| LenItem { len: g.steps.len(), meta: g.meta }).collect();
                    results.lock().unwrap()[k] = Some(items);
                }
            }));
        }
        for h in hs {
            h.join().map_err(|_| anyhow::anyhow!("reader thread panicked"))??;
        }
        Ok(())
    })?;
    let by_path: std::collections::HashMap<&PathBuf, &Vec<LenItem>> = files.iter().zip(per_file.iter().map(|v| v.as_ref().unwrap())).collect();
    let count = |sel: Selection| -> Result<EpochCount> {
        let src = ShardSource::from_dirs(dirs)?.select(sel).shuffled(cfg.data.seed);
        let mut lens = Vec::new();
        for (k, f) in src.files().iter().enumerate() {
            lens.extend(src.order_file(k, by_path[f].clone()).into_iter().map(|i| i.len));
        }
        let games = lens.iter().filter(|&&l| l > 0).count();
        let steps = lens.iter().map(|&l| l as u64).sum();
        Ok(EpochCount { updates: epoch_updates(lens, cfg.lanes, cfg.seq_len), games, steps })
    };
    let s1 = count(Selection::S1)?;
    let s2 = count(Selection::S2)?;
    Ok(EpochTable { lanes: cfg.lanes, seq_len: cfg.seq_len, seed: cfg.data.seed, s1, s2, ablation_a: s2 })
}

/// Stops a source after `n` games.
struct Limit<'a> {
    src: &'a mut dyn StepSource,
    left: Option<usize>,
}

impl StepSource for Limit<'_> {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        if self.left == Some(0) {
            return Ok(None);
        }
        if let Some(l) = &mut self.left {
            *l -= 1;
        }
        self.src.next_game()
    }
}

#[derive(Serialize, Debug, Default)]
pub struct EvalReport {
    pub checkpoint: String,
    pub split: String,
    pub games: usize,
    /// Steps with a usable label (041-R11 weight 1).
    pub steps: u64,
    pub action_top1: f64,
    pub action_top5: f64,
    /// Mean |predicted - label| delay bucket.
    pub delay_mae_buckets: f64,
    pub target_loc_steps: u64,
    /// Mean / median Euclidean distance in target_loc tiles.
    pub target_loc_err_mean: f64,
    pub target_loc_err_median: f64,
    pub target_loc_exact: f64,
    pub selected_steps: u64,
    /// Micro F1 over selected units.
    pub selected_f1: f64,
    /// Mean -log p of the label per head (041-R1 terms, unweighted).
    pub nll: std::collections::BTreeMap<String, f64>,
    /// 041-AC1: target-map test action_type top-1 >= 50%.
    pub ac1_pass: Option<bool>,
}

#[derive(Default)]
struct Acc {
    steps: u64,
    top1: u64,
    top5: u64,
    delay_abs: f64,
    loc_n: u64,
    loc_err: Vec<f32>,
    loc_exact: u64,
    sel_n: u64,
    sel_tp: u64,
    sel_fp: u64,
    sel_fn: u64,
    nll: [f64; 6],
}

pub fn evaluate<C: CoreInit>(dirs: &[PathBuf], ckpt: &Path, split: Split, max_games: Option<usize>, cfg: &BcConfig, dev: Device) -> Result<EvalReport> {
    let cols = uranium_model::batch::ActionColumns::from_table(uranium_env::features::extract::action_table())?;
    let (net, meta) = uranium_model::checkpoint::load::<C>(ckpt, &cols, uranium_env::features::FEATURE_SCHEMA_VERSION, uranium_env::actions::ACTION_SCHEMA_VERSION, &dev)?;
    let net_cfg = meta.net_config()?;
    let mut ecfg = cfg.clone();
    ecfg.data.z_drop = 1.0;
    ecfg.data.augment = false;
    let mut feeder = Feeder::new(&ecfg);
    let mut src = ShardSource::from_dirs(dirs)?.select(Selection::Only(split));
    let mut limited = Limit { src: &mut src, left: max_games };
    let t = cfg.seq_len;
    let mut state: Option<C::State> = None;
    let mut acc = Acc::default();
    while let Some(batch) = feeder.next_batch(&mut limited, None)? {
        let (chunks, fresh, results) = batch.parts();
        let b = chunks.len();
        let mut rows: Vec<&TrainStep> = Vec::with_capacity(b * t);
        let mut weights = Vec::with_capacity(b * t);
        for c in chunks {
            for k in 0..t {
                rows.push(&c[k.min(c.len() - 1)]);
                weights.push(if k < c.len() { c[k].weight } else { 0.0 });
            }
        }
        let _ = results;
        let obs: Vec<_> = rows.iter().map(|s| &s.obs).collect();
        let labels: Vec<&HostLabel> = rows.iter().map(|s| &s.label).collect();
        let privs: Vec<_> = rows.iter().map(|s| &s.privileged).collect();
        let ob = obs_batch_padded(&obs, cfg.pad_entities, &dev);
        let ab = action_batch(&labels, ob.entities(), net_cfg.heads.max_select, &dev);
        let pb = priv_batch(&privs, &dev);
        let keep: Vec<bool> = fresh.iter().map(|f| !f).collect();
        let keep_t = Tensor::<1, Bool>::from_data(TensorData::new(keep, [b]), &dev);
        let s0 = match state.take() {
            Some(s) if b == cfg.lanes => C::mask_state(s, keep_t),
            _ => net.init_state(b, &dev),
        };
        let out = net.forward_seq(&ob, &ab, Some(&pb), s0, t);
        state = Some(out.state);
        score(&mut acc, &out.heads, &labels, &weights);
    }
    let n = acc.steps.max(1) as f64;
    let mut errs = acc.loc_err.clone();
    errs.sort_by(|a, b| a.total_cmp(b));
    let prec = acc.sel_tp as f64 / (acc.sel_tp + acc.sel_fp).max(1) as f64;
    let rec = acc.sel_tp as f64 / (acc.sel_tp + acc.sel_fn).max(1) as f64;
    let names = ["action", "delay", "queued", "selected", "target_unit", "target_loc"];
    let top1 = acc.top1 as f64 / n;
    Ok(EvalReport {
        checkpoint: ckpt.display().to_string(),
        split: format!("{split:?}").to_lowercase(),
        games: feeder.state(None).games,
        steps: acc.steps,
        action_top1: top1,
        action_top5: acc.top5 as f64 / n,
        delay_mae_buckets: acc.delay_abs / n,
        target_loc_steps: acc.loc_n,
        target_loc_err_mean: if errs.is_empty() { 0.0 } else { errs.iter().map(|&e| e as f64).sum::<f64>() / errs.len() as f64 },
        target_loc_err_median: errs.get(errs.len() / 2).map_or(0.0, |&e| e as f64),
        target_loc_exact: acc.loc_exact as f64 / acc.loc_n.max(1) as f64,
        selected_steps: acc.sel_n,
        selected_f1: if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 },
        nll: names.iter().zip(acc.nll).map(|(k, v)| (k.to_string(), v / n)).collect(),
        ac1_pass: (split == Split::Test).then_some(top1 >= 0.5),
    })
}

fn host_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
}

fn host_i64<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_into_vec::<i64>().unwrap()
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
            acc.nll[h] -= v[r] as f64;
        }
    }
}
