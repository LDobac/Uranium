use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use uranium_env::actions::CAPABILITIES;
use uranium_env::features::augment::{Sym, transform_target_loc, transform_with_starts};
use uranium_env::features::extract::{action_table, dims};
use uranium_env::features::*;
use uranium_env::features::label::LabelWalker;
use uranium_env::features::label::HostLabel;
use uranium_replay::shard::{GameMeta, ShardGame, read, result_source};

pub trait StepSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>>;

    fn last_id(&self) -> Option<GameId> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GameId {
    pub file: usize,
    pub game: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourcePos {
    pub next: GameId,
}

pub struct ShardSource {
    files: Vec<PathBuf>,
    keys: Vec<String>,
    next_file: usize,
    games: VecDeque<(usize, ShardGame)>,
    cur_file: usize,
    last: Option<GameId>,
    select: Option<crate::split::Selection>,
    subsample: Option<(crate::split::Subsample, u64)>,
    seed: Option<u64>,
    only_files: Option<HashSet<usize>>,
    dirs: Vec<PathBuf>,
}

impl ShardSource {
    pub fn new(dir: &std::path::Path) -> Result<Self> {
        Self::from_dirs(std::slice::from_ref(&dir.to_path_buf()))
    }

    pub fn from_dirs(dirs: &[PathBuf]) -> Result<Self> {
        let mut found: Vec<(String, PathBuf)> = Vec::new();
        for d in dirs {
            found.extend(shard_keys(d)?);
        }
        let mut seen: HashMap<PathBuf, &Path> = HashMap::new();
        for (_, f) in &found {
            let real = std::fs::canonicalize(f).with_context(|| f.display().to_string())?;
            if let Some(first) = seen.insert(real, f) {
                anyhow::bail!("the shard file {} is given twice (also as {}): give each source once, by --dataset or --shards (041-R13)", f.display(), first.display());
            }
        }
        drop(seen);
        found.sort_by(|a, b| key_cmp(&a.0, &b.0).then_with(|| a.1.cmp(&b.1)));
        if let Some(w) = found.windows(2).find(|w| w[0].0 == w[1].0) {
            anyhow::bail!(
                "two shard files have the same key {} ({} and {}): give each source once, by --dataset or --shards (041-R13)",
                w[0].0,
                w[0].1.display(),
                w[1].1.display()
            );
        }
        let (keys, files) = found.into_iter().unzip();
        Ok(Self { files, keys, next_file: 0, games: VecDeque::new(), cur_file: 0, last: None, select: None, subsample: None, seed: None, only_files: None, dirs: dirs.to_vec() })
    }

    pub fn select(mut self, sel: crate::split::Selection) -> Self {
        self.select = Some(sel);
        self
    }

    pub fn subsample(mut self, sub: Option<crate::split::Subsample>, seed: u64) -> Self {
        self.subsample = sub.map(|s| (s, seed));
        self
    }

    pub fn shuffled(mut self, seed: u64) -> Self {
        let mut order: Vec<usize> = (0..self.files.len()).collect();
        shuffle(&mut order, &mut Rng::new(seed));
        self.files = order.iter().map(|&i| self.files[i].clone()).collect();
        self.keys = order.iter().map(|&i| self.keys[i].clone()).collect();
        self.seed = Some(seed);
        self
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    pub fn only_files(mut self, keep: HashSet<usize>) -> Self {
        self.only_files = Some(keep);
        self
    }

    pub fn fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for k in &self.keys {
            h.update(key_hash_text(k).as_bytes());
            h.update(b"\n");
        }
        let hex: String = h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect();
        format!("{}:{hex}", self.files.len())
    }

    pub fn file_games(&self, k: usize) -> Result<Vec<ShardGame>> {
        let f = &self.files[k];
        let games = read(f).with_context(|| f.display().to_string())?.games;
        Ok(self.order_file(k, games))
    }

    pub fn order_file<T: HasMeta>(&self, k: usize, mut games: Vec<T>) -> Vec<T> {
        if let Some(sel) = self.select {
            games.retain(|g| crate::split::selects(sel, g.meta()));
        }
        if let Some((sub, seed)) = &self.subsample {
            games.retain(|g| sub.keeps(g.meta(), *seed));
        }
        if let Some(seed) = self.seed {
            shuffle(&mut games, &mut Rng::new(seed ^ (k as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03)));
        }
        games
    }

    pub fn position(&self) -> SourcePos {
        match self.games.front() {
            Some((g, _)) => SourcePos { next: GameId { file: self.cur_file, game: *g } },
            None => SourcePos { next: GameId { file: self.next_file, game: 0 } },
        }
    }

    pub fn seek(&mut self, pos: SourcePos) -> Result<()> {
        self.games.clear();
        self.next_file = pos.next.file;
        if pos.next.file < self.files.len() {
            let games = self.file_games(pos.next.file)?;
            self.cur_file = pos.next.file;
            self.next_file = pos.next.file + 1;
            self.games = games.into_iter().enumerate().skip(pos.next.game).collect();
        }
        Ok(())
    }

    pub fn game(&self, id: GameId) -> Result<ShardGame> {
        let mut games = self.file_games(id.file)?;
        anyhow::ensure!(id.game < games.len(), "{:?} is past the end of {}", id, self.files[id.file].display());
        Ok(games.swap_remove(id.game))
    }

    pub fn games(&self, ids: &[GameId]) -> Result<Vec<ShardGame>> {
        let mut by_file: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (slot, id) in ids.iter().enumerate() {
            by_file.entry(id.file).or_default().push(slot);
        }
        let mut out: Vec<Option<ShardGame>> = (0..ids.len()).map(|_| None).collect();
        for (file, slots) in by_file {
            let mut games: Vec<Option<ShardGame>> = self.file_games(file)?.into_iter().map(Some).collect();
            for slot in slots {
                let id = ids[slot];
                anyhow::ensure!(id.game < games.len(), "{:?} is past the end of {}", id, self.files[file].display());
                out[slot] = Some(games[id.game].take().ok_or_else(|| anyhow::anyhow!("{id:?} asked for twice"))?);
            }
        }
        Ok(out.into_iter().map(|g| g.expect("every slot filled")).collect())
    }
}

pub trait HasMeta {
    fn meta(&self) -> &uranium_replay::shard::GameMeta;
}

impl HasMeta for ShardGame {
    fn meta(&self) -> &uranium_replay::shard::GameMeta {
        &self.meta
    }
}

pub fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
}

pub fn shard_keys(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    const LAYOUT: &str = "expected <source>/<part>/shard_*.bin.zst under a source folder's parent, or <part>/shard_*.bin.zst under a source folder (040-R13, 041-R13)";
    let files = shard_files(dir)?;
    let real = std::fs::canonicalize(dir).with_context(|| dir.display().to_string())?;
    let name = real.file_name().map(|n| n.to_string_lossy().into_owned());
    let mut depth = None;
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let rel: Vec<String> = f.strip_prefix(dir)?.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        let key = match (rel.len(), &name) {
            (3, _) => rel.join("/"),
            (2, Some(n)) => format!("{n}/{}", rel.join("/")),
            _ => anyhow::bail!("{}: shard at depth {} under {}; {LAYOUT}", f.display(), rel.len(), dir.display()),
        };
        if *depth.get_or_insert(rel.len()) != rel.len() {
            anyhow::bail!("{}: shards at two depths under {}; {LAYOUT}", f.display(), dir.display());
        }
        out.push((key, f));
    }
    Ok(out)
}

fn key_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.split('/').cmp(b.split('/'))
}

fn key_hash_text(key: &str) -> String {
    key.rsplit('/').collect::<Vec<_>>().join("/")
}

pub fn shard_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for e in std::fs::read_dir(dir).with_context(|| dir.display().to_string())? {
            let e = e?;
            let p = e.path();
            if e.file_type()?.is_dir() {
                walk(&p, out)?;
            } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("shard_") && n.ends_with(".bin.zst")) {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, &mut files)?;
    files.sort();
    Ok(files)
}

impl StepSource for ShardSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        while self.games.is_empty() {
            if self.next_file >= self.files.len() {
                return Ok(None);
            }
            let k = self.next_file;
            self.next_file += 1;
            if self.only_files.as_ref().is_some_and(|keep| !keep.contains(&k)) {
                continue;
            }
            self.cur_file = k;
            self.games = self.file_games(k)?.into_iter().enumerate().collect();
        }
        let (i, g) = self.games.pop_front().unwrap();
        self.last = Some(GameId { file: self.cur_file, game: i });
        Ok(Some(g))
    }

    fn last_id(&self) -> Option<GameId> {
        self.last
    }
}

pub struct VecSource(pub VecDeque<ShardGame>);

impl StepSource for VecSource {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        Ok(self.0.pop_front())
    }
}

pub fn epoch_updates(lens: impl IntoIterator<Item = usize>, lanes: usize, seq_len: usize) -> u64 {
    let mut lens = lens.into_iter().filter(|&l| l > 0);
    let mut rem = vec![0usize; lanes];
    let mut updates = 0u64;
    loop {
        for r in &mut rem {
            if *r == 0 {
                match lens.next() {
                    Some(l) => *r = l,
                    None => return updates,
                }
            }
        }
        for r in &mut rem {
            *r -= (*r).min(seq_len);
        }
        updates += 1;
    }
}

#[derive(Debug, Clone)]
pub struct TrainStep {
    pub obs: Observation,
    pub label: HostLabel,
    pub privileged: PrivilegedTensors,
    pub has_privileged: bool,
    pub weight: f32,
    pub frame: u32,
    pub surrender_rule: bool,
}

#[derive(Debug, Clone)]
pub struct GameSeq {
    pub steps: Vec<TrainStep>,
    pub result: f32,
    pub invalid_labels: usize,
}

pub use uranium_env::maps::family_index;

pub struct MapStore {
    kind: MapsKind,
    cache: Mutex<HashMap<String, Arc<MapPrecompute>>>,
}

enum MapsKind {
    Dir(PathBuf),
    Unset(String),
    Synthetic,
}

impl MapStore {
    pub fn from_env() -> Self {
        match uranium_env::maps::maps_dir() {
            Ok(d) => Self::dir(d),
            Err(e) => Self { kind: MapsKind::Unset(format!("{e:#}")), cache: Mutex::default() },
        }
    }

    pub fn dir(dir: impl Into<PathBuf>) -> Self {
        Self { kind: MapsKind::Dir(dir.into()), cache: Mutex::default() }
    }

    pub fn synthetic() -> Self {
        Self { kind: MapsKind::Synthetic, cache: Mutex::default() }
    }

    fn folder(&self) -> Result<Option<&Path>> {
        match &self.kind {
            MapsKind::Synthetic => Ok(None),
            MapsKind::Unset(e) => anyhow::bail!("{e} (B-2: BC needs the map precomputes)"),
            MapsKind::Dir(d) => Ok(Some(d)),
        }
    }

    pub fn get(&self, map_hash: &str) -> Result<Option<Arc<MapPrecompute>>> {
        let Some(dir) = self.folder()? else { return Ok(None) };
        if let Some(p) = self.cache.lock().unwrap().get(map_hash) {
            return Ok(Some(p.clone()));
        }
        let p = uranium_env::maps::load_precompute(dir, map_hash).ok_or_else(|| {
            anyhow::anyhow!(
                "no map precompute v{} for map {map_hash} in {} (B-2: copy the build's precomputes there, or rebuild them with `uranium-replay build`)",
                uranium_env::features::PRECOMPUTE_VERSION,
                dir.display()
            )
        })?;
        let p = Arc::new(p);
        self.cache.lock().unwrap().insert(map_hash.to_string(), p.clone());
        Ok(Some(p))
    }

    pub fn missing<'a>(&self, hashes: impl IntoIterator<Item = &'a str>) -> Result<Vec<String>> {
        let Some(dir) = self.folder()? else { return Ok(Vec::new()) };
        let set: std::collections::BTreeSet<&str> = hashes.into_iter().collect();
        Ok(set.into_iter().filter(|h| uranium_env::maps::load_precompute(dir, h).is_none()).map(str::to_string).collect())
    }

    pub fn check(&self, games: &[&GameSummary]) -> Result<()> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for g in games {
            *counts.entry(g.meta.map_hash.as_str()).or_default() += 1;
        }
        let missing = self.missing(counts.keys().copied())?;
        if missing.is_empty() {
            return Ok(());
        }
        let list: Vec<String> = missing.iter().map(|h| format!("{h} ({} games)", counts[h.as_str()])).collect();
        let dir = self.folder()?.map_or_else(String::new, |d| d.display().to_string());
        anyhow::bail!(
            "{} map hash(es) of the selected games have no map precompute v{} in {dir} (B-2; copy the build's `<map_hash>.toml` files there): {}",
            missing.len(),
            uranium_env::features::PRECOMPUTE_VERSION,
            list.join(", ")
        )
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GameSummary {
    pub meta: GameMeta,
    pub steps: usize,
    pub privileged_none: usize,
}

impl HasMeta for GameSummary {
    fn meta(&self) -> &GameMeta {
        &self.meta
    }
}

pub fn scan_files(files: &[PathBuf]) -> Result<Vec<Vec<GameSummary>>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(4);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut per_file: Vec<Option<Vec<GameSummary>>> = vec![None; files.len()];
    let results = Mutex::new(&mut per_file);
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
                    let items = games
                        .into_iter()
                        .map(|g| GameSummary { steps: g.steps.len(), privileged_none: g.steps.iter().filter(|s| s.privileged.is_none()).count(), meta: g.meta })
                        .collect();
                    results.lock().unwrap()[k] = Some(items);
                }
            }));
        }
        for h in hs {
            h.join().map_err(|_| anyhow::anyhow!("reader thread panicked"))??;
        }
        Ok(())
    })?;
    Ok(per_file.into_iter().map(|v| v.expect("every file read")).collect())
}

pub fn check_maps(src: &ShardSource, sel: Option<crate::split::Selection>, maps: &MapStore) -> Result<Vec<Vec<GameSummary>>> {
    let started = std::time::Instant::now();
    let scan = scan_cached(src)?;
    let games: Vec<&GameSummary> = scan.iter().flatten().filter(|g| sel.is_none_or(|s| crate::split::selects(s, &g.meta))).collect();
    maps.check(&games)?;
    let n_maps = games.iter().map(|g| g.meta.map_hash.as_str()).collect::<HashSet<_>>().len();
    crate::elog!("{}", serde_json::json!({ "map_precomputes": "ok (B-2)", "games": games.len(), "maps": n_maps, "secs": started.elapsed().as_secs() }));
    Ok(scan)
}

const CACHE_DIR_ENV: &str = "URANIUM_CACHE_DIR";
const SCAN_CACHE_VERSION: u32 = 1;
const SCAN_SHARD_SCHEMA: u32 = 6;

pub fn scan_cache_dir() -> Option<PathBuf> {
    std::env::var_os(CACHE_DIR_ENV).filter(|v| !v.is_empty()).map(|v| PathBuf::from(v).join("scans"))
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScanCache {
    version: u32,
    shard_schema: u32,
    key: String,
    files: Vec<(String, Vec<GameSummary>)>,
}

pub fn scan_cached(src: &ShardSource) -> Result<Vec<Vec<GameSummary>>> {
    scan_cached_in(src, scan_cache_dir().as_deref())
}

pub fn scan_cached_in(src: &ShardSource, cache: Option<&Path>) -> Result<Vec<Vec<GameSummary>>> {
    let mut order: Vec<usize> = (0..src.files().len()).collect();
    order.sort_by(|&a, &b| key_cmp(&src.keys()[a], &src.keys()[b]));
    let sorted: Vec<PathBuf> = order.iter().map(|&i| src.files()[i].clone()).collect();
    let keys: Vec<String> = order.iter().map(|&i| src.keys()[i].clone()).collect();
    let key = files_fingerprint(src.dirs(), &sorted, &keys, true)?.replace(':', "-");
    let path = cache.map(|d| d.join(format!("scan-{key}.json")));
    let cached = path.as_ref().and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice::<ScanCache>(&b).ok()).filter(|c| {
        c.version == SCAN_CACHE_VERSION
            && c.shard_schema == SCAN_SHARD_SCHEMA
            && c.key == key
            && c.files.len() == keys.len()
            && c.files.iter().zip(&keys).all(|((k, _), want)| k == want)
    });
    let by_sorted: Vec<Vec<GameSummary>> = match cached {
        Some(c) => {
            crate::elog!("[scan] {} files from the cache {}", c.files.len(), path.as_ref().unwrap().display());
            c.files.into_iter().map(|(_, g)| g).collect()
        }
        None => {
            let scan = scan_files(&sorted)?;
            if let Some(p) = &path {
                let c = ScanCache { version: SCAN_CACHE_VERSION, shard_schema: SCAN_SHARD_SCHEMA, key: key.clone(), files: keys.iter().cloned().zip(scan.iter().cloned()).collect() };
                let write = || -> Result<()> {
                    std::fs::create_dir_all(p.parent().unwrap())?;
                    let tmp = p.with_extension(format!("json.tmp-{}", std::process::id()));
                    std::fs::write(&tmp, serde_json::to_vec(&c)?)?;
                    std::fs::rename(&tmp, p)?;
                    Ok(())
                };
                if let Err(e) = write() {
                    crate::elog!("[scan] could not write the scan cache {} ({e:#}); the next start scans again", p.display());
                }
            }
            scan
        }
    };
    let index: HashMap<&PathBuf, usize> = sorted.iter().enumerate().map(|(i, f)| (f, i)).collect();
    Ok(src.files().iter().map(|f| by_sorted[index[f]].clone()).collect())
}

pub fn value_weight(meta: &GameMeta) -> f32 {
    if meta.result_source == result_source::NONE { 0.0 } else { 1.0 }
}

pub fn dataset_fingerprint(src: &ShardSource) -> Result<String> {
    files_fingerprint(src.dirs(), src.files(), src.keys(), false)
}

fn files_fingerprint(dirs: &[PathBuf], files: &[PathBuf], keys: &[String], local: bool) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut manifests: HashMap<PathBuf, Option<uranium_store::storage::Manifest>> = HashMap::new();
    let mut h = Sha256::new();
    for (f, key) in files.iter().zip(keys) {
        let mut entry = None;
        if let Some(root) = dirs.iter().find(|d| f.starts_with(d)) {
            for anc in f.ancestors().skip(1) {
                if !anc.starts_with(root) {
                    break;
                }
                if !manifests.contains_key(anc) {
                    let m = if anc.join(uranium_store::storage::MANIFEST_FILE).is_file() { Some(uranium_store::storage::Manifest::read(anc)?) } else { None };
                    manifests.insert(anc.to_path_buf(), m);
                }
                if let Some(m) = &manifests[anc] {
                    let rel = f.strip_prefix(anc)?.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
                    entry = m.files.get(&rel).map(|e| e.sha256.clone());
                    break;
                }
            }
        }
        let id = match entry {
            Some(sha) => sha,
            None => {
                let m = std::fs::metadata(f).with_context(|| f.display().to_string())?;
                let mtime = if local { m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos()) } else { 0 };
                if local { format!("bytes:{}:mtime:{mtime}", m.len()) } else { format!("bytes:{}", m.len()) }
            }
        };
        h.update(key_hash_text(key).as_bytes());
        h.update(b"\t");
        h.update(id.as_bytes());
        h.update(b"\n");
    }
    let hex: String = h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect();
    Ok(format!("{}:{hex}", files.len()))
}

pub fn sample_games(counts: &[usize], n: usize, seed: u64) -> Vec<GameId> {
    let mut ids: Vec<GameId> = counts.iter().enumerate().flat_map(|(file, &c)| (0..c).map(move |game| GameId { file, game })).collect();
    shuffle(&mut ids, &mut Rng::new(seed));
    ids.truncate(n);
    ids.sort_by_key(|g| (g.file, g.game));
    ids
}

pub const CHOSEN_PREFETCH_FILES: usize = 3;
pub const CHOSEN_PREFETCH_FILES_CUT: usize = 24;

pub struct ChosenGames {
    rx: std::sync::mpsc::Receiver<Result<ChosenFile>>,
    cur: VecDeque<(usize, ShardGame)>,
    cur_file: usize,
    last: Option<GameId>,
    thread: Option<std::thread::JoinHandle<()>>,
}

type ChosenFile = (usize, Vec<(usize, ShardGame)>);

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "non-text panic".into())
}

pub fn cut_game(game: &mut ShardGame, until: u32) {
    let keep = game.steps.partition_point(|s| s.raw_frame.frame < until).max(1);
    game.steps.truncate(keep);
    game.steps.shrink_to_fit();
}

impl ChosenGames {
    pub fn new(src: ShardSource, ids: &[GameId]) -> Self {
        Self::with_until(src, ids, None)
    }

    pub fn with_until(src: ShardSource, ids: &[GameId], until: Option<u32>) -> Self {
        let ids: HashSet<GameId> = ids.iter().copied().collect();
        let files: std::collections::BTreeSet<usize> = ids.iter().map(|g| g.file).collect();
        let (tx, rx) = std::sync::mpsc::sync_channel(if until.is_some() { CHOSEN_PREFETCH_FILES_CUT } else { CHOSEN_PREFETCH_FILES });
        let thread = std::thread::spawn(move || {
            for k in files {
                let r = src.file_games(k).map(|games| {
                    let mut kept: Vec<(usize, ShardGame)> = games.into_iter().enumerate().filter(|(i, _)| ids.contains(&GameId { file: k, game: *i })).collect();
                    if let Some(f) = until {
                        kept.iter_mut().for_each(|(_, g)| cut_game(g, f));
                    }
                    (k, kept)
                });
                let failed = r.is_err();
                if tx.send(r).is_err() || failed {
                    return;
                }
            }
        });
        Self { rx, cur: VecDeque::new(), cur_file: 0, last: None, thread: Some(thread) }
    }
}

impl StepSource for ChosenGames {
    fn next_game(&mut self) -> Result<Option<ShardGame>> {
        while self.cur.is_empty() {
            let Ok(r) = self.rx.recv() else {
                if let Some(t) = self.thread.take() {
                    t.join().map_err(|p| anyhow::anyhow!("the chosen-games read-ahead thread panicked: {}", panic_text(&*p)))?;
                }
                return Ok(None);
            };
            let (k, games) = r?;
            self.cur_file = k;
            self.cur = games.into();
        }
        let (i, g) = self.cur.pop_front().expect("a game");
        self.last = Some(GameId { file: self.cur_file, game: i });
        Ok(Some(g))
    }

    fn last_id(&self) -> Option<GameId> {
        self.last
    }
}

pub fn replay_z(game: &ShardGame) -> Strategy {
    let table = action_table();
    let build: Vec<u16> = game
        .steps
        .iter()
        .map(|s| s.action_label.action)
        .filter(|&a| table.actions.get(a as usize).is_some_and(|x| matches!(x.command.as_str(), "Build" | "BuildAddon" | "Train" | "Research" | "Upgrade")))
        .take(20)
        .collect();
    let d = dims();
    let me = game.game_info.self_player;
    let mut seen: HashSet<u32> = HashSet::new();
    let mut counts = vec![0f32; d.z_units()];
    for s in &game.steps {
        for u in s.raw_frame.units.iter().filter(|u| u.owner == me) {
            if let Some(i) = d.terran_types.iter().position(|&t| t == u.unit_type) {
                if seen.insert(u.id) {
                    counts[i] += 1.0;
                }
            }
        }
    }
    let total: f32 = counts.iter().sum();
    if total > 0.0 {
        counts.iter_mut().for_each(|c| *c /= total);
    }
    Strategy { build, units: counts }
}

pub use uranium_env::features::label::{UNKNOWN_REASON, acting_ids, host_label, host_label_check};

#[derive(Debug, Default)]
pub struct LabelStats {
    pub fails: std::collections::BTreeMap<(String, &'static str), usize>,
    pub totals: std::collections::BTreeMap<String, usize>,
    pub cap_detail: std::collections::BTreeMap<String, usize>,
    pub type_detail: std::collections::BTreeMap<String, usize>,
    pub cause_detail: std::collections::BTreeMap<String, usize>,
    pub per_game: Vec<(String, String, usize, usize, usize, usize)>,
}

fn age_bucket(frames: u32) -> &'static str {
    match frames {
        0..=8 => "<=8f",
        9..=24 => "<=24f",
        25..=96 => "<=96f",
        _ => ">96f",
    }
}

fn loc_cause(w: &uranium_env::features::world::World, o: &Observation, a: &uranium_env::actions::Action, tile: Option<(u16, u16)>) -> String {
    use uranium_env::features::world::top_left;
    let Some((x, y)) = tile.map(|(x, y)| (x as i32, y as i32)) else { return "no_tile".into() };
    let tt = uranium_env::types::TypeTables::get();
    let lc = uranium_env::actions::LOC_CLASSES.iter().position(|c| *c == a.loc_class).unwrap();
    let ok_at = |tx: i32, ty: i32| tx >= 0 && ty >= 0 && (tx as usize) < MAP && (ty as usize) < MAP && o.masks.loc_class_mask[lc * CELLS + ty as usize * MAP + tx as usize];
    let mut causes: Vec<String> = vec![];
    let built = a.unit_type.and_then(|u| tt.unit(u));
    if a.loc_class == "refinery_slot" {
        let geyser = w.entities.iter().find(|e| e.unit.unit_type == w.ids.geyser && w.ty(&e.unit).is_some_and(|t| top_left(&e.unit, t) == (x, y)));
        let refinery = w.frame.units.iter().find(|u| tt.unit(u.unit_type).is_some_and(|t| t.name.ends_with("_Refinery") || t.name.ends_with("_Assimilator") || t.name.ends_with("_Extractor")) && tt.unit(u.unit_type).is_some_and(|t| top_left(u, t) == (x, y)));
        causes.push(match (geyser, refinery) {
            (_, Some(r)) if r.owner == w.mem.self_player => format!("own_refinery_here(completed={})", r.has(uranium_proto::flags::COMPLETED)),
            (_, Some(_)) => "enemy_refinery_here".into(),
            (Some(g), None) if g.remembered => format!("geyser_out_of_vision(explored={})", w.frame.explored.get(x as u16, y as u16)),
            (Some(_), None) => "geyser_visible".into(),
            (None, None) => "no_geyser_at_tile".into(),
        });
    } else {
        let (fw, fh) = match a.loc_class.as_str() {
            "buildable_4x3" | "landable_4x3" | "cc_slot" => (4, 3),
            "buildable_3x2" => (3, 2),
            "buildable_2x2" => (2, 2),
            _ => (1, 1),
        };
        for ty in y..y + fh {
            for tx in x..x + fw {
                if !w.map.in_map(tx, ty) {
                    causes.push("out_of_map".into());
                    continue;
                }
                let i = ty as usize * MAP + tx as usize;
                if !w.map.buildable[i] {
                    causes.push("unbuildable".into());
                }
                if !w.frame.explored.get(tx as u16, ty as u16) {
                    causes.push("unexplored".into());
                }
                if w.frame.creep.get(tx as u16, ty as u16) {
                    causes.push("creep".into());
                }
                if w.blocked[i] {
                    let on = w.entities.iter().find(|e| {
                        w.ty(&e.unit).is_some_and(|t| (t.is_building() || t.is_resource()) && !e.unit.has(uranium_proto::flags::LIFTED) && {
                            let (bx, by) = top_left(&e.unit, t);
                            tx >= bx && tx < bx + t.tile_width && ty >= by && ty < by + t.tile_height
                        })
                    });
                    causes.push(match on {
                        Some(e) => {
                            let t = w.ty(&e.unit).unwrap();
                            let same = built.is_some_and(|b| b.id as u16 == e.unit.unit_type);
                            let owner = if e.owner == OWNER_SELF { "own" } else if e.owner == OWNER_NEUTRAL { "neutral" } else if e.remembered { "remembered_enemy" } else { "enemy" };
                            if same && top_left(&e.unit, t) == (x, y) {
                                format!("same_building_same_tile(completed={})", e.unit.has(uranium_proto::flags::COMPLETED))
                            } else if t.is_resource() {
                                "blocked_by_resource".into()
                            } else {
                                format!("blocked_by_{owner}_building")
                            }
                        }
                        None => "blocked_by_remembered_resource".into(),
                    });
                }
            }
        }
        if causes.is_empty() && a.loc_class == "cc_slot" {
            causes.push("near_resource".into());
        }
        if causes.is_empty() {
            causes.push("mask_other".into());
        }
    }
    causes.sort();
    causes.dedup();
    let shift = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)].into_iter().find(|&(dx, dy)| ok_at(x + dx, y + dy));
    let shift = shift.map_or(String::new(), |(dx, dy)| format!(" (passes at {dx:+},{dy:+})"));
    format!("{}{shift}", causes.join("+"))
}

fn prereq_fail(a: &uranium_env::actions::Action, w: &uranium_env::features::world::World) -> Option<&'static str> {
    let t = uranium_env::types::TypeTables::get();
    let me = &w.frame.self_state;
    let afford = |m: i32, g: i32| me.minerals >= m && me.gas >= g;
    match a.command.as_str() {
        "Train" => {
            let u = t.unit(a.unit_type?)?;
            let supply = uranium_env::features::masks::train_supply_ok(w, a);
            if !afford(u.minerals, u.gas) {
                Some("afford")
            } else if !supply {
                Some("supply")
            } else {
                None
            }
        }
        "Research" => {
            let tech = a.tech?;
            let r = &t.techs[tech as usize];
            if me.tech & (1u64 << tech) != 0 {
                Some("researched")
            } else if me.researching & (1u64 << tech) != 0 {
                Some("researching")
            } else if !afford(r.minerals, r.gas) {
                Some("afford")
            } else {
                None
            }
        }
        "Upgrade" => {
            let up = a.upgrade?;
            let u = &t.upgrades[up as usize];
            let level = me.upgrades.get(up as usize).copied().unwrap_or(0) as i32;
            if level >= u.max_repeats {
                Some("max_level")
            } else if me.upgrading & (1u64 << up) != 0 {
                Some("upgrading")
            } else if !afford(u.minerals(level + 1), u.gas(level + 1)) {
                Some("afford")
            } else {
                None
            }
        }
        "Build" => {
            let u = t.unit(a.unit_type?)?;
            if !afford(u.minerals, u.gas) { Some("afford") } else { None }
        }
        _ => None,
    }
}

pub fn label_check(game: &ShardGame, st: &mut LabelStats, maps: &MapStore) -> Result<()> {
    let mut walker = LabelWalker::new(&game.game_info, family_index(&game.meta.map_family), maps.get(&game.meta.map_hash)?.as_deref(), game.meta.enemy_random)?;
    let k = CAPABILITIES.len();
    let mut prev: Option<&uranium_proto::RawFrame> = None;
    let mut seen: HashMap<u32, (u32, u8, u16)> = HashMap::new();
    let tt = uranium_env::types::TypeTables::get();
    let type_name = |t: u16| tt.unit(t).map_or("?", |t| t.name.as_str());
    let mut g = (game.meta.src.clone(), format!("{} {} map {} {}x{}", game.meta.replay_sha256.chars().take(12).collect::<String>(), game.meta.matchup, game.meta.map_hash.chars().take(8).collect::<String>(), game.game_info.map_size_tiles.0, game.game_info.map_size_tiles.1), 0usize, 0usize, 0usize, 0usize);
    for (si, s) in game.steps.iter().enumerate() {
        let (o, label, why) = walker.step_with_memory(&s.raw_frame, &s.action_label, None, s.memory_action.as_deref());
        let a = action_table().actions.get(label.action as usize).filter(|_| !s.action_label.is_unknown());
        let name = if s.action_label.is_unknown() { "Unknown".to_string() } else { a.map_or_else(|| format!("#{}", label.action), |a| a.name.clone()) };
        *st.totals.entry(name.clone()).or_default() += 1;
        if let Some(a) = a.filter(|a| a.uses_select && matches!(why, None | Some("select_cap"))) {
            let c = CAPABILITIES.iter().position(|c| *c == a.select_cap).unwrap();
            let mut types: Vec<&str> = s
                .action_label
                .selected
                .iter()
                .filter_map(|id| o.entity_ids.iter().position(|e| e == id))
                .filter(|&i| !o.masks.unit_cap[i * k + c])
                .filter_map(|i| s.raw_frame.units.iter().find(|u| u.id == o.entity_ids[i]))
                .filter_map(|u| uranium_env::types::TypeTables::get().unit(u.unit_type).map(|t| t.name.as_str()))
                .collect();
            types.sort_unstable();
            types.dedup();
            if !types.is_empty() {
                let kind = if why.is_some() { "void" } else { "kept" };
                *st.cap_detail.entry(format!("{name} <- {} ({kind})", types.join("+"))).or_default() += 1;
            }
        }
        if let Some(w @ ("target_unit_cap" | "target_unit_missing")) = why {
            let target = s.action_label.target_unit.and_then(|id| s.raw_frame.units.iter().find(|u| u.id == id));
            let me = game.game_info.self_player;
            let desc = target.map_or_else(
                || "not_in_frame".to_string(),
                |u| {
                    let owner = if u.owner == me { "self" } else if game.game_info.players.iter().any(|p| p.id == u.owner) { "enemy" } else { "neutral" };
                    let ty = uranium_env::types::TypeTables::get().unit(u.unit_type).map_or("?", |t| t.name.as_str());
                    format!("{owner} {ty}")
                },
            );
            *st.type_detail.entry(format!("{name} [{w}] -> {desc}")).or_default() += 1;
        }
        if let (Some(_), Some(w @ ("target_unit_missing" | "target_unit_cap"))) = (a, why) {
            let id = s.action_label.target_unit;
            let in_frame = id.and_then(|id| s.raw_frame.units.iter().find(|u| u.id == id));
            let cause = match (id, in_frame) {
                (None, _) => "no_target_id".to_string(),
                (Some(_), Some(u)) if w == "target_unit_missing" => format!("in_frame_not_entity({})", type_name(u.unit_type)),
                (Some(_), Some(u)) => {
                    let done = u.has(uranium_proto::flags::COMPLETED);
                    let loaded = u.has(uranium_proto::flags::LOADED) || u.loaded_in.is_some();
                    format!("present_ineligible({}, completed={done}, loaded={loaded})", type_name(u.unit_type))
                }
                (Some(id), None) => {
                    let later = game.steps[si + 1..].iter().take(3).any(|n| n.raw_frame.units.iter().any(|u| u.id == id));
                    match seen.get(&id) {
                        Some(&(f, _, t)) => format!("seen_before({}, {}{})", type_name(t), age_bucket(s.raw_frame.frame.saturating_sub(f)), if later { ", back later" } else { "" }),
                        None if later => "first_seen_later".to_string(),
                        None => "never_seen".to_string(),
                    }
                }
            };
            *st.cause_detail.entry(format!("{name} [{w}] -> {cause}")).or_default() += 1;
        }
        if let (Some(a), Some("target_loc")) = (a, why) {
            let w = uranium_env::features::world::World::new(&s.raw_frame, &walker.mem, &walker.map);
            let ty = a.unit_type.map_or("-", type_name);
            let me = game.game_info.self_player;
            let built = s.action_label.target_tile.zip(a.unit_type).is_some_and(|((x, y), ut)| {
                game.steps[si + 1..].iter().take(200).any(|n| {
                    n.raw_frame.units.iter().any(|u| {
                        u.owner == me && u.unit_type == ut && tt.unit(ut).is_some_and(|t| uranium_env::features::world::top_left(u, t) == (x as i32, y as i32))
                    })
                })
            });
            let cause = loc_cause(&w, &o, a, s.action_label.target_tile);
            *st.cause_detail.entry(format!("{name} [target_loc] {ty} -> {cause}{}", if built { " [built there]" } else { " [not built there]" })).or_default() += 1;
        }
        if let (Some(a), Some("type_mask")) = (a, why) {
            let world = |f| uranium_env::features::world::World::new(f, &walker.mem, &walker.map);
            let cond = prereq_fail(a, &world(&s.raw_frame)).unwrap_or("producer_or_required");
            let age = match prev.and_then(|p| prereq_fail(a, &world(p))) {
                Some(c) if c == cond => "old",
                _ if cond == "producer_or_required" => "-",
                _ => "new",
            };
            *st.type_detail.entry(format!("{name} <- {cond} ({age})")).or_default() += 1;
        }
        let unknown = why == Some(UNKNOWN_REASON);
        g.2 += !unknown as usize;
        if let Some(w) = why {
            g.3 += !unknown as usize;
            g.4 += (w == "target_unit_missing") as usize;
            g.5 += (w == "target_loc") as usize;
            *st.fails.entry((name, w)).or_default() += 1;
        }
        prev = Some(&s.raw_frame);
        for u in &s.raw_frame.units {
            seen.insert(u.id, (s.raw_frame.frame, u.owner, u.unit_type));
        }
    }
    st.per_game.push(g);
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn state(&self) -> u64 {
        self.0
    }
    pub fn from_state(s: u64) -> Self {
        Self(s)
    }
}

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

#[cfg(test)]
mod chosen_tests {
    use super::*;

    #[test]
    fn chosen_games_read_ahead_panic_is_an_error_050_r9() {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<ChosenFile>>(1);
        let thread = std::thread::spawn(move || {
            let _tx = tx;
            panic!("decode failed");
        });
        let mut c = ChosenGames { rx, cur: VecDeque::new(), cur_file: 0, last: None, thread: Some(thread) };
        let e = c.next_game().expect_err("a panic is an error");
        assert!(format!("{e:#}").contains("decode failed"));
    }
}
