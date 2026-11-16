use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use uranium_replay::shard::{ShardGame, read};
use super::{StepSource, GameId, SourcePos, HasMeta, Rng};

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

pub(super) fn key_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.split('/').cmp(b.split('/'))
}

pub(super) fn key_hash_text(key: &str) -> String {
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
