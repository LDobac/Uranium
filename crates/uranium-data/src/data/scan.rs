use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use anyhow::{Context, Result};
use uranium_replay::shard::{GameMeta, read};
use super::source::{ShardSource, key_cmp, key_hash_text};
use super::HasMeta;
use super::maps::MapStore;

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
    uranium_store::elog!("{}", serde_json::json!({ "map_precomputes": "ok (B-2)", "games": games.len(), "maps": n_maps, "secs": started.elapsed().as_secs() }));
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
            uranium_store::elog!("[scan] {} files from the cache {}", c.files.len(), path.as_ref().unwrap().display());
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
                    uranium_store::elog!("[scan] could not write the scan cache {} ({e:#}); the next start scans again", p.display());
                }
            }
            scan
        }
    };
    let index: HashMap<&PathBuf, usize> = sorted.iter().enumerate().map(|(i, f)| (f, i)).collect();
    Ok(src.files().iter().map(|f| by_sorted[index[f]].clone()).collect())
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
