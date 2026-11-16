use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use anyhow::{Context, Result};
use crate::shard::{GameMeta, read};
use super::source::{ShardSource, key_hash_text};
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
const SCAN_CACHE_VERSION: u32 = 2;
const SCAN_SHARD_SCHEMA: u32 = 6;

pub fn scan_cache_dir() -> Option<PathBuf> {
    std::env::var_os(CACHE_DIR_ENV).filter(|v| !v.is_empty()).map(|v| PathBuf::from(v).join("scans"))
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScanEntry {
    version: u32,
    shard_schema: u32,
    key: String,
    id: String,
    games: Vec<GameSummary>,
}

fn entry_path(dir: &Path, key: &str, id: &str) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update(b"\t");
    h.update(id.as_bytes());
    let hex: String = h.finalize().iter().take(16).map(|b| format!("{b:02x}")).collect();
    dir.join(format!("file-{hex}.json"))
}

fn read_entry(path: &Path, key: &str, id: &str) -> Option<Vec<GameSummary>> {
    let e: ScanEntry = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    (e.version == SCAN_CACHE_VERSION && e.shard_schema == SCAN_SHARD_SCHEMA && e.key == key && e.id == id).then_some(e.games)
}

fn write_entry(path: &Path, key: &str, id: &str, games: &[GameSummary]) -> Result<()> {
    let e = ScanEntry { version: SCAN_CACHE_VERSION, shard_schema: SCAN_SHARD_SCHEMA, key: key.to_string(), id: id.to_string(), games: games.to_vec() };
    let tmp = path.with_extension(format!("json.tmp-{}", std::process::id()));
    std::fs::write(&tmp, serde_json::to_vec(&e)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn scan_cached(src: &ShardSource) -> Result<Vec<Vec<GameSummary>>> {
    scan_cached_in(src, scan_cache_dir().as_deref())
}

pub fn scan_cached_in(src: &ShardSource, cache: Option<&Path>) -> Result<Vec<Vec<GameSummary>>> {
    let (files, keys) = (src.files(), src.keys());
    let Some(dir) = cache else { return scan_files(files) };
    let ids = file_ids(src.dirs(), files, true)?;
    let paths: Vec<PathBuf> = keys.iter().zip(&ids).map(|(k, id)| entry_path(dir, k, id)).collect();
    let mut per_file: Vec<Option<Vec<GameSummary>>> = (0..files.len()).map(|i| read_entry(&paths[i], &keys[i], &ids[i])).collect();
    let missing: Vec<usize> = (0..files.len()).filter(|&i| per_file[i].is_none()).collect();
    uranium_store::elog!("[scan] {} files from the cache {}, {} to scan", files.len() - missing.len(), dir.display(), missing.len());
    if !missing.is_empty() {
        let scanned = scan_files(&missing.iter().map(|&i| files[i].clone()).collect::<Vec<_>>())?;
        let mut written = std::fs::create_dir_all(dir).map_err(anyhow::Error::from);
        for (&i, games) in missing.iter().zip(scanned) {
            if written.is_ok() {
                written = write_entry(&paths[i], &keys[i], &ids[i], &games);
            }
            per_file[i] = Some(games);
        }
        if let Err(e) = written {
            uranium_store::elog!("[scan] could not write the scan cache under {} ({e:#}); the next start scans those files again", dir.display());
        }
    }
    Ok(per_file.into_iter().map(|v| v.expect("every file scanned")).collect())
}

pub fn dataset_fingerprint(src: &ShardSource) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for (key, id) in src.keys().iter().zip(file_ids(src.dirs(), src.files(), false)?) {
        h.update(key_hash_text(key).as_bytes());
        h.update(b"\t");
        h.update(id.as_bytes());
        h.update(b"\n");
    }
    let hex: String = h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect();
    Ok(format!("{}:{hex}", src.files().len()))
}

fn file_ids(dirs: &[PathBuf], files: &[PathBuf], local: bool) -> Result<Vec<String>> {
    let mut manifests: HashMap<PathBuf, Option<uranium_store::storage::Manifest>> = HashMap::new();
    let mut ids = Vec::with_capacity(files.len());
    for f in files {
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
        ids.push(match entry {
            Some(sha) => sha,
            None => {
                let m = std::fs::metadata(f).with_context(|| f.display().to_string())?;
                let mtime = if local { m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos()) } else { 0 };
                if local { format!("bytes:{}:mtime:{mtime}", m.len()) } else { format!("bytes:{}", m.len()) }
            }
        });
    }
    Ok(ids)
}
