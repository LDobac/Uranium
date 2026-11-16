use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use anyhow::Result;
use uranium_env::features::*;
use super::scan::GameSummary;

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
