use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use anyhow::{Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uranium_env::maps::{MapFamilies, clean_title};
use super::{MAP_EXCLUDE_TOML, hex};

#[derive(Debug, Default, Clone)]
pub struct Duplicates {
    pub global: Vec<usize>,
    pub first: HashMap<String, usize>,
    pub repeats: usize,
}

impl Duplicates {
    pub fn find(all: &[PathBuf], mine: &[usize], jobs: usize) -> Self {
        let sizes: Vec<Option<u64>> = all.iter().map(|f| std::fs::metadata(f).ok().map(|m| m.len())).collect();
        let mut by_size: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, s) in sizes.iter().enumerate() {
            if let Some(s) = s {
                by_size.entry(*s).or_default().push(i);
            }
        }
        let mut todo: Vec<usize> = mine.iter().filter_map(|&i| sizes[i]).filter_map(|s| by_size.get(&s)).filter(|g| g.len() > 1).flatten().copied().collect();
        todo.sort_unstable();
        todo.dedup();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let found = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..jobs.max(1) {
                s.spawn(|| {
                    let mut got = Vec::new();
                    while let Some(&i) = todo.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                        if let Ok(b) = std::fs::read(&all[i]) {
                            got.push((hex(&Sha256::digest(&b)), i));
                        }
                    }
                    found.lock().unwrap().extend(got);
                });
            }
        });
        let found = found.into_inner().unwrap();
        let mut first = HashMap::new();
        for (sha, i) in &found {
            first.entry(sha.clone()).and_modify(|j: &mut usize| *j = (*j).min(*i)).or_insert(*i);
        }
        let mine_set: std::collections::HashSet<usize> = mine.iter().copied().collect();
        let repeats = found.iter().filter(|(sha, i)| mine_set.contains(i) && first[sha] != *i).count();
        Self { global: mine.to_vec(), first, repeats }
    }

    pub fn earlier_copy(&self, index: usize, sha: &str) -> Option<usize> {
        let me = self.global.get(index).copied().unwrap_or(index);
        self.first.get(sha).copied().filter(|&f| f < me)
    }
}

#[derive(Deserialize)]
struct Excluded {
    name: String,
    name_patterns: Vec<String>,
    chk_sha256: Vec<String>,
}

#[derive(Deserialize)]
struct ExcludeFile {
    map: Vec<Excluded>,
}

pub struct MapExcludes(Vec<Excluded>);

impl MapExcludes {
    pub fn load(families: &MapFamilies) -> Result<Self> {
        let list = toml::from_str::<ExcludeFile>(MAP_EXCLUDE_TOML).context("map_exclude.toml")?.map;
        for m in &list {
            for p in &m.name_patterns {
                anyhow::ensure!(families.classify(p, "") == "other", "map_exclude.toml: {} matches a target map family (040-R16)", m.name);
            }
        }
        Ok(Self(list))
    }

    pub fn matches(&self, map_name: &str, chk_sha256: &str) -> Option<&str> {
        let title = clean_title(map_name);
        self.0.iter().find(|m| m.chk_sha256.iter().any(|h| h == chk_sha256) || m.name_patterns.iter().any(|p| title.contains(&p.to_lowercase()))).map(|m| m.name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_desync_maps_are_excluded_040_r16() {
        let f = MapFamilies::load().unwrap();
        let x = MapExcludes::load(&f).unwrap();
        assert_eq!(x.matches("\u{7}Attitude SE \u{5}2.1", ""), Some("Attitude SE 2.1"));
        assert_eq!(x.matches("\u{1}C\u{6}olorless \u{2}F\u{6}ate \u{5}1.1", ""), Some("Colorless Fate 1.1"));
        assert_eq!(x.matches("Attitude SE 2.2", ""), None);
        assert_eq!(x.matches("\u{7}투혼 \u{5}1.4", ""), None);
        assert_eq!(x.matches("Octagon SE 2.0", ""), None);
    }
}
