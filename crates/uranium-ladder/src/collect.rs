//! `collect` and `status` (043-R4..R9): leaderboard → tracked players' profile `replays[]` →
//! match info → replay download, with dedup, resume and the daily budget.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::api::{Api, Http};
use crate::env::Settings;
use crate::limit::Daily;
use crate::schema::{self, LeaderboardEntry, LeaderboardList, LeaderboardPage, MatchInfo, Profile};

/// 040-R14 / 043-R7 source tag.
pub const SRC: &str = "scr_ladder";
/// 043-R9: stop after this many failures in a row.
const MAX_CONSECUTIVE_FAILURES: u32 = 20;

/// 043-R5 collection plan (`config/ladder/<plan>.toml`).
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub gateways: Vec<u32>,
    pub gamemode: u32,
    /// Lowest leaderboard bucket of tracked players. Unset until the rank letter ↔ bucket
    /// mapping is confirmed (043-R5 TBD(M4)); `collect` refuses to run without it.
    pub min_bucket: Option<i64>,
    #[serde(default = "default_page_size")]
    pub page_size: u32,
}

fn default_page_size() -> u32 {
    100
}

impl Plan {
    pub fn load(path: &Path) -> Result<Self> {
        toml::from_str(&std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?)
            .with_context(|| format!("{}", path.display()))
    }
}

/// Where the collector reads and writes (043-R7, R8).
pub struct Paths {
    /// `datasets/replays/ladder`: collect.jsonl, daily.json, raw_index.json
    pub ladder: PathBuf,
    /// `datasets/replays/raw`: every source, for the cross-source SHA-256 dedup (043-R6)
    pub raw_root: PathBuf,
}

impl Paths {
    pub fn from_settings(settings: &Settings) -> Result<Self> {
        let ladder = settings.ladder_dir()?;
        let raw_root = ladder.parent().context("ladder dir has no parent")?.join("raw");
        Ok(Self { ladder, raw_root })
    }

    fn records(&self) -> PathBuf {
        self.ladder.join("collect.jsonl")
    }

    fn raw_src(&self) -> PathBuf {
        self.raw_root.join(SRC)
    }
}

pub struct Opts {
    /// Most downloads in this run.
    pub limit: Option<u32>,
    /// List candidate matches without fetching match info or replays.
    pub dry_run: bool,
    /// Instead of walking the leaderboard, fetch again the matches whose latest record is
    /// `failed` with a reason starting with this prefix (043-R6, R9).
    pub retry_failed: Option<String>,
}

/// 043-R8 record, one line of collect.jsonl per match.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Record {
    #[serde(rename = "match")]
    pub link: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub md5: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub season_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub map_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub map_md5: Option<String>,
    #[serde(default)]
    pub players: Vec<RecordPlayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create_time: Option<u64>,
    pub collected_at: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordPlayer {
    pub toon: String,
    pub gateway_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub race: Option<String>,
    /// MMR and leaderboard bucket; empty when the match info has no `score` for this player.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rating: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bucket: Option<i64>,
    /// This player's client report per toon (`Win`, `Loss`, `Disconnect`, `Undecided`, …).
    pub reports: BTreeMap<String, String>,
}

#[derive(Default, Debug, PartialEq)]
pub struct Summary {
    pub ok: u32,
    pub skipped: u32,
    pub failed: u32,
}

struct Recorder {
    file: std::fs::File,
    seen: HashSet<String>,
    summary: Summary,
}

impl Recorder {
    fn open(path: &Path) -> Result<Self> {
        let mut seen = HashSet::new();
        if let Ok(text) = std::fs::read_to_string(path) {
            for l in text.lines() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
                    if let Some(m) = v["match"].as_str() {
                        seen.insert(m.to_string());
                    }
                }
            }
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file, seen, summary: Summary::default() })
    }

    fn write(&mut self, r: Record) -> Result<()> {
        match r.status.as_str() {
            "ok" => self.summary.ok += 1,
            "skipped" => self.summary.skipped += 1,
            _ => self.summary.failed += 1,
        }
        self.seen.insert(r.link.clone());
        writeln!(self.file, "{}", serde_json::to_string(&r)?)?;
        self.file.flush()?;
        Ok(())
    }
}

struct Failures(u32);

impl Failures {
    fn fail(&mut self, what: &str) -> Result<()> {
        self.0 += 1;
        if self.0 >= MAX_CONSECUTIVE_FAILURES {
            bail!("{} failures in a row (last: {what}); stopping (043-R9)", self.0);
        }
        Ok(())
    }
}

pub fn run(settings: &Settings, paths: &Paths, plan: &Plan, opt: &Opts) -> Result<Summary> {
    let Some(min_bucket) = plan.min_bucket else {
        bail!("plan has no min_bucket: the rank letter ↔ bucket mapping is not confirmed yet (043-R5)");
    };
    let mut rec = Recorder::open(&paths.records())?;
    let mut daily = Daily::load_today(&paths.ladder.join("daily.json"), settings.daily_max)?;
    let mut raw_shas = if opt.dry_run { HashSet::new() } else { raw_index(&paths.raw_root, &paths.ladder.join("raw_index.json"))? };
    let mut http = Http::new(settings.rps);
    if let Some(prefix) = &opt.retry_failed {
        let targets = retry_targets(&paths.records(), prefix)?;
        println!("retry: {} matches whose latest record failed with this reason", targets.len());
        if opt.dry_run || targets.is_empty() {
            return Ok(rec.summary);
        }
        let api = Api::connect(&mut http, settings)?;
        let mut failures = Failures(0);
        let mut downloads = 0u32;
        let n = targets.len();
        for (i, (link, create_time)) in targets.into_iter().enumerate() {
            if daily.remaining() == 0 {
                println!("daily download budget ({}) reached (043-R4)", settings.daily_max);
                break;
            }
            if opt.limit.is_some_and(|l| downloads >= l) {
                break;
            }
            let (r, downloaded) = fetch_match(&api, &mut http, paths, &mut raw_shas, &link, create_time)?;
            if downloaded {
                downloads += 1;
                daily.record()?;
            }
            let failed = r.status == "failed";
            println!("[retry {}/{n}] {} {}", i + 1, r.status, r.reason.as_deref().or(r.sha256.as_deref().map(|s| &s[..12])).unwrap_or(""));
            rec.write(r)?;
            if failed {
                failures.fail("match")?;
            } else {
                failures.0 = 0;
            }
        }
        return Ok(rec.summary);
    }
    let api = Api::connect(&mut http, settings)?;

    let list: LeaderboardList = api.json(&mut http, schema::LEADERBOARD_LIST).map_err(|e| anyhow::anyhow!("leaderboard list: {e}"))?;
    let mut players: Vec<LeaderboardEntry> = Vec::new();
    for &gw in &plan.gateways {
        let lb = list.current(gw, plan.gamemode).with_context(|| format!("no leaderboard for gateway {gw}, gamemode {}", plan.gamemode))?;
        let before = players.len();
        let mut offset = 0;
        loop {
            let page: LeaderboardPage = api
                .json(&mut http, &schema::leaderboard_page(lb.id, offset, plan.page_size))
                .map_err(|e| anyhow::anyhow!("leaderboard {}: {e}", lb.id))?;
            let entries = page.entries().map_err(|e| anyhow::anyhow!("leaderboard {}: schema: {e}", lb.id))?;
            let tracked: Vec<_> = entries.iter().filter(|e| e.bucket >= min_bucket).cloned().collect();
            let last = tracked.is_empty() || entries.len() < plan.page_size as usize;
            players.extend(tracked);
            if last {
                break;
            }
            offset += plan.page_size;
        }
        println!("gateway {gw}: leaderboard {} season {}: {} tracked players", lb.id, lb.season_id, players.len() - before);
    }
    let mut uniq = HashSet::new();
    players.retain(|p| uniq.insert((p.toon.clone(), p.gateway_id)));

    let mut failures = Failures(0);
    let mut downloads = 0u32;
    let n = players.len();
    'players: for (i, p) in players.iter().enumerate() {
        let tag = format!("[{}/{n} gw{}]", i + 1, p.gateway_id);
        let profile: Profile = match api.json(&mut http, &schema::profile(&p.toon, p.gateway_id)) {
            Ok(v) => v,
            Err(e) => {
                println!("{tag} profile failed: {e}");
                failures.fail("profile")?;
                continue;
            }
        };
        // Placeholder times mark games whose replay may not exist yet; they are left for a
        // later run instead of being recorded.
        let links: Vec<_> = profile
            .replays
            .iter()
            .filter(|r| r.is_ladder_1v1() && r.create_time < schema::MAX_REAL_TIME && !rec.seen.contains(&r.link))
            .map(|r| (r.link.clone(), r.create_time))
            .collect();
        println!("{tag} {} new ladder 1v1 links", links.len());
        if opt.dry_run {
            continue;
        }
        for (link, create_time) in links {
            if rec.seen.contains(&link) {
                continue; // reached through the opponent's profile earlier in this run
            }
            if daily.remaining() == 0 {
                println!("daily download budget ({}) reached (043-R4)", settings.daily_max);
                break 'players;
            }
            if opt.limit.is_some_and(|l| downloads >= l) {
                break 'players;
            }
            let (r, downloaded) = fetch_match(&api, &mut http, paths, &mut raw_shas, &link, create_time)?;
            if downloaded {
                downloads += 1;
                daily.record()?;
            }
            let failed = r.status == "failed";
            println!("{tag} {} {}", r.status, r.reason.as_deref().or(r.sha256.as_deref().map(|s| &s[..12])).unwrap_or(""));
            rec.write(r)?;
            if failed {
                failures.fail("match")?;
            } else {
                failures.0 = 0;
            }
        }
    }
    Ok(rec.summary)
}

/// Latest record per match, in first-seen order (later lines replace earlier ones), and the
/// number of lines that did not parse as a record.
fn latest_records(text: &str) -> (Vec<Record>, usize) {
    let mut order: Vec<String> = Vec::new();
    let mut last: std::collections::HashMap<String, Record> = std::collections::HashMap::new();
    let mut bad = 0;
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(r) = serde_json::from_str::<Record>(l) else {
            bad += 1;
            continue;
        };
        if !last.contains_key(&r.link) {
            order.push(r.link.clone());
        }
        last.insert(r.link.clone(), r);
    }
    (order.into_iter().filter_map(|k| last.remove(&k)).collect(), bad)
}

/// `latest_records` of a file, with a stderr summary of skipped unparsable lines.
fn latest_records_of(path: &Path) -> Result<Vec<Record>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let (records, bad) = latest_records(&text);
    if bad > 0 {
        eprintln!("warning: {bad} unparsable lines in {} skipped", path.display());
    }
    Ok(records)
}

/// `--retry-failed`: matches whose latest record is `failed` with a reason starting with `prefix`.
fn retry_targets(records: &Path, prefix: &str) -> Result<Vec<(String, u64)>> {
    Ok(latest_records_of(records)?
        .into_iter()
        .filter(|r| r.status == "failed" && r.reason.as_deref().is_some_and(|x| x.starts_with(prefix)))
        .map(|r| (r.link, r.create_time.unwrap_or(0)))
        .collect())
}

/// Match info → download → verify → store. Returns the record and whether a download happened.
fn fetch_match(api: &Api, http: &mut Http, paths: &Paths, raw_shas: &mut HashSet<String>, link: &str, create_time: u64) -> Result<(Record, bool)> {
    let mut r = Record { link: link.to_string(), create_time: Some(create_time), collected_at: chrono::Local::now().to_rfc3339(), ..Default::default() };
    let info: MatchInfo = match api.json(http, &schema::match_info(link)) {
        Ok(v) => v,
        Err(e) => return Ok((failed(r, &e.to_string()), false)),
    };
    describe(&mut r, &info);
    if info.replays.is_empty() {
        r.status = "skipped".into();
        r.reason = Some("no replay url".into());
        return Ok((r, false));
    }
    let mut last_err = String::new();
    let mut fetched = false;
    for rep in &info.replays {
        let bytes = match http.get_ok(&rep.url) {
            Ok(b) => {
                fetched = true;
                b
            }
            Err(e) => {
                last_err = format!("download: {e}");
                continue;
            }
        };
        let md5 = hex(&md5::Md5::digest(&bytes));
        if !md5.eq_ignore_ascii_case(&rep.md5) {
            last_err = "download: md5 mismatch".into();
            continue;
        }
        let sha = hex(&Sha256::digest(&bytes));
        r.md5 = Some(md5);
        r.sha256 = Some(sha.clone());
        if raw_shas.contains(&sha) {
            r.status = "skipped".into();
            r.reason = Some("duplicate".into());
            return Ok((r, true));
        }
        // 043-R7: file name from the match id, never from player names.
        let name = format!("{}.rep", &hex(&Sha256::digest(link.as_bytes()))[..32]);
        let dir = paths.raw_src();
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(&name), &bytes)?;
        raw_shas.insert(sha);
        r.file = Some(name);
        r.status = "ok".into();
        return Ok((r, true));
    }
    Ok((failed(r, &last_err), fetched))
}

fn failed(mut r: Record, reason: &str) -> Record {
    r.status = "failed".into();
    r.reason = Some(reason.to_string());
    r
}

fn describe(r: &mut Record, info: &MatchInfo) {
    let first = info.players.values().next();
    r.map_name = first.and_then(|p| p.game_attr("map_name")).map(str::to_string);
    r.map_md5 = first.and_then(|p| p.game_attr("map_md5")).map(str::to_string);
    r.season_id = info.players.values().find_map(|p| p.score.as_ref()?.season_id);
    r.winner = info.agreed_winner().map(str::to_string);
    r.players = info
        .players
        .values()
        .map(|p| RecordPlayer {
            toon: p.name.clone(),
            gateway_id: p.gateway_id,
            race: p.race().map(str::to_string),
            rating: p.score.as_ref().map(|s| s.base),
            bucket: p.score.as_ref().map(|s| s.bucket_new),
            reports: p.game_result.iter().filter(|(t, _)| !t.is_empty()).map(|(t, g)| (t.clone(), g.result.clone())).collect(),
        })
        .collect();
}

#[derive(Serialize, Deserialize)]
struct IndexEntry {
    len: u64,
    mtime: u64,
    sha256: String,
}

/// 043-R6: SHA-256 of every `.rep` under `raw/`, cached by (path, size, mtime) so a daily run
/// only hashes new files.
fn raw_index(raw_root: &Path, cache: &Path) -> Result<HashSet<String>> {
    let mut old: BTreeMap<String, IndexEntry> =
        std::fs::read_to_string(cache).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    let mut files = Vec::new();
    walk(raw_root, &mut files);
    let mut new = BTreeMap::new();
    let total = files.len();
    let mut hashed = 0usize;
    println!("raw index: {total} replays under raw/");
    for f in files {
        let meta = std::fs::metadata(&f)?;
        let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let key = f.strip_prefix(raw_root).unwrap_or(&f).to_string_lossy().replace('\\', "/");
        let e = match old.remove(&key) {
            Some(e) if e.len == meta.len() && e.mtime == mtime => e,
            _ => {
                hashed += 1;
                if hashed % 5000 == 0 {
                    println!("raw index: hashed {hashed} new files ({} of {total} scanned)", new.len());
                }
                IndexEntry { len: meta.len(), mtime, sha256: hex(&Sha256::digest(std::fs::read(&f)?)) }
            }
        };
        new.insert(key, e);
    }
    if let Some(dir) = cache.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(cache, serde_json::to_string(&new)?)?;
    println!("raw index: {total} replays, {hashed} newly hashed");
    Ok(new.into_values().map(|e| e.sha256).collect())
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("rep")) {
            out.push(p);
        }
    }
}

/// `status`: counts per status/reason, and per matchup and map for stored replays (no names).
pub fn status_markdown(records: &Path) -> Result<String> {
    let mut by_status: BTreeMap<String, u32> = BTreeMap::new();
    let mut by_matchup: BTreeMap<String, u32> = BTreeMap::new();
    let mut by_map: BTreeMap<String, u32> = BTreeMap::new();
    // A retried match counts once, by its latest record.
    for r in latest_records_of(records)? {
        let reason = r.reason.as_deref().map(|s| s.split(':').next().unwrap_or(s).to_string());
        *by_status.entry(match reason {
            Some(x) => format!("{} ({x})", r.status),
            None => r.status.clone(),
        }).or_default() += 1;
        if r.status == "ok" {
            *by_matchup.entry(matchup(&r)).or_default() += 1;
            *by_map.entry(r.map_name.clone().unwrap_or_else(|| "?".into())).or_default() += 1;
        }
    }
    let mut md = format!("# {SRC} collection status\n\n| status | games |\n|---|---|\n");
    for (k, v) in &by_status {
        md += &format!("| {k} | {v} |\n");
    }
    md += "\n| matchup | stored |\n|---|---|\n";
    for (k, v) in &by_matchup {
        md += &format!("| {k} | {v} |\n");
    }
    md += "\n| map | stored |\n|---|---|\n";
    for (k, v) in &by_map {
        md += &format!("| {k} | {v} |\n");
    }
    Ok(md)
}

fn matchup(r: &Record) -> String {
    let mut races: Vec<char> = r
        .players
        .iter()
        .map(|p| p.race.as_deref().and_then(|s| s.chars().next()).map_or('?', |c| c.to_ascii_uppercase()))
        .collect();
    // Terran first, then P, Z (TvP, TvZ, PvZ, ...).
    races.sort_by_key(|c| match c {
        'T' => 0,
        'P' => 1,
        'Z' => 2,
        _ => 3,
    });
    races.iter().map(char::to_string).collect::<Vec<_>>().join("v")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    type Routes = Arc<Mutex<BTreeMap<String, (u16, Vec<u8>)>>>;

    /// Minimal HTTP/1.1 server answering GETs from `routes`; counts file downloads.
    fn serve(listener: TcpListener, routes: Routes, downloads: Arc<AtomicU32>) {
        serve_logged(listener, routes, downloads, Arc::default());
    }

    /// `serve` that also records when each request line arrived (043-AC4, server side).
    fn serve_logged(listener: TcpListener, routes: Routes, downloads: Arc<AtomicU32>, arrivals: Arc<Mutex<Vec<std::time::Instant>>>) {
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                arrivals.lock().unwrap().push(std::time::Instant::now());
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).is_err() || h.trim().is_empty() {
                        break;
                    }
                }
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                if path.starts_with("/files/") {
                    downloads.fetch_add(1, Ordering::SeqCst);
                }
                let (status, body) = routes.lock().unwrap().get(&path).cloned().unwrap_or((404, b"not found".to_vec()));
                let mut s = stream;
                let _ = write!(s, "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            }
        });
    }

    fn player(toon: &str, race: &str, results: &[(&str, &str)]) -> serde_json::Value {
        let gr: serde_json::Map<String, serde_json::Value> =
            results.iter().map(|(t, r)| (t.to_string(), json!({"result": r, "attributes": {}}))).collect();
        json!({"aurora_id": 1, "name": toon, "gateway_id": 30,
               "game_info": {"attributes": {"map_name": "Test Map", "map_md5": "00"}},
               "game_result": gr, "info_attributes": {"race": race},
               "score": {"base": 2100, "bucket_new": 6, "season_id": 20}})
    }

    fn replay(link: &str, humans: &str, time: u64) -> serde_json::Value {
        json!({"attributes": {"replay_humans": humans}, "create_time": time, "link": link})
    }

    /// 043-AC2: leaderboard → profile → match → download, in-run and cross-run dedup,
    /// cross-source SHA-256 dedup, schema mismatch recorded as failed.
    #[test]
    fn r043_ac2_collect_flow_dedup_resume_schema() {
        let dir = std::env::temp_dir().join(format!("uranium-ladder-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = Paths { ladder: dir.join("ladder"), raw_root: dir.join("raw") };
        // A replay another source already holds (043-R6).
        let dup = b"seRS fake replay already in personal".to_vec();
        std::fs::create_dir_all(paths.raw_root.join("personal")).unwrap();
        std::fs::write(paths.raw_root.join("personal").join("x.rep"), &dup).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let new_rep = b"seRS fake ladder replay 1".to_vec();
        let md5_of = |b: &[u8]| hex(&md5::Md5::digest(b));
        let file = |n: &str| format!("http://127.0.0.1:{port}/files/{n}");
        let pa = player("toonA", "Terran", &[("toonA", "Win"), ("toonB", "Loss"), ("", "Undecided")]);
        let pb = player("toonB", "Zerg", &[("toonA", "Win"), ("toonB", "Loss")]);
        let api = |p: &str| format!("/web-api{p}");
        let mut r: BTreeMap<String, (u16, Vec<u8>)> = BTreeMap::new();
        let mut put = |path: String, v: serde_json::Value| r.insert(path, (200, serde_json::to_vec(&v).unwrap()));
        put(api("/v1/leaderboard"), json!({"leaderboards": {
            "1": {"id": 101, "gamemode_id": 1, "gateway_id": 30, "season_id": 20, "name": "Korea"},
            "2": {"id": 100, "gamemode_id": 1, "gateway_id": 30, "season_id": 19, "name": "Korea"}}}));
        let cols = json!(["rank", "toon", "gateway_id", "rating", "bucket"]);
        put(api("/v1/leaderboard/101?offset=0&length=2"), json!({"columns": cols, "rows": [[1, "toonA", 30, 2300, 7], [2, "toonB", 30, 2100, 6]]}));
        put(api("/v1/leaderboard/101?offset=2&length=2"), json!({"columns": cols, "rows": [[3, "toonC", 30, 1500, 3]]}));
        put(api(&schema::profile("toonA", 30)), json!({"replays": [
            replay("MM-1", "2", 1000), replay("MM-2", "2", u32::MAX as u64), replay("123", "2", 1000), replay("MM-3", "6", 1000)]}));
        put(api(&schema::profile("toonB", 30)), json!({"replays": [replay("MM-1", "2", 1000), replay("MM-4", "2", 1000), replay("MM-5", "2", 1000)]}));
        put(api(&schema::match_info("MM-1")), json!({"players": {"1": pa, "2": pb}, "replays": [{"md5": md5_of(&new_rep), "url": file("1.rep")}]}));
        put(api(&schema::match_info("MM-4")), json!({"unexpected": true}));
        put(api(&schema::match_info("MM-5")), json!({"players": {"1": pa, "2": pb}, "replays": [{"md5": md5_of(&dup), "url": file("5.rep")}]}));
        r.insert("/files/1.rep".into(), (200, new_rep.clone()));
        r.insert("/files/5.rep".into(), (200, dup.clone()));
        let downloads = Arc::new(AtomicU32::new(0));
        let routes: Routes = Arc::new(Mutex::new(r));
        serve(listener, routes.clone(), downloads.clone());

        let settings = Settings { port: Some(port), rps: 1000.0, daily_max: 1000, shared_dir: None };
        let plan = Plan { gateways: vec![30], gamemode: 1, min_bucket: Some(6), page_size: 2 };
        let opt = Opts { limit: None, dry_run: false, retry_failed: None };

        let s = run(&settings, &paths, &plan, &opt).unwrap();
        assert_eq!(s, Summary { ok: 1, skipped: 1, failed: 1 });
        assert_eq!(downloads.load(Ordering::SeqCst), 2);
        let stored: Vec<_> = std::fs::read_dir(paths.raw_root.join(SRC)).unwrap().flatten().collect();
        assert_eq!(stored.len(), 1);
        assert_eq!(std::fs::read(stored[0].path()).unwrap(), new_rep);

        let lines: Vec<Record> = std::fs::read_to_string(paths.records()).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let by = |m: &str| lines.iter().find(|r| r.link == m).unwrap();
        assert_eq!(by("MM-1").status, "ok");
        assert_eq!(by("MM-1").winner.as_deref(), Some("toonA"));
        assert_eq!(by("MM-1").players.len(), 2);
        assert_eq!(by("MM-1").map_name.as_deref(), Some("Test Map"));
        assert!(by("MM-4").reason.as_deref().unwrap().starts_with("schema:"));
        assert_eq!(by("MM-5").reason.as_deref(), Some("duplicate"));
        assert!(!lines.iter().any(|r| r.link == "MM-2" || r.link == "MM-3" || r.link == "123"));

        // Resume: a second run fetches nothing new (043-R6).
        let s = run(&settings, &paths, &plan, &opt).unwrap();
        assert_eq!(s, Summary::default());
        assert_eq!(downloads.load(Ordering::SeqCst), 2);

        let md = status_markdown(&paths.records()).unwrap();
        assert!(md.contains("| TvZ | 1 |") && md.contains("| failed (schema) | 1 |"), "{md}");
        assert!(!md.contains("toon"), "status must not name players: {md}");

        // 043-R9 retry: MM-4 now answers with a player lacking `score`; only it is fetched again.
        let rep4 = b"seRS fake ladder replay 4".to_vec();
        let mut pb_no_score = pb.clone();
        pb_no_score.as_object_mut().unwrap().remove("score");
        routes.lock().unwrap().insert(api(&schema::match_info("MM-4")), (200, serde_json::to_vec(&json!({"players": {"1": pa, "2": pb_no_score}, "replays": [{"md5": md5_of(&rep4), "url": file("4.rep")}]})).unwrap()));
        routes.lock().unwrap().insert("/files/4.rep".into(), (200, rep4.clone()));
        let retry = Opts { retry_failed: Some("schema:".into()), ..opt };
        let s = run(&settings, &paths, &plan, &retry).unwrap();
        assert_eq!(s, Summary { ok: 1, skipped: 0, failed: 0 });
        assert_eq!(downloads.load(Ordering::SeqCst), 3);
        let last: Record = serde_json::from_str(std::fs::read_to_string(paths.records()).unwrap().lines().last().unwrap()).unwrap();
        assert_eq!((last.link.as_str(), last.status.as_str()), ("MM-4", "ok"));
        assert_eq!(last.players.iter().filter(|p| p.rating.is_none()).count(), 1);
        assert_eq!(last.season_id, Some(20));
        // Nothing left to retry, and status counts MM-4 once (latest record).
        assert_eq!(run(&settings, &paths, &plan, &retry).unwrap(), Summary::default());
        let md = status_markdown(&paths.records()).unwrap();
        assert!(md.contains("| ok | 2 |") && !md.contains("failed (schema)"), "{md}");
        let opt = Opts { retry_failed: None, ..retry };
        let (_, bad) = latest_records("{\"match\":\"MM-9\",\"status\":\"ok\",\"collected_at\":\"t\"}
not json
");
        assert_eq!(bad, 1);

        // 043-R5: refuse to run without a confirmed min_bucket.
        let unset = Plan { min_bucket: None, ..plan };
        assert!(run(&settings, &paths, &unset, &opt).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// 043-AC4: a whole collect run (leaderboard, profile, match info, download) reaches the
    /// server no faster than the configured rate, and the collector's own send log agrees.
    #[test]
    fn r043_ac4_collect_requests_stay_under_the_rate() {
        let dir = std::env::temp_dir().join(format!("uranium-ladder-rate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = Paths { ladder: dir.join("ladder"), raw_root: dir.join("raw") };
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let rep = b"seRS fake ladder replay rate".to_vec();
        let api = |p: &str| format!("/web-api{p}");
        let mut r: BTreeMap<String, (u16, Vec<u8>)> = BTreeMap::new();
        let mut put = |path: String, v: serde_json::Value| r.insert(path, (200, serde_json::to_vec(&v).unwrap()));
        put(api("/v1/leaderboard"), json!({"leaderboards": {"1": {"id": 101, "gamemode_id": 1, "gateway_id": 30, "season_id": 20, "name": "Korea"}}}));
        let cols = json!(["rank", "toon", "gateway_id", "rating", "bucket"]);
        put(api("/v1/leaderboard/101?offset=0&length=100"), json!({"columns": cols, "rows": [[1, "toonA", 30, 2300, 7]]}));
        put(api(&schema::profile("toonA", 30)), json!({"replays": [replay("MM-1", "2", 1000)]}));
        let pa = player("toonA", "Terran", &[("toonA", "Win"), ("toonB", "Loss")]);
        let pb = player("toonB", "Zerg", &[("toonA", "Win"), ("toonB", "Loss")]);
        put(api(&schema::match_info("MM-1")), json!({"players": {"1": pa, "2": pb},
            "replays": [{"md5": hex(&md5::Md5::digest(&rep)), "url": format!("http://127.0.0.1:{port}/files/1.rep")}]}));
        r.insert("/files/1.rep".into(), (200, rep));
        let arrivals = Arc::new(Mutex::new(Vec::new()));
        serve_logged(listener, Arc::new(Mutex::new(r)), Arc::new(AtomicU32::new(0)), arrivals.clone());

        let rps = 10.0;
        let settings = Settings { port: Some(port), rps, daily_max: 1000, shared_dir: None };
        let plan = Plan { gateways: vec![30], gamemode: 1, min_bucket: Some(6), page_size: 100 };
        let s = run(&settings, &paths, &plan, &Opts { limit: None, dry_run: false, retry_failed: None }).unwrap();
        assert_eq!(s.ok, 1);
        let t = arrivals.lock().unwrap().clone();
        // connect (prefix "" then "/web-api"), list, page, profile, match info, download.
        assert_eq!(t.len(), 7, "every request is paced");
        // Server side: sends are 100 ms apart; allow 20 ms of loopback jitter per gap.
        let min_gap = t.windows(2).map(|w| w[1] - w[0]).min().unwrap();
        assert!(min_gap >= std::time::Duration::from_millis(80), "requests {min_gap:?} apart at {rps} req/s");
        assert!(t[t.len() - 1] - t[0] >= std::time::Duration::from_millis(580));

        // The collector's own send log (printed when the run's `Http` is dropped).
        let mut http = Http::new(rps);
        for _ in 0..4 {
            http.get(&format!("http://127.0.0.1:{port}/web-api/v1/leaderboard")).unwrap();
        }
        let st = http.rate_stats();
        assert_eq!(st.requests, 4);
        assert!(st.within(rps), "{st:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
