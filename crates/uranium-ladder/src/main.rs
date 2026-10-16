//! `uranium-ladder`: SC:R ladder replay collector over the logged-in client's local web API
//! (spec 043). `probe` is the 043-AC1 spike, `collect` the daily run, `status` its summary.

mod api;
mod collect;
mod env;
mod limit;
mod port;
mod redact;
mod schema;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use sha2::{Digest, Sha256};

use crate::api::{Api, Http};
use crate::env::Settings;
use crate::limit::Daily;

#[derive(Parser)]
#[command(name = "uranium-ladder", about = "SC:R ladder replay collector (spec 043)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 043-AC1 spike: detect the API, fetch one step of the leaderboard → profile → match →
    /// replay chain, save the raw response outside git and print only its shape.
    Probe(ProbeArgs),
    /// Print the redacted shape (043-R11) of a saved response, optionally of one JSON pointer.
    Shape {
        file: PathBuf,
        /// RFC 6901 pointer, e.g. /replays/0
        #[arg(long)]
        pointer: Option<String>,
    },
    /// Daily collection run (043-R4..R9) into raw/scr_ladder and ladder/collect.jsonl.
    Collect {
        #[arg(long)]
        plan: PathBuf,
        /// Most downloads in this run
        #[arg(long)]
        limit: Option<u32>,
        /// List new ladder links per tracked player without fetching match info or replays
        #[arg(long)]
        dry_run: bool,
    },
    /// Counts of collect.jsonl per status, matchup and map (no player names).
    Status {
        /// Write the markdown here instead of stdout
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Args)]
struct ProbeArgs {
    /// Raw-response folder (default: <URANIUM_SHARED_DIR>/datasets/replays/ladder/probe)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Step 2: fetch the first page of this leaderboard id
    #[arg(long)]
    leaderboard: Option<u64>,
    /// Step 3: profile of --toon on --gateway (its `replays[]` list); with --gamemode and
    /// --season also the match history
    #[arg(long, requires = "gateway")]
    toon: Option<String>,
    #[arg(long)]
    gateway: Option<u32>,
    #[arg(long)]
    gamemode: Option<String>,
    #[arg(long)]
    season: Option<String>,
    /// Step 4: match info (with the replay links) of this profile `link`
    #[arg(long)]
    match_id: Option<String>,
    /// Step 5: download a replay link that appears in a saved probe response (043-R2)
    #[arg(long)]
    download: Option<String>,
}

fn main() -> Result<()> {
    env::load_env();
    let cli = Cli::parse();
    let settings = Settings::from_env()?;
    match cli.cmd {
        Cmd::Probe(args) => probe(&settings, args),
        Cmd::Shape { file, pointer } => {
            let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&file)?)?;
            let v = match &pointer {
                Some(p) => v.pointer(p).with_context(|| format!("no value at {p}"))?,
                None => &v,
            };
            println!("{}", serde_json::to_string_pretty(&redact::shape(v))?);
            Ok(())
        }
        Cmd::Collect { plan, limit, dry_run } => {
            let plan = collect::Plan::load(&plan)?;
            let paths = collect::Paths::from_settings(&settings)?;
            let s = collect::run(&settings, &paths, &plan, &collect::Opts { limit, dry_run })?;
            println!("done: ok {}, skipped {}, failed {}", s.ok, s.skipped, s.failed);
            Ok(())
        }
        Cmd::Status { out } => {
            let paths = collect::Paths::from_settings(&settings)?;
            let md = collect::status_markdown(&paths.ladder.join("collect.jsonl"))?;
            match out {
                Some(p) => std::fs::write(p, md)?,
                None => print!("{md}"),
            }
            Ok(())
        }
    }
}

fn probe(settings: &Settings, a: ProbeArgs) -> Result<()> {
    let out = match a.out {
        Some(p) => p,
        None => settings.ladder_dir()?.join("probe"),
    };
    std::fs::create_dir_all(&out)?;
    let mut http = Http::new(settings.rps);

    if let Some(url) = &a.download {
        return probe_download(settings, &mut http, &out, url);
    }
    let api = Api::connect(&mut http, settings)?;
    // Each step is chosen by the caller after reading the previous raw response, so no field
    // is picked automatically here (043-R9).
    let mut steps: Vec<(&str, String, String)> = Vec::new();
    if let Some(id) = a.leaderboard {
        steps.push(("leaderboard page", schema::leaderboard_page(id, 0, 100), format!("leaderboard_{id}.json")));
    }
    if let (Some(toon), Some(gw)) = (&a.toon, a.gateway) {
        steps.push(("profile", schema::profile(toon, gw), "profile.json".into()));
        if let (Some(gm), Some(season)) = (&a.gamemode, &a.season) {
            steps.push(("match history", schema::match_history(toon, &gw.to_string(), gm, season), "history.json".into()));
        }
    }
    if let Some(id) = &a.match_id {
        steps.push(("match info", schema::match_info(id), "match.json".into()));
    }
    if steps.is_empty() {
        steps.push(("leaderboard list", schema::LEADERBOARD_LIST.into(), "leaderboard.json".into()));
        steps.push(("gateways", schema::GATEWAYS.into(), "gateway.json".into()));
    }
    for (what, path, file) in steps {
        let r = http.get(&api.url(&path))?;
        std::fs::write(out.join(&file), &r.body)?;
        // 043-R11: the path may hold a toon, so only the step name is printed.
        println!("\n== {what}: HTTP {} ({} bytes) -> <probe>/{file}", r.status, r.body.len());
        println!("headers: {}", r.header_names.join(", "));
        print_shape(&r.body);
    }
    println!("\nraw responses: {}", out.display());
    Ok(())
}

/// Step 5: the link must come from a saved API response (043-R2).
fn probe_download(settings: &Settings, http: &mut Http, out: &Path, url: &str) -> Result<()> {
    if !link_in_saved_responses(out, url)? {
        bail!("this link does not appear in any saved probe response in {} (043-R2)", out.display());
    }
    let mut daily = Daily::load_today(&out.join("..").join("daily.json"), settings.daily_max)?;
    if daily.remaining() == 0 {
        bail!("daily download budget ({}) used up (043-R4)", settings.daily_max);
    }
    let r = http.get(url)?;
    println!("== replay download from {}: HTTP {} ({} bytes)", redact::url_host(url), r.status, r.body.len());
    println!("headers: {}", r.header_names.join(", "));
    if r.status != 200 {
        print_shape(&r.body);
        bail!("download failed with HTTP {}", r.status);
    }
    daily.record()?;
    let sha = hex(&Sha256::digest(&r.body));
    std::fs::write(out.join("replay.rep"), &r.body)?;
    println!("sha256 {sha}, first bytes {}", hex(&r.body[..r.body.len().min(16)]));
    println!("saved: {}", out.join("replay.rep").display());
    Ok(())
}

fn link_in_saved_responses(dir: &Path, url: &str) -> Result<bool> {
    let escaped = url.replace('/', "\\/");
    for e in std::fs::read_dir(dir)?.flatten() {
        if e.path().extension().is_some_and(|x| x == "json") {
            let text = String::from_utf8_lossy(&std::fs::read(e.path())?).into_owned();
            if text.contains(url) || text.contains(&escaped) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn print_shape(body: &[u8]) {
    match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&redact::shape(&v)).unwrap_or_default()),
        Err(_) => println!("<non-JSON body, {} bytes>", body.len()),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r043_r2_download_link_must_come_from_saved_response() {
        let dir = std::env::temp_dir().join(format!("uranium-ladder-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("match.json"), r#"{"replays":[{"url":"http:\/\/cdn.example.com\/r\/MM-1.rep"}]}"#).unwrap();
        assert!(link_in_saved_responses(&dir, "http://cdn.example.com/r/MM-1.rep").unwrap());
        assert!(!link_in_saved_responses(&dir, "http://other.example.com/x.rep").unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
