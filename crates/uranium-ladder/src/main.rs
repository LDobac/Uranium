//! `uranium-ladder`: SC:R ladder replay collector over the logged-in client's local web API
//! (spec 043). Stage 1 provides `probe` (043-AC1); `collect` and `status` follow once the
//! response schema has been measured (043-R9, 043-R10).

mod env;
mod limit;
mod port;
mod redact;

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use sha2::{Digest, Sha256};

use crate::env::Settings;
use crate::limit::{Daily, Limiter};

/// Path prefixes tried in front of the documented `/v1/...` routes (043 background: TBD(M4)).
const PATH_PREFIXES: [&str; 2] = ["", "/web-api"];
/// Body size cap for one response or replay.
const MAX_BODY: u64 = 64 << 20;

#[derive(Parser)]
#[command(name = "uranium-ladder", about = "SC:R ladder replay collector (spec 043)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 043-AC1 spike: detect the API, fetch one step of the leaderboard → history → match →
    /// replay chain, save the raw response outside git and print only its shape.
    Probe(ProbeArgs),
    /// Print the redacted shape (043-R11) of a saved response, optionally of one JSON pointer.
    Shape {
        file: PathBuf,
        /// RFC 6901 pointer, e.g. /replays/0
        #[arg(long)]
        pointer: Option<String>,
    },
}

#[derive(Args)]
struct ProbeArgs {
    /// Raw-response folder (default: <URANIUM_SHARED_DIR>/datasets/replays/ladder/probe)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Step 2: fetch the first page of this leaderboard id
    #[arg(long)]
    leaderboard: Option<String>,
    /// Step 3: profile of --toon on --gateway (its `replays[]` list); with --gamemode and
    /// --season also the match history
    #[arg(long, requires = "gateway")]
    toon: Option<String>,
    #[arg(long)]
    gateway: Option<String>,
    #[arg(long)]
    gamemode: Option<String>,
    #[arg(long)]
    season: Option<String>,
    /// Step 4: match info (with the replay link) of this match id
    #[arg(long)]
    match_id: Option<String>,
    /// Step 5: download a replay link that appears in a saved probe response (043-R2)
    #[arg(long)]
    download: Option<String>,
}

struct Http {
    agent: ureq::Agent,
    limiter: Limiter,
}

struct Reply {
    status: u16,
    header_names: Vec<String>,
    body: Vec<u8>,
}

impl Http {
    fn new(rps: f64) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        Self { agent, limiter: Limiter::new(rps) }
    }

    /// One paced GET (043-R4).
    fn get(&mut self, url: &str) -> Result<Reply> {
        self.limiter.wait();
        let mut resp = self.agent.get(url).call().context("request failed")?;
        let status = resp.status().as_u16();
        let header_names = resp.headers().keys().map(|k| k.as_str().to_string()).collect();
        let body = resp.body_mut().with_config().limit(MAX_BODY).read_to_vec()?;
        Ok(Reply { status, header_names, body })
    }
}

/// Local API root: `http://127.0.0.1:<port><prefix>`.
struct Api {
    base: String,
}

impl Api {
    /// 043-R3: the first candidate port and prefix whose leaderboard list answers with JSON.
    fn connect(http: &mut Http, settings: &Settings) -> Result<Self> {
        let ports = match settings.port {
            Some(p) => vec![p],
            None => port::candidates()?,
        };
        for port in &ports {
            for prefix in PATH_PREFIXES {
                let base = format!("http://127.0.0.1:{port}{prefix}");
                match http.get(&format!("{base}/v1/leaderboard")) {
                    Ok(r) if r.status == 200 && serde_json::from_slice::<serde_json::Value>(&r.body).is_ok() => {
                        println!("api: port {port}, prefix {:?}", prefix);
                        return Ok(Self { base });
                    }
                    Ok(r) => println!("api: port {port}, prefix {prefix:?} -> HTTP {}", r.status),
                    Err(e) => println!("api: port {port}, prefix {prefix:?} -> {e:#}"),
                }
            }
        }
        bail!("no candidate port answered /v1/leaderboard with JSON (ports tried: {ports:?})")
    }
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
    // of the (not yet measured) schema is guessed here (043-R9).
    let mut steps: Vec<(&str, String, String)> = Vec::new();
    if let Some(id) = &a.leaderboard {
        steps.push(("leaderboard page", format!("/v1/leaderboard/{}?offset=0&length=100", enc(id)), format!("leaderboard_{}.json", file_part(id))));
    }
    if let (Some(toon), Some(gw)) = (&a.toon, &a.gateway) {
        steps.push((
            "profile",
            format!("/v2/aurora-profile-by-toon/{}/{}?request_flags=scr_profile", enc(toon), enc(gw)),
            "profile.json".into(),
        ));
    }
    if let (Some(toon), Some(gw), Some(gm), Some(season)) = (&a.toon, &a.gateway, &a.gamemode, &a.season) {
        steps.push((
            "match history",
            format!("/v1/matchmaker-gameinfo-by-toon/{}/{}/{}/{}", enc(toon), enc(gw), enc(gm), enc(season)),
            "history.json".into(),
        ));
    }
    if let Some(id) = &a.match_id {
        steps.push(("match info", format!("/v1/matchmaker-gameinfo-playerinfo/{}", enc(id)), "match.json".into()));
    }
    if steps.is_empty() {
        steps.push(("leaderboard list", "/v1/leaderboard".into(), "leaderboard.json".into()));
        steps.push(("gateways", "/v1/gateway".into(), "gateway.json".into()));
    }
    for (what, path, file) in steps {
        let r = http.get(&format!("{}{path}", api.base))?;
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

/// Percent-encodes one path segment.
fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn file_part(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
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

    #[test]
    fn path_segment_encoding() {
        assert_eq!(enc("a b/#c"), "a%20b%2F%23c");
        assert_eq!(enc("Toon_1.x"), "Toon_1.x");
    }
}
