//! Paced HTTP to the local web API and replay storage (043-R2, R3, R4, R9).

use std::time::Duration;

use anyhow::{Result, bail};
use serde::de::DeserializeOwned;

use crate::env::Settings;
use crate::limit::Limiter;
use crate::{port, schema};

/// Body size cap for one response or replay.
const MAX_BODY: u64 = 64 << 20;
/// Transient failures (network, HTTP 5xx, `Internal error` bodies at 200) are retried this
/// many times with a growing pause.
const RETRIES: u32 = 2;
const RETRY_PAUSE: Duration = Duration::from_secs(5);

pub struct Http {
    agent: ureq::Agent,
    limiter: Limiter,
    rps: f64,
}

/// 043-AC4: every run logs the rate its paced requests actually used, also when it stops early.
impl Drop for Http {
    fn drop(&mut self) {
        let s = self.limiter.stats();
        if s.requests > 0 {
            println!("{}", s.line(self.rps));
        }
    }
}

pub struct Reply {
    pub status: u16,
    pub header_names: Vec<String>,
    pub body: Vec<u8>,
}

/// Why one fetch failed; recorded as the `failed` reason (043-R9).
#[derive(Debug)]
pub enum FetchError {
    Net(String),
    Status(u16),
    Schema(String),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Net(e) => write!(f, "net: {e}"),
            Self::Status(s) => write!(f, "http {s}"),
            Self::Schema(e) => write!(f, "schema: {e}"),
        }
    }
}

impl Http {
    pub fn new(rps: f64) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        Self { agent, limiter: Limiter::new(rps), rps }
    }

    /// 043-AC4: observed rate so far.
    pub fn rate_stats(&self) -> crate::limit::RateStats {
        self.limiter.stats()
    }

    /// One paced GET (043-R4).
    pub fn get(&mut self, url: &str) -> Result<Reply> {
        self.limiter.wait();
        let mut resp = self.agent.get(url).call()?;
        let status = resp.status().as_u16();
        let header_names = resp.headers().keys().map(|k| k.as_str().to_string()).collect();
        let body = resp.body_mut().with_config().limit(MAX_BODY).read_to_vec()?;
        Ok(Reply { status, header_names, body })
    }

    /// GET with retries for transient failures; the body of a 200 reply.
    pub fn get_ok(&mut self, url: &str) -> Result<Vec<u8>, FetchError> {
        let mut attempt = 0;
        loop {
            let err = match self.get(url) {
                Ok(r) if r.status == 200 && !is_internal_error(&r.body) => return Ok(r.body),
                Ok(r) if r.status == 200 => FetchError::Net("Internal error body".into()),
                Ok(r) if r.status < 500 => return Err(FetchError::Status(r.status)),
                Ok(r) => FetchError::Status(r.status),
                Err(e) => FetchError::Net(format!("{e:#}")),
            };
            if attempt == RETRIES {
                return Err(err);
            }
            attempt += 1;
            std::thread::sleep(RETRY_PAUSE * attempt);
        }
    }
}

fn is_internal_error(body: &[u8]) -> bool {
    body.len() < 256 && String::from_utf8_lossy(body).contains("Internal error")
}

/// Local API root: `http://127.0.0.1:<port><prefix>`.
pub struct Api {
    pub base: String,
}

impl Api {
    /// 043-R3: the first candidate port and prefix whose leaderboard list answers with JSON.
    pub fn connect(http: &mut Http, settings: &Settings) -> Result<Self> {
        let ports = match settings.port {
            Some(p) => vec![p],
            None => port::candidates()?,
        };
        for port in &ports {
            for prefix in schema::PATH_PREFIXES {
                let base = format!("http://127.0.0.1:{port}{prefix}");
                match http.get(&format!("{base}{}", schema::LEADERBOARD_LIST)) {
                    Ok(r) if r.status == 200 && serde_json::from_slice::<serde_json::Value>(&r.body).is_ok() => {
                        println!("api: port {port}, prefix {prefix:?}");
                        return Ok(Self { base });
                    }
                    Ok(r) => println!("api: port {port}, prefix {prefix:?} -> HTTP {}", r.status),
                    Err(e) => println!("api: port {port}, prefix {prefix:?} -> {e:#}"),
                }
            }
        }
        bail!("no candidate port answered {} with JSON (ports tried: {ports:?})", schema::LEADERBOARD_LIST)
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    /// GET and parse; a body that does not match the schema is `FetchError::Schema` (043-R9).
    pub fn json<T: DeserializeOwned>(&self, http: &mut Http, path: &str) -> Result<T, FetchError> {
        let body = http.get_ok(&self.url(path))?;
        serde_json::from_slice(&clean(&body)).map_err(|e| FetchError::Schema(mask_quoted(&e.to_string())))
    }
}

/// Some names arrive cut mid-character (invalid UTF-8) or as unpaired surrogate escapes; both
/// become U+FFFD so the rest of the response still parses.
fn clean(body: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(body);
    fix_lone_surrogates(text.as_bytes()).into_owned()
}

/// The API escapes some names with unpaired UTF-16 surrogates (`\ud83d` alone), which
/// serde_json rejects; replace each unpaired one with `�`.
fn fix_lone_surrogates(body: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    fn esc(b: &[u8], i: usize) -> Option<u16> {
        if b.get(i) != Some(&b'\\') || b.get(i + 1).is_none_or(|c| *c != b'u') {
            return None;
        }
        u16::from_str_radix(std::str::from_utf8(b.get(i + 2..i + 6)?).ok()?, 16).ok()
    }
    let hi = |u: u16| (0xD800..0xDC00).contains(&u);
    let lo = |u: u16| (0xDC00..0xE000).contains(&u);
    let mut out: Option<Vec<u8>> = None;
    let mut i = 0;
    while i < body.len() {
        if body[i] == b'\\' {
            if let Some(u) = esc(body, i) {
                let paired = hi(u) && esc(body, i + 6).is_some_and(lo);
                if paired {
                    if let Some(o) = out.as_mut() {
                        o.extend_from_slice(&body[i..i + 12]);
                    }
                    i += 12;
                    continue;
                }
                if hi(u) || lo(u) {
                    let o = out.get_or_insert_with(|| body[..i].to_vec());
                    o.extend_from_slice(b"\\ufffd");
                    i += 6;
                    continue;
                }
            }
            // Any other escape: copy the backslash and the escaped byte together.
            let end = (i + 2).min(body.len());
            if let Some(o) = out.as_mut() {
                o.extend_from_slice(&body[i..end]);
            }
            i = end;
            continue;
        }
        if let Some(o) = out.as_mut() {
            o.push(body[i]);
        }
        i += 1;
    }
    match out {
        Some(o) => std::borrow::Cow::Owned(o),
        None => std::borrow::Cow::Borrowed(body),
    }
}

/// serde errors quote the offending value (possibly a toon); keep only the message (043-R11).
fn mask_quoted(s: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for c in s.chars() {
        if c == '"' {
            inside = !inside;
            if inside {
                out.push_str("\"<s>\"");
            }
        } else if !inside {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r043_r11_schema_error_masks_values() {
        let e = serde_json::from_str::<u32>("\"SomeToon\"").unwrap_err().to_string();
        assert!(e.contains("SomeToon"));
        let m = mask_quoted(&e);
        assert!(!m.contains("SomeToon") && m.contains("<s>"), "{m}");
    }

    #[test]
    fn unpaired_surrogates_become_replacement_char() {
        let raw = br#"{"a":"x\ud83dy","b":"\ud83d\ude00","c":"\ude00","d":"\\ud83d","e":"\u00e9"}"#;
        assert!(serde_json::from_slice::<serde_json::Value>(raw).is_err());
        let v: serde_json::Value = serde_json::from_slice(&fix_lone_surrogates(raw)).unwrap();
        assert_eq!(v["a"], "x\u{fffd}y");
        assert_eq!(v["b"], "\u{1f600}");
        assert_eq!(v["c"], "\u{fffd}");
        assert_eq!(v["d"], "\\ud83d");
        assert_eq!(v["e"], "é");
        assert!(matches!(fix_lone_surrogates(br#"{"ok":"\u00e9"}"#), std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn truncated_utf8_name_still_parses() {
        // A Hangul syllable cut after its first byte, right before the closing quote.
        let raw = b"{\"name\":\"\xeb\x9e\x98\xec\",\"n\":1}";
        assert!(serde_json::from_slice::<serde_json::Value>(raw).is_err());
        let v: serde_json::Value = serde_json::from_slice(&clean(raw)).unwrap();
        assert_eq!(v["name"], "\u{b798}\u{fffd}");
        assert_eq!(v["n"], 1);
    }
}
