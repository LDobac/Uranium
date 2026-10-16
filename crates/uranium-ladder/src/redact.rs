//! 043-R11: reduce an API response to its shape so it can be printed and reported without
//! player names, toons or other values.

use serde_json::{Map, Value, json};

const MAX_NUMERIC_KEY_LEN: usize = 5;

/// Replaces every value by a type marker. Arrays keep their length and the shape of the first
/// element. Object keys are kept only when they look like schema field names (lowercase
/// snake_case or digits); other keys could be player names used as map keys and are masked.
pub fn shape(v: &Value) -> Value {
    match v {
        Value::Null => Value::Null,
        Value::Bool(_) => json!("<bool>"),
        Value::Number(n) => json!(if n.is_f64() { "<float>" } else { "<int>" }),
        Value::String(s) => json!(format!("<str len={}>", s.chars().count())),
        Value::Array(a) => match a.first() {
            None => json!([]),
            Some(first) => json!({ "<array len>": a.len(), "<first>": shape(first) }),
        },
        Value::Object(m) => {
            let mut out = Map::new();
            for (i, (k, v)) in m.iter().enumerate() {
                let key = if is_schema_key(k) { k.clone() } else { format!("<key#{i} len={}>", k.chars().count()) };
                out.insert(key, shape(v));
            }
            Value::Object(out)
        }
    }
}

fn is_schema_key(k: &str) -> bool {
    let mut chars = k.chars();
    let Some(c0) = chars.next() else { return false };
    // Short numeric keys are gateway/leaderboard/gamemode ids; longer ones can be aurora ids or
    // all-digit toons (seen in match history), so they are masked.
    if k.chars().all(|c| c.is_ascii_digit()) {
        return k.len() <= MAX_NUMERIC_KEY_LEN;
    }
    c0.is_ascii_lowercase() && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Scheme and host of a URL (no path or query), for reporting where replay links point.
pub fn url_host(url: &str) -> String {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit_once('@').map_or(host, |(_, h)| h);
    format!("{scheme}://{host}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r043_r11_shape_drops_values() {
        let v = json!({
            "toon": "SomePlayer",
            "mmr": 2345,
            "ratio": 0.5,
            "online": true,
            "matches": [{"match_id": "MM-abc", "players": []}, {"match_id": "x"}],
            "SomePlayer": {"gateway_id": 30},
            "987654321": {"1234567890": 1},
            "10": null,
            "10414": null
        });
        let s = shape(&v);
        let text = s.to_string();
        for leaked in ["SomePlayer", "2345", "MM-abc", "0.5", "987654321", "1234567890"] {
            assert!(!text.contains(leaked), "{leaked} leaked: {text}");
        }
        assert_eq!(s["toon"], json!("<str len=10>"));
        assert_eq!(s["mmr"], json!("<int>"));
        assert_eq!(s["matches"]["<array len>"], json!(2));
        assert_eq!(s["matches"]["<first>"]["players"], json!([]));
        assert!(s.get("10").is_some() && s.get("10414").is_some());
    }

    #[test]
    fn r043_r11_url_host_only() {
        assert_eq!(url_host("http://cdn.example.com/replays/MM-1.rep?sig=abc"), "http://cdn.example.com");
        assert_eq!(url_host("https://u:p@cdn.example.com:8443/x"), "https://cdn.example.com:8443");
    }
}
