use std::path::Path;

use uranium_store::dotenv::load_env;

fn clear(case: &serde_json::Value) {
    for k in case["keys"].as_array().unwrap() {
        unsafe { std::env::remove_var(k.as_str().unwrap()) };
    }
    for k in case["env"].as_object().unwrap().keys() {
        unsafe { std::env::remove_var(k) };
    }
}

fn preset(case: &serde_json::Value) {
    clear(case);
    for (k, v) in case["env"].as_object().unwrap() {
        unsafe { std::env::set_var(k, v.as_str().unwrap()) };
    }
}

fn check(case: &serde_json::Value, how: &str) {
    let name = case["name"].as_str().unwrap();
    let want = case["want"].as_object().unwrap();
    for (k, v) in want {
        assert_eq!(std::env::var(k).ok().as_deref(), v.as_str(), "{name} ({how}): {k}");
    }
    for k in case["keys"].as_array().unwrap() {
        let k = k.as_str().unwrap();
        if !want.contains_key(k) {
            assert!(std::env::var_os(k).is_none(), "{name} ({how}): {k} must stay unset");
        }
    }
}

#[test]
fn dotenvy_reads_the_shared_case_table_001_r9() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dotenv_cases.json");
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("uranium-store-dotenv-cases-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("case.env");
    let cases = cases.as_array().unwrap();
    assert!(cases.len() > 100);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        std::fs::write(&file, case["text"].as_str().unwrap()).unwrap();
        preset(case);
        let res = dotenvy::from_path(&file);
        assert_eq!(res.is_err(), !case["error_line"].is_null(), "{name}: {res:?}");
        check(case, "dotenvy");
        preset(case);
        unsafe { std::env::set_var("URANIUM_ENV_FILE", &file) };
        load_env();
        check(case, "load_env");
        clear(case);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
