use std::io::Write as _;

pub fn load_env() {
    let res = match std::env::var_os("URANIUM_ENV_FILE") {
        Some(path) => dotenvy::from_path(path),
        None => dotenvy::dotenv().map(|_| ()),
    };
    if let Err(e) = res {
        if !e.not_found() {
            let _ = writeln!(std::io::stderr(), "warning: failed to load .env: {e}");
        }
    }
}
