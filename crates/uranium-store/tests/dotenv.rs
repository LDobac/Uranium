use uranium_store::dotenv::load_env;

#[test]
fn load_env_reads_uranium_env_file_and_keeps_process_values_001_r9() {
    let dir = std::env::temp_dir().join(format!("uranium-store-dotenv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("custom.env");
    std::fs::write(&file, "URANIUM_TEST_DOTENV_FROM_FILE=file\nURANIUM_TEST_DOTENV_FROM_PROCESS=file\n").unwrap();
    unsafe {
        std::env::set_var("URANIUM_ENV_FILE", &file);
        std::env::set_var("URANIUM_TEST_DOTENV_FROM_PROCESS", "process");
        std::env::remove_var("URANIUM_TEST_DOTENV_FROM_FILE");
    }
    load_env();
    assert_eq!(std::env::var("URANIUM_TEST_DOTENV_FROM_FILE").unwrap(), "file");
    assert_eq!(std::env::var("URANIUM_TEST_DOTENV_FROM_PROCESS").unwrap(), "process");
    unsafe {
        std::env::set_var("URANIUM_ENV_FILE", dir.join("missing.env"));
        std::env::remove_var("URANIUM_TEST_DOTENV_FROM_FILE");
    }
    load_env();
    assert!(std::env::var_os("URANIUM_TEST_DOTENV_FROM_FILE").is_none(), "a missing URANIUM_ENV_FILE loads nothing");
    assert_eq!(std::env::var("URANIUM_TEST_DOTENV_FROM_PROCESS").unwrap(), "process");
    let _ = std::fs::remove_dir_all(&dir);
}
