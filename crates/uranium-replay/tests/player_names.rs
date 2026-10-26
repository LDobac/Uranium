//! 020-AC7 check on a real replay: the header's player names are the ones hostd gave
//! (`p<i>-<name>`, 020-R16). Names are only compared, never printed (040-R14).
//!   URANIUM_TEST_REP=<g1.rep> URANIUM_TEST_REP_NAMES=p0-<run>,p1-T0-zerg \
//!     cargo test -p uranium-replay --test player_names -- --ignored

#[test]
#[ignore = "needs URANIUM_TEST_REP and URANIUM_TEST_REP_NAMES"]
fn replay_player_names_020_ac7() {
    let rep = std::env::var("URANIUM_TEST_REP").expect("URANIUM_TEST_REP");
    let want = std::env::var("URANIUM_TEST_REP_NAMES").expect("URANIUM_TEST_REP_NAMES");
    let r = uranium_replay::parse(&std::fs::read(&rep).unwrap()).unwrap();
    let mut names: Vec<String> = r.header.players.iter().map(|p| String::from_utf8_lossy(&p.name).trim_end_matches('\0').to_string()).collect();
    let mut want: Vec<String> = want.split(',').map(str::to_string).collect();
    names.sort();
    want.sort();
    assert!(names == want, "player names differ from URANIUM_TEST_REP_NAMES ({} players)", names.len());
}
