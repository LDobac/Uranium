use uranium_env::features::ORDER_NONE;
use uranium_env::types::{NO_TYPE, TypeTables, order_id};

#[test]
fn sentinel_ids_match_the_tables_013_r1() {
    assert_eq!(TypeTables::get().unit_by_name("None").map(|u| u.id as u16), Some(NO_TYPE));
    assert_eq!(order_id("None"), Some(ORDER_NONE));
}
