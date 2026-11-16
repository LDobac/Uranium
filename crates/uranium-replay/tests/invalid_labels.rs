use uranium_data::testkit::game_with_invalid;

#[test]
fn invalid_label_share_filters_perspective_040_r19() {
    use uranium_env::features::label::INVALID_LABEL_MAX;
    use uranium_replay::pipeline::invalid_label_filter;
    assert_eq!(INVALID_LABEL_MAX, 0.20);
    assert_eq!(invalid_label_filter(&game_with_invalid(20, 0), None).unwrap(), None);
    assert_eq!(invalid_label_filter(&game_with_invalid(20, 3), None).unwrap(), None, "15% is kept");
    let r = invalid_label_filter(&game_with_invalid(20, 4), None).unwrap().expect("20% is rejected");
    assert_eq!(r, "invalid_labels>=20% (20.0%, 4/20) (040-R19)");
    assert_eq!(uranium_replay::stats::filter_group(&r), uranium_replay::stats::INVALID_LABEL_GROUP);
    let r = invalid_label_filter(&game_with_invalid(20, 11), None).unwrap().unwrap();
    assert!(r.starts_with("invalid_labels>=20% (55.0%, 11/20)"), "{r}");
    assert_eq!(invalid_label_filter(&game_with_invalid(20, 11), None).unwrap().unwrap(), r);
}
