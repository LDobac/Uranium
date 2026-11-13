//! 051-AC1 (intervention part) on a synthetic observation: a narrowed `type_mask` leaves
//! `unit_cap`, `loc_class_mask` and the model inputs alone, and the executor builds the same
//! command for a decision the mask still allows (031-R7) while refusing a closed one (031-R4).

mod common;

use common::*;
use uranium_env::branch::{Alternative, Intervention, InterventionRule, Op, open_ids};
use uranium_env::features::executor::{HostAction, Rejected, execute};
use uranium_env::features::extract::action_table;
use uranium_env::features::world::World;
use uranium_env::features::*;

fn host(action: &str) -> HostAction {
    HostAction { action: action_table().by_name(action).unwrap().id, delay_bucket: 0, queued: false, selected: vec![], target_unit: None, target_loc: None }
}

fn alt(rules: &[(Op, &str, u32, u32)]) -> Alternative {
    Alternative { name: "alt".into(), rules: rules.iter().map(|&(op, a, from_s, to_s)| InterventionRule { op, action: a.into(), from_s, to_s }).collect() }
}

#[test]
fn branch_intervention_narrows_type_mask_only_051_ac1() {
    let map = map_info();
    let mut mem = AgentMemory::new(&game_info());
    let f = frame(100, rich_units().units, true);
    let o = extract(&f, &mut mem, &map, None);
    let w = World::new(&f, &mem, &map);
    let marine = action_table().by_name("Train_Marine").unwrap().id as usize;
    assert!(o.masks.type_mask[marine], "fixture must allow Train_Marine");
    let want = execute(&w, &o, &host("Train_Marine")).unwrap();

    for rules in [
        vec![(Op::Force, "Train_Marine", 0, 0)],
        vec![(Op::Ban, "Build_Bunker", 0, 90)],
        vec![(Op::Ban, "Train_Marine", 0, 90)],
        vec![(Op::Ban, "Build_Bunker", 0, 90), (Op::Force, "Train_Marine", 0, 0)],
    ] {
        let mut iv = Intervention::new(&alt(&rules), action_table(), f.frame).unwrap();
        let mut n = o.clone();
        iv.apply_to(f.frame, &mut n);
        // 051-R7 ③④: only type_mask changes, and only by closing bits.
        assert_eq!((&n.obs, &n.masks.unit_cap, &n.masks.loc_class_mask, &n.masks.is_building, &n.entity_ids), (&o.obs, &o.masks.unit_cap, &o.masks.loc_class_mask, &o.masks.is_building, &o.entity_ids));
        assert!(n.masks.type_mask.iter().zip(&o.masks.type_mask).all(|(a, b)| !*a || *b), "{rules:?} opened a bit");
        assert!(n.masks.type_mask.iter().any(|&b| b));
        let got = execute(&w, &n, &host("Train_Marine"));
        if n.masks.type_mask[marine] {
            // 031-R7: same command under the narrowed mask.
            assert_eq!(got.unwrap(), want, "{rules:?}");
        } else {
            assert_eq!(got.unwrap_err(), Rejected::TypeMasked, "{rules:?}");
        }
        if rules.iter().any(|r| r.0 == Op::Force) {
            assert_eq!(open_ids(&n.masks.type_mask), vec![marine as u16]);
            // NoOp is closed under a force, so the executor refuses it.
            assert_eq!(execute(&w, &n, &host("NoOp")).unwrap_err(), Rejected::TypeMasked);
        }
    }
}
