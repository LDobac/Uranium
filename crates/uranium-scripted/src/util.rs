use uranium_proto::{Pos, UnitState, flags};

pub(crate) fn done(u: &UnitState) -> bool {
    u.has(flags::COMPLETED) && !u.has(flags::BEING_CONSTRUCTED) && u.remaining_build == 0
}

pub(crate) fn dist2(a: Pos, b: Pos) -> i64 {
    let dx = a.0 as i64 - b.0 as i64;
    let dy = a.1 as i64 - b.1 as i64;
    dx * dx + dy * dy
}

pub(crate) fn centroid(units: &[&UnitState]) -> Pos {
    let n = units.len().max(1) as u64;
    let (sx, sy) = units.iter().fold((0u64, 0u64), |(x, y), u| (x + u.pos.0 as u64, y + u.pos.1 as u64));
    ((sx / n) as u16, (sy / n) as u16)
}

pub(crate) fn nearest<'a>(p: Pos, units: &[&'a UnitState]) -> Option<&'a UnitState> {
    units.iter().copied().min_by_key(|u| dist2(p, u.pos))
}

pub(crate) fn is_neutral_owner(owner: u8) -> bool {
    owner >= 8
}

pub(crate) fn is_enemy(u: &UnitState, me: u8) -> bool {
    u.owner != me && !is_neutral_owner(u.owner)
}
