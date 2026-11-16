use uranium_env::features::Observation;
pub use uranium_env::host::collate::*;

pub fn obs_batch_packed(rows: &[&Observation], planes: &[&crate::pack::Planes], pad: usize, threads: usize) -> HostObsBatch {
    assert_eq!(rows.len(), planes.len(), "one plane set per row");
    let mut x = obs_batch(rows, pad);
    let (sc, lc) = planes.first().map_or((0, 0), |p| (p.spatial_len(), p.loc_len()));
    x.l = lc / CELLS;
    x.spatial = vec![0f32; rows.len() * sc];
    x.loc_class_mask = vec![false; rows.len() * lc];
    let (mut sp, mut lm): (&mut [f32], &mut [bool]) = (&mut x.spatial, &mut x.loc_class_mask);
    let mut jobs = Vec::with_capacity(planes.len());
    for p in planes {
        let (s, rest) = std::mem::take(&mut sp).split_at_mut(sc);
        sp = rest;
        let (l, rest) = std::mem::take(&mut lm).split_at_mut(lc);
        lm = rest;
        jobs.push((*p, s, l));
    }
    let per = jobs.len().div_ceil(threads.clamp(1, jobs.len().max(1))).max(1);
    std::thread::scope(|s| {
        for run in jobs.chunks_mut(per) {
            s.spawn(move || {
                for (p, sp, lm) in run.iter_mut() {
                    p.restore_into(sp, lm);
                }
            });
        }
    });
    x
}
