//! Optional wall-time marks inside the forward pass, for profiling only (`uranium-train
//! bc-profile`). Off by default: `mark` is a no-op until [`enable`] on this thread. When on, the
//! device is synced at every mark so the GPU work queued since the previous mark is charged to
//! the named phase.

use std::cell::RefCell;
use std::time::Instant;

use burn::tensor::Device;

struct State {
    device: Device,
    last: Instant,
    spent: Vec<(&'static str, f64)>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Starts collecting on this thread (resets the totals).
pub fn enable(device: &Device) {
    STATE.with(|s| *s.borrow_mut() = Some(State { device: device.clone(), last: Instant::now(), spent: Vec::new() }));
}

/// Stops collecting and returns seconds per phase, in first-seen order.
pub fn take() -> Vec<(&'static str, f64)> {
    STATE.with(|s| s.borrow_mut().take().map(|st| st.spent).unwrap_or_default())
}

/// Restarts the clock without charging anything (call before the region of interest).
pub fn reset_clock() {
    STATE.with(|s| {
        if let Some(st) = s.borrow_mut().as_mut() {
            let _ = st.device.sync();
            st.last = Instant::now();
        }
    });
}

/// Charges the time since the previous mark to `phase`.
pub fn mark(phase: &'static str) {
    STATE.with(|s| {
        let mut b = s.borrow_mut();
        let Some(st) = b.as_mut() else { return };
        let _ = st.device.sync();
        let now = Instant::now();
        let dt = (now - st.last).as_secs_f64();
        st.last = now;
        match st.spent.iter_mut().find(|(p, _)| *p == phase) {
            Some(e) => e.1 += dt,
            None => st.spent.push((phase, dt)),
        }
    });
}
