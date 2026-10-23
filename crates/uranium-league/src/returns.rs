//! Return and advantage targets on the host (042): V-trace (IMPALA), UPGO (AlphaStar) and GAE
//! (PPO baseline). One call handles one sequence of one reward component (033-R3); targets carry
//! no gradient, so computing them on the host keeps the recursions plain and testable.

/// One sequence of length T. `values` has T + 1 entries: the target network's value at each
/// step plus the bootstrap value after the last step (0 when the episode ended there).
pub struct Seq<'a> {
    pub rewards: &'a [f32],
    pub values: &'a [f32],
    /// log π(a_t) - log μ(a_t): target minus behaviour log-prob of the taken action.
    pub log_rho: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VTrace {
    /// v_s targets for the value head, length T.
    pub vs: Vec<f32>,
    /// ρ_t (r_t + γ v_{s,t+1} - V_t), length T.
    pub pg_adv: Vec<f32>,
    /// Truncated importance weight min(ρ̄, π/μ), length T.
    pub rho: Vec<f32>,
}

/// V-trace (Espeholt et al. 2018) with λ folded into the trace cut: c_t = λ min(c̄, π/μ).
pub fn vtrace(s: &Seq, gamma: f32, lambda: f32, rho_bar: f32, c_bar: f32) -> VTrace {
    let t = s.rewards.len();
    assert_eq!(s.values.len(), t + 1);
    let ratio: Vec<f32> = s.log_rho.iter().map(|l| l.exp()).collect();
    let rho: Vec<f32> = ratio.iter().map(|r| r.min(rho_bar)).collect();
    let c: Vec<f32> = ratio.iter().map(|r| lambda * r.min(c_bar)).collect();
    let mut vs = vec![0.0; t + 1];
    vs[t] = s.values[t];
    for i in (0..t).rev() {
        let delta = rho[i] * (s.rewards[i] + gamma * s.values[i + 1] - s.values[i]);
        vs[i] = s.values[i] + delta + gamma * c[i] * (vs[i + 1] - s.values[i + 1]);
    }
    let pg_adv = (0..t).map(|i| rho[i] * (s.rewards[i] + gamma * vs[i + 1] - s.values[i])).collect();
    vs.truncate(t);
    VTrace { vs, pg_adv, rho }
}

/// UPGO advantage (Vinyals et al. 2019): the return follows the trajectory while the next action
/// looked at least as good as the value (Q_{t+1} >= V_{t+1}, Q_{t+1} ≈ r_{t+1} + γ V_{t+2}) and
/// bootstraps from V otherwise; times the truncated weight ρ_t.
pub fn upgo(s: &Seq, gamma: f32, rho_bar: f32) -> Vec<f32> {
    let t = s.rewards.len();
    assert_eq!(s.values.len(), t + 1);
    let mut g = vec![0.0; t + 1];
    g[t] = s.values[t];
    for i in (0..t).rev() {
        let next = if i + 1 < t {
            let q_next = s.rewards[i + 1] + gamma * s.values[i + 2];
            if q_next >= s.values[i + 1] { g[i + 1] } else { s.values[i + 1] }
        } else {
            s.values[t]
        };
        g[i] = s.rewards[i] + gamma * next;
    }
    (0..t).map(|i| s.log_rho[i].exp().min(rho_bar) * (g[i] - s.values[i])).collect()
}

/// GAE (Schulman et al. 2016) for the PPO baseline: (advantages, returns), length T each.
pub fn gae(s: &Seq, gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    let t = s.rewards.len();
    assert_eq!(s.values.len(), t + 1);
    let mut adv = vec![0.0; t];
    let mut acc = 0.0;
    for i in (0..t).rev() {
        let delta = s.rewards[i] + gamma * s.values[i + 1] - s.values[i];
        acc = delta + gamma * lambda * acc;
        adv[i] = acc;
    }
    let ret = adv.iter().zip(s.values).map(|(a, v)| a + v).collect();
    (adv, ret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    /// On-policy (ρ = 1) V-trace equals the TD(λ) return.
    #[test]
    fn vtrace_on_policy_is_td_lambda_042() {
        let rewards = [0.0, 1.0, 0.0, -1.0];
        let values = [0.5, 0.2, -0.1, 0.3, 0.0];
        let (gamma, lambda) = (1.0, 0.8);
        let v = vtrace(&Seq { rewards: &rewards, values: &values, log_rho: &[0.0; 4] }, gamma, lambda, 1.0, 1.0);
        // TD(λ): G_t = r_t + γ((1-λ) V_{t+1} + λ G_{t+1}), G_T = V_T.
        let mut g = [0.0f32; 5];
        g[4] = values[4];
        for i in (0..4).rev() {
            g[i] = rewards[i] + gamma * ((1.0 - lambda) * values[i + 1] + lambda * g[i + 1]);
        }
        close(&v.vs, &g[..4]);
        let adv: Vec<f32> = (0..4).map(|i| rewards[i] + gamma * if i + 1 < 4 { v.vs[i + 1] } else { values[4] } - values[i]).collect();
        close(&v.pg_adv, &adv);
    }

    /// Hand-computed off-policy example: T = 2, ρ = (2, 0.5), ρ̄ = c̄ = 1, γ = 1, λ = 1.
    #[test]
    fn vtrace_off_policy_by_hand_042() {
        let s = Seq { rewards: &[1.0, 0.0], values: &[0.0, 0.5, 2.0], log_rho: &[2f32.ln(), 0.5f32.ln()] };
        let v = vtrace(&s, 1.0, 1.0, 1.0, 1.0);
        // t=1: rho=0.5, delta = 0.5*(0 + 2 - 0.5) = 0.75, vs1 = 0.5 + 0.75 = 1.25
        // t=0: rho=1,   delta = 1*(1 + 0.5 - 0) = 1.5, c0 = 1, vs0 = 0 + 1.5 + (1.25 - 0.5) = 2.25
        close(&v.vs, &[2.25, 1.25]);
        close(&v.rho, &[1.0, 0.5]);
        // pg: t0 = 1*(1 + 1.25 - 0) = 2.25, t1 = 0.5*(0 + 2 - 0.5) = 0.75
        close(&v.pg_adv, &[2.25, 0.75]);
    }

    /// UPGO keeps following the trajectory only while the next action beat the value.
    #[test]
    fn upgo_by_hand_042() {
        // T = 3, bootstrap 0. Q_1 = r1 + V2 = 0 + 0.2 = 0.2 < V1 = 1 -> cut at t=1.
        // Q_2 = r2 + V3 = 1 + 0 = 1 >= V2 = 0.2 -> follow at t=2.
        let s = Seq { rewards: &[0.0, 0.0, 1.0], values: &[0.5, 1.0, 0.2, 0.0], log_rho: &[0.0; 3] };
        let u = upgo(&s, 1.0, 1.0);
        // G2 = 1 + V3 = 1; G1 = 0 + G2 = 1 (Q_2 >= V2); G0 = 0 + V1 = 1 (Q_1 < V1).
        close(&u, &[1.0 - 0.5, 1.0 - 1.0, 1.0 - 0.2]);
    }

    #[test]
    fn gae_by_hand_042() {
        let s = Seq { rewards: &[1.0, 0.0], values: &[0.0, 0.5, 2.0], log_rho: &[0.0; 2] };
        let (adv, ret) = gae(&s, 1.0, 0.5);
        // delta1 = 0 + 2 - 0.5 = 1.5; delta0 = 1 + 0.5 - 0 = 1.5; A0 = 1.5 + 0.5*1.5 = 2.25
        close(&adv, &[2.25, 1.5]);
        close(&ret, &[2.25, 2.0]);
    }
}
