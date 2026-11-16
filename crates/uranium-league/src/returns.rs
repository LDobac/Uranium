pub struct Seq<'a> {
    pub rewards: &'a [f32],
    pub values: &'a [f32],
    pub log_rho: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VTrace {
    pub vs: Vec<f32>,
    pub pg_adv: Vec<f32>,
    pub rho: Vec<f32>,
}

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

    #[test]
    fn vtrace_on_policy_is_td_lambda_042() {
        let rewards = [0.0, 1.0, 0.0, -1.0];
        let values = [0.5, 0.2, -0.1, 0.3, 0.0];
        let (gamma, lambda) = (1.0, 0.8);
        let v = vtrace(&Seq { rewards: &rewards, values: &values, log_rho: &[0.0; 4] }, gamma, lambda, 1.0, 1.0);
        let mut g = [0.0f32; 5];
        g[4] = values[4];
        for i in (0..4).rev() {
            g[i] = rewards[i] + gamma * ((1.0 - lambda) * values[i + 1] + lambda * g[i + 1]);
        }
        close(&v.vs, &g[..4]);
        let adv: Vec<f32> = (0..4).map(|i| rewards[i] + gamma * if i + 1 < 4 { v.vs[i + 1] } else { values[4] } - values[i]).collect();
        close(&v.pg_adv, &adv);
    }

    #[test]
    fn vtrace_off_policy_by_hand_042() {
        let s = Seq { rewards: &[1.0, 0.0], values: &[0.0, 0.5, 2.0], log_rho: &[2f32.ln(), 0.5f32.ln()] };
        let v = vtrace(&s, 1.0, 1.0, 1.0, 1.0);
        close(&v.vs, &[2.25, 1.25]);
        close(&v.rho, &[1.0, 0.5]);
        close(&v.pg_adv, &[2.25, 0.75]);
    }

    #[test]
    fn upgo_by_hand_042() {
        let s = Seq { rewards: &[0.0, 0.0, 1.0], values: &[0.5, 1.0, 0.2, 0.0], log_rho: &[0.0; 3] };
        let u = upgo(&s, 1.0, 1.0);
        close(&u, &[1.0 - 0.5, 1.0 - 1.0, 1.0 - 0.2]);
    }

    #[test]
    fn gae_by_hand_042() {
        let s = Seq { rewards: &[1.0, 0.0], values: &[0.0, 0.5, 2.0], log_rho: &[0.0; 2] };
        let (adv, ret) = gae(&s, 1.0, 0.5);
        close(&adv, &[2.25, 1.5]);
        close(&ret, &[2.25, 2.0]);
    }
}
