//! A Core state on the host (032-R16, 042 core_state0), shared by the inference server, the RL
//! trajectories and the learner whatever framework runs the model (AMB-000-4).

/// Every state tensor as f32 values with its shape (bools as 0/1). Row 0 of every shape is the
/// batch.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HostState {
    pub parts: Vec<(Vec<usize>, Vec<f32>)>,
}

impl HostState {
    pub fn batch(&self) -> usize {
        self.parts.first().map_or(0, |(s, _)| s[0])
    }

    /// Stacks single-batch (or any batch) states along the batch dimension.
    pub fn cat(rows: &[&HostState]) -> HostState {
        let Some(first) = rows.first() else { return HostState::default() };
        let parts = (0..first.parts.len())
            .map(|i| {
                let mut shape = first.parts[i].0.clone();
                shape[0] = rows.iter().map(|r| r.parts[i].0[0]).sum();
                let data = rows.iter().flat_map(|r| r.parts[i].1.iter().copied()).collect();
                (shape, data)
            })
            .collect();
        HostState { parts }
    }
}
