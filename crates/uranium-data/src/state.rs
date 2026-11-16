#[derive(Debug, Clone, PartialEq, Default)]
pub struct HostState {
    pub parts: Vec<(Vec<usize>, Vec<f32>)>,
}

impl HostState {
    pub fn batch(&self) -> usize {
        self.parts.first().map_or(0, |(s, _)| s[0])
    }

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
