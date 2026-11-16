#[derive(Debug, Clone, PartialEq, Default)]
pub struct HostState {
    pub parts: Vec<(Vec<usize>, Vec<f32>)>,
}

impl HostState {
    pub fn batch(&self) -> usize {
        self.parts.first().map_or(0, |(s, _)| s[0])
    }
}
