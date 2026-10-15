use serde::{Deserialize, Serialize};

/// Row-major bit grid (1 bit per cell, LSB first).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct BitGrid {
    pub width: u16,
    pub height: u16,
    pub bits: Vec<u8>,
}

impl BitGrid {
    pub fn new(width: u16, height: u16) -> Self {
        let n = width as usize * height as usize;
        Self { width, height, bits: vec![0; n.div_ceil(8)] }
    }

    /// Builds from one byte per cell (non-zero = set).
    pub fn from_bytes(width: u16, height: u16, cells: &[u8]) -> Self {
        let mut g = Self::new(width, height);
        for (i, &c) in cells.iter().enumerate().take(width as usize * height as usize) {
            if c != 0 {
                g.bits[i / 8] |= 1 << (i % 8);
            }
        }
        g
    }

    pub fn get(&self, x: u16, y: u16) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let i = y as usize * self.width as usize + x as usize;
        self.bits[i / 8] & (1 << (i % 8)) != 0
    }

    pub fn set(&mut self, x: u16, y: u16, v: bool) {
        assert!(x < self.width && y < self.height, "BitGrid::set out of bounds");
        let i = y as usize * self.width as usize + x as usize;
        if v {
            self.bits[i / 8] |= 1 << (i % 8);
        } else {
            self.bits[i / 8] &= !(1 << (i % 8));
        }
    }

    pub fn count_ones(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }
}
