//! Lossless compression of an observation's two large planes (spatial features and location
//! masks, mostly zero): ~1.9 MB per step raw, a batch is 2,048 steps. RL trajectories (042) and
//! BC batches on hosts that ask for it (041-R18) keep them this way until the update.

use uranium_env::features::Observation;

/// The spatial features and location masks of one observation, zstd-compressed.
#[derive(Clone)]
pub struct Planes {
    spatial: Vec<u8>,
    loc_mask: Vec<u8>,
    spatial_len: usize,
    loc_len: usize,
}

impl Planes {
    /// Moves the planes out of `o` (left empty) into compressed form.
    pub fn take(o: &mut Observation) -> Self {
        let spatial_bytes: Vec<u8> = o.obs.spatial.iter().flat_map(|f| f.to_le_bytes()).collect();
        let loc_bytes: Vec<u8> = o.masks.loc_class_mask.iter().map(|&b| b as u8).collect();
        let (spatial_len, loc_len) = (o.obs.spatial.len(), o.masks.loc_class_mask.len());
        o.obs.spatial = Vec::new();
        o.masks.loc_class_mask = Vec::new();
        Self {
            spatial: zstd::bulk::compress(&spatial_bytes, 1).expect("zstd compress"),
            loc_mask: zstd::bulk::compress(&loc_bytes, 1).expect("zstd compress"),
            spatial_len,
            loc_len,
        }
    }

    /// `o` (whose planes were taken) with its planes back, bit for bit.
    pub fn restore(&self, o: &Observation) -> Observation {
        let mut o = o.clone();
        // Streamed through a small buffer into the f32 plane: no second plane-sized allocation.
        let mut dec = zstd::stream::read::Decoder::with_buffer(&self.spatial[..]).expect("zstd spatial");
        let mut spatial = Vec::with_capacity(self.spatial_len);
        let mut buf = [0u8; 1 << 16];
        while spatial.len() < self.spatial_len {
            let n = ((self.spatial_len - spatial.len()) * 4).min(buf.len());
            std::io::Read::read_exact(&mut dec, &mut buf[..n]).expect("zstd spatial");
            spatial.extend(buf[..n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        o.obs.spatial = spatial;
        o.masks.loc_class_mask = zstd::bulk::decompress(&self.loc_mask, self.loc_len).expect("zstd loc mask").into_iter().map(|b| b != 0).collect();
        o
    }

    pub fn bytes(&self) -> usize {
        self.spatial.len() + self.loc_mask.len()
    }
}

/// An observation with its two large planes compressed.
#[derive(Clone)]
pub struct PackedObs {
    obs: Observation,
    planes: Planes,
}

impl PackedObs {
    pub fn pack(o: &Observation) -> Self {
        let mut obs = o.clone();
        let planes = Planes::take(&mut obs);
        Self { obs, planes }
    }

    pub fn unpack(&self) -> Observation {
        self.planes.restore(&self.obs)
    }

    pub fn bytes(&self) -> usize {
        self.planes.bytes() + self.obs.obs.entity_cont.len() * 4 + self.obs.masks.unit_cap.len()
    }
}
