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
        let mut spatial = vec![0f32; self.spatial_len];
        let mut loc = vec![false; self.loc_len];
        self.restore_into(&mut spatial, &mut loc);
        o.obs.spatial = spatial;
        o.masks.loc_class_mask = loc;
        o
    }

    /// The planes decompressed into `spatial` ([`Planes::spatial_len`]) and `loc_mask`
    /// ([`Planes::loc_len`]), e.g. a row of a collated batch: streamed through a small buffer, no
    /// plane-sized temporary.
    pub fn restore_into(&self, spatial: &mut [f32], loc_mask: &mut [bool]) {
        assert!(spatial.len() == self.spatial_len && loc_mask.len() == self.loc_len, "plane sizes");
        let mut dec = zstd::stream::read::Decoder::with_buffer(&self.spatial[..]).expect("zstd spatial");
        let mut buf = [0u8; 1 << 16];
        for out in spatial.chunks_mut(buf.len() / 4) {
            let bytes = &mut buf[..out.len() * 4];
            std::io::Read::read_exact(&mut dec, bytes).expect("zstd spatial");
            for (v, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let loc = zstd::bulk::decompress(&self.loc_mask, self.loc_len).expect("zstd loc mask");
        for (v, b) in loc_mask.iter_mut().zip(loc) {
            *v = b != 0;
        }
    }

    pub fn spatial_len(&self) -> usize {
        self.spatial_len
    }

    pub fn loc_len(&self) -> usize {
        self.loc_len
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
