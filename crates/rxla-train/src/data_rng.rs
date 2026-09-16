//! Schedule-independent random streams for reproducible input pipelines.

/// Root seed for deterministic per-sample augmentation streams.
///
/// A draw depends only on the seed, epoch, sample identity, transform stream,
/// and draw index. It therefore does not depend on Rayon worker assignment or
/// completion order.
#[derive(Clone, Copy, Debug)]
pub struct DataRng {
    seed: u64,
}

/// Random view derived for one logical sample in one epoch.
#[derive(Clone, Copy, Debug)]
pub struct SampleRng {
    key: u64,
}

impl DataRng {
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn sample(self, epoch: u64, sample_id: u64) -> SampleRng {
        SampleRng {
            key: mix(self.seed ^ mix(epoch) ^ mix(sample_id)),
        }
    }
}

impl SampleRng {
    /// A reproducible value in `[0, 1)` for one transform stream and draw.
    /// Assign stable stream numbers to transforms so inserting an unrelated
    /// transform does not perturb existing augmentation decisions.
    pub fn uniform(self, stream: u64, draw: u64) -> f32 {
        let bits = mix(self.key ^ mix(stream) ^ mix(draw));
        ((bits >> 40) as f32) * (1.0 / (1_u32 << 24) as f32)
    }

    pub fn bernoulli(self, stream: u64, draw: u64, probability: f32) -> bool {
        debug_assert!((0.0..=1.0).contains(&probability));
        self.uniform(stream, draw) < probability
    }
}

const fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_streams_are_order_independent() {
        let rng = DataRng::new(42);
        let forward = (0..32)
            .map(|sample| rng.sample(3, sample).uniform(7, 0))
            .collect::<Vec<_>>();
        let mut reverse = (0..32)
            .rev()
            .map(|sample| (sample, rng.sample(3, sample).uniform(7, 0)))
            .collect::<Vec<_>>();
        reverse.sort_by_key(|(sample, _)| *sample);
        assert_eq!(
            forward,
            reverse
                .into_iter()
                .map(|(_, value)| value)
                .collect::<Vec<_>>()
        );
        assert_ne!(
            rng.sample(3, 0).uniform(7, 0),
            rng.sample(4, 0).uniform(7, 0)
        );
    }
}
