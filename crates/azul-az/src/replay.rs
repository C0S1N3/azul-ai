//! Replay buffer (plan stage 5, minimal for this phase).
//!
//! Current design: the single batch of data produced directly by self-play is
//! the dataset (train/train_az.py reads the binary file directly); no
//! buffering/sampling is done.
//!
//! TODO (garden lesson: "no replay buffer" failed — but this phase first
//! validates the pipeline end-to-end with a single batch of data, and adds
//! these later):
//!   - Mixed buffer of self-play data across network generations (sampling by
//!     game/generation to avoid overfitting to the latest weights)
//!   - Priority / balanced sampling (by game outcome, by value-target
//!     distribution)
//!   - Multi-file disk index + random batch reads

use crate::selfplay::{Sample, RECORD_BYTES};
use azul_nnue::features::FEATURE_DIM;
use crate::action::ACTION_DIM;

/// Single-batch dataset: an in-memory mirror of the self-play output file.
/// The only data carrier in this phase.
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Load from a self-play binary file (validates record alignment).
    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() % RECORD_BYTES != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: {} bytes not a multiple of {}", path, bytes.len(), RECORD_BYTES),
            ));
        }
        let mut samples = Vec::with_capacity(bytes.len() / RECORD_BYTES);
        let mut i = 0usize;
        let read_f32 = |i: &mut usize| -> f32 {
            let v = f32::from_le_bytes(bytes[*i..*i + 4].try_into().expect("slice"));
            *i += 4;
            v
        };
        while i < bytes.len() {
            let mut obs = [0.0f32; FEATURE_DIM];
            for v in obs.iter_mut() {
                *v = read_f32(&mut i);
            }
            let mut policy = [0.0f32; ACTION_DIM];
            for v in policy.iter_mut() {
                *v = read_f32(&mut i);
            }
            let value = read_f32(&mut i);
            samples.push(Sample { obs, policy, value });
        }
        Ok(Dataset { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::PvNet;
    use crate::selfplay::{play_game, SelfplayConfig};

    /// Self-play output -> Dataset::load round trip is consistent.
    #[test]
    fn dataset_roundtrip() {
        let mut net = PvNet::new();
        net.init_random(0xC1);
        let cfg = SelfplayConfig { sims: 4, temp_cutoff: 2, ..Default::default() };
        let samples = play_game(&net, &cfg, 5);
        let path = "target/az_test_dataset.bin";
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut bytes = Vec::new();
        for s in &samples {
            s.write(&mut bytes);
        }
        std::fs::write(path, &bytes).unwrap();
        let ds = Dataset::load(path).expect("load");
        assert_eq!(ds.len(), samples.len());
        for (a, b) in ds.samples.iter().zip(samples.iter()) {
            assert_eq!(a.obs, b.obs);
            assert_eq!(a.policy, b.policy);
            assert_eq!(a.value, b.value);
        }
    }
}
