//! Rust inference for the policy+value two-headed net (AlphaZero track,
//! plan stage 5).
//!
//! Architecture:
//!   trunk  : 171 -> 256 -> 256, ReLU between layers
//!   policy : 256 -> 180, outputs logits (no softmax; the training side applies
//!            cross-entropy to the logits; MCTS normalises them into priors)
//!   value  : 256 -> 1, tanh, output ∈ (-1, 1), win/loss score from the
//!            "current side to move" perspective
//!
//! The observation vector reuses `azul_nnue::features::extract` (171 dims,
//! side-to-move perspective).
//!
//! # Binary layout
//!
//! Little-endian; the Python training side train/train_az.py must match it
//! exactly.
//!
//! ```text
//! [u32 n_dims = 2]
//! [u32 in_dim = 171][u32 trunk_hidden = 256]
//! [u32 policy_out = 180][u32 value_out = 1]
//! trunk layer 0 : [f32 × (trunk_hidden * in_dim) weights][f32 × trunk_hidden biases]
//! trunk layer 1 : [f32 × (trunk_hidden * trunk_hidden) weights][f32 × trunk_hidden biases]
//! policy head   : [f32 × (policy_out * trunk_hidden) weights][f32 × policy_out biases]
//! value head    : [f32 × (value_out * trunk_hidden) weights][f32 × value_out biases]
//! ```
//!
//! Weights are all row-major (out, in): `w[j * fan_in + i]` = weight for output
//! j and input i, matching the layout of PyTorch `nn.Linear`'s `weight`
//! (out, in), so it can be written out with tobytes directly. The head order is
//! fixed: policy first, then value.

use azul_core::fast::FastState;
use azul_nnue::features::FEATURE_DIM;
use crate::action::ACTION_DIM;

/// Trunk hidden-layer width (both layers share the same width).
pub const TRUNK_HIDDEN: usize = 256;
/// Value-head output dimension (a single scalar).
pub const VALUE_OUT: usize = 1;

/// Policy+value two-headed net (plain f32, no quantisation).
#[derive(Clone, Debug)]
pub struct PvNet {
    /// [in_dim, trunk_hidden, trunk_hidden], a fixed 3-segment trunk.
    pub dims: [usize; 3],
    /// Trunk weights (2 layers), row-major (out, in).
    pub trunk_w: Vec<Vec<f32>>,
    /// Trunk biases (2 layers).
    pub trunk_b: Vec<Vec<f32>>,
    /// Policy-head weights [policy_out * trunk_hidden], row-major (out, in).
    pub policy_w: Vec<f32>,
    /// Policy-head biases [policy_out].
    pub policy_b: Vec<f32>,
    /// Value-head weights [trunk_hidden] (a single output, so row-major
    /// degenerates to one dimension).
    pub value_w: Vec<f32>,
    /// Value-head biases [1].
    pub value_b: Vec<f32>,
}

impl PvNet {
    /// Default net: 171 -> 256 -> 256 trunk + policy 180 + value 1.
    pub fn new() -> Self {
        PvNet::with_dims(FEATURE_DIM, TRUNK_HIDDEN, TRUNK_HIDDEN)
    }

    /// Build with a given trunk hidden-layer width (in is fixed at
    /// FEATURE_DIM=171; both layers share the same width).
    pub fn with_dims(in_dim: usize, h1: usize, h2: usize) -> Self {
        let dims = [in_dim, h1, h2];
        let mut trunk_w = Vec::new();
        let mut trunk_b = Vec::new();
        for l in 0..2 {
            trunk_w.push(vec![0.0; dims[l + 1] * dims[l]]);
            trunk_b.push(vec![0.0; dims[l + 1]]);
        }
        PvNet {
            dims,
            trunk_w,
            trunk_b,
            policy_w: vec![0.0; ACTION_DIM * h2],
            policy_b: vec![0.0; ACTION_DIM],
            value_w: vec![0.0; VALUE_OUT * h2],
            value_b: vec![0.0; VALUE_OUT],
        }
    }

    /// Deterministic random initialisation (Xavier-style, same recipe as
    /// azul_nnue::net::Mlp::init_random). Drives azul_core::rng::Rnd with
    /// `seed` and scales each layer by its fan_in.
    pub fn init_random(&mut self, seed: u64) {
        use azul_core::rng::Rnd;
        let mut rng = Rnd::new(seed);
        let init_layer = |rng: &mut Rnd, fan_in: usize, w: &mut [f32], b: &mut [f32]| {
            let scale = (2.0 / fan_in as f32).sqrt();
            for v in w.iter_mut() {
                let u = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 - 1.0;
                *v = u * scale;
            }
            for v in b.iter_mut() {
                *v = 0.0;
            }
        };
        for l in 0..2 {
            let (w, b) = (&mut self.trunk_w[l], &mut self.trunk_b[l]);
            init_layer(&mut rng, self.dims[l], w, b);
        }
        init_layer(&mut rng, self.dims[2], &mut self.policy_w, &mut self.policy_b);
        init_layer(&mut rng, self.dims[2], &mut self.value_w, &mut self.value_b);
    }

    /// Forward pass: takes a 171-dim observation vector and returns
    /// (policy logits[180], value ∈ (-1,1)). The value is a tanh output from
    /// the "current side to move" perspective.
    pub fn forward(&self, obs: &[f32; FEATURE_DIM]) -> ([f32; ACTION_DIM], f32) {
        let mut cur: Vec<f32> = obs.to_vec();
        for l in 0..2 {
            let fan_in = self.dims[l];
            let fan_out = self.dims[l + 1];
            let mut next = vec![0.0f32; fan_out];
            for j in 0..fan_out {
                let mut acc = self.trunk_b[l][j];
                for i in 0..fan_in {
                    acc += cur[i] * self.trunk_w[l][j * fan_in + i];
                }
                next[j] = acc.max(0.0); // ReLU
            }
            cur = next;
        }
        let h = &cur; // trunk 输出，长度 = dims[2]

        // policy 头：纯 logits。
        let mut logits = [0.0f32; ACTION_DIM];
        for j in 0..ACTION_DIM {
            let mut acc = self.policy_b[j];
            for i in 0..h.len() {
                acc += h[i] * self.policy_w[j * h.len() + i];
            }
            logits[j] = acc;
        }

        // value 头：tanh。
        let mut v = self.value_b[0];
        for i in 0..h.len() {
            v += h[i] * self.value_w[i];
        }
        (logits, v.tanh())
    }

    /// Convenience wrapper: observation vector (side-to-move perspective) +
    /// forward from a position.
    pub fn forward_state(&self, st: &FastState) -> ([f32; ACTION_DIM], f32) {
        self.forward(&azul_nnue::features::extract(st))
    }

    /// Serialise to a little-endian byte stream (see the module docs for the
    /// layout). The trunk hidden-layer width is encoded in dims.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        for d in &self.dims {
            out.extend_from_slice(&(*d as u32).to_le_bytes());
        }
        out.extend_from_slice(&(ACTION_DIM as u32).to_le_bytes()); // policy_out
        out.extend_from_slice(&(VALUE_OUT as u32).to_le_bytes()); // value_out
        for l in 0..2 {
            for v in &self.trunk_w[l] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in &self.trunk_b[l] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in &self.policy_w {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.policy_b {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.value_w {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.value_b {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Deserialise the output of `to_bytes`. Returns None on any inconsistency
    /// (dimensions, lengths, out-of-bounds).
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        let mut i = 0usize;
        let read_u32 = |i: &mut usize| -> Option<u32> {
            if *i + 4 > buf.len() {
                return None;
            }
            let v = u32::from_le_bytes(buf[*i..*i + 4].try_into().ok()?);
            *i += 4;
            Some(v)
        };
        let read_f32 = |i: &mut usize| -> Option<f32> {
            if *i + 4 > buf.len() {
                return None;
            }
            let v = f32::from_le_bytes(buf[*i..*i + 4].try_into().ok()?);
            *i += 4;
            Some(v)
        };
        if read_u32(&mut i)? != 2 {
            return None;
        }
        let d0 = read_u32(&mut i)? as usize;
        let d1 = read_u32(&mut i)? as usize;
        let d2 = read_u32(&mut i)? as usize;
        if d0 != FEATURE_DIM {
            return None;
        }
        let policy_out = read_u32(&mut i)? as usize;
        let value_out = read_u32(&mut i)? as usize;
        if policy_out != ACTION_DIM || value_out != VALUE_OUT {
            return None;
        }
        let mut net = PvNet::with_dims(d0, d1, d2);
        for l in 0..2 {
            for v in net.trunk_w[l].iter_mut() {
                *v = read_f32(&mut i)?;
            }
            for v in net.trunk_b[l].iter_mut() {
                *v = read_f32(&mut i)?;
            }
        }
        for v in net.policy_w.iter_mut() {
            *v = read_f32(&mut i)?;
        }
        for v in net.policy_b.iter_mut() {
            *v = read_f32(&mut i)?;
        }
        for v in net.value_w.iter_mut() {
            *v = read_f32(&mut i)?;
        }
        for v in net.value_b.iter_mut() {
            *v = read_f32(&mut i)?;
        }
        if i != buf.len() {
            return None; // 尾部多余字节视为损坏
        }
        Some(net)
    }
}

impl Default for PvNet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialisation round-trip identity: to_bytes -> from_bytes -> to_bytes is
    /// byte-for-byte identical.
    #[test]
    fn net_serialization_roundtrip() {
        let mut net = PvNet::new();
        net.init_random(0x1234_5678);
        let bytes = net.to_bytes();
        let back = PvNet::from_bytes(&bytes).expect("from_bytes must parse");
        assert_eq!(back.to_bytes(), bytes);
        // 前向一致。
        let obs = [0.5f32; FEATURE_DIM];
        let (l0, v0) = net.forward(&obs);
        let (l1, v1) = back.forward(&obs);
        assert_eq!(l0, l1);
        assert_eq!(v0, v1);
    }

    /// Random initialisation is deterministic: identical seed gives
    /// byte-identical output, different seeds differ.
    #[test]
    fn net_init_deterministic() {
        let mut a = PvNet::new();
        let mut b = PvNet::new();
        a.init_random(7);
        b.init_random(7);
        assert_eq!(a.to_bytes(), b.to_bytes());
        b.init_random(8);
        assert_ne!(a.to_bytes(), b.to_bytes());
    }

    /// Value output stays within the tanh range, and logits are finite.
    #[test]
    fn net_output_ranges() {
        let mut net = PvNet::new();
        net.init_random(1);
        let (logits, v) = net.forward(&[0.0f32; FEATURE_DIM]);
        assert!(v.abs() <= 1.0);
        assert!(logits.iter().all(|l| l.is_finite()));
    }
}
