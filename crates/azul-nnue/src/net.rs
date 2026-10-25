//! NNUE value-net inference (plan stage 2 §2.1): fully-connected value-only
//! net, ReLU between layers. Output is a **sigmoid** ∈ [0,1] trained with BCE
//! against {loss=0, draw=0.5, win=1} — the Rzepecki §5.3.3 winning recipe
//! (BCE loss + sigmoid output + linearly mapped targets). No ClippedReLU
//! (it destroys score information).
//!
//! Architecture: FEATURE_DIM -> H1 -> H2 -> H3 -> 1 (default 64-128-16-16-1).
//! Weights are f32; int8 quantisation + AVX2 SIMD is a later optimisation
//! (plan stage 2 §deploy, stage 4). Keep this forward pass simple & correct.

use azul_core::fast::FastState;

/// A dense MLP value net. `dims[0]` = input dim, `dims[last]` = 1.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub dims: Vec<usize>,
    /// Layer weights: `layers[l]` has `dims[l+1] * dims[l]` values (row-major).
    pub weights: Vec<Vec<f32>>,
    /// Layer biases: `biases[l]` has `dims[l+1]` values.
    pub biases: Vec<Vec<f32>>,
}

impl Mlp {
    /// New net with the given layer dimensions (input .. output).
    pub fn new(dims: &[usize]) -> Self {
        assert!(dims.len() >= 2, "need at least input and output dims");
        assert_eq!(*dims.last().unwrap(), 1, "value net must have 1 output");
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for w in 0..dims.len() - 1 {
            let (fan_in, fan_out) = (dims[w], dims[w + 1]);
            weights.push(vec![0.0; fan_in * fan_out]);
            biases.push(vec![0.0; fan_out]);
        }
        Mlp { dims: dims.to_vec(), weights, biases }
    }

    /// Xavier-ish init from a deterministic RNG.
    pub fn init_random(&mut self, seed: u64) {
        use azul_core::rng::Rnd;
        let mut rng = Rnd::new(seed);
        for l in 0..self.weights.len() {
            let fan_in = self.dims[l];
            let scale = (2.0 / fan_in as f32).sqrt();
            for w in self.weights[l].iter_mut() {
                let u = (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 - 1.0;
                *w = u * scale;
            }
            for b in self.biases[l].iter_mut() {
                *b = 0.0;
            }
        }
    }

    /// Forward pass. ReLU on hidden layers, sigmoid on the single output
    /// (probability of side-to-move winning, with 0.5 = draw-ish).
    pub fn forward(&self, input: &[f32]) -> f32 {
        let mut cur = input.to_vec();
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let (fan_in, fan_out) = (self.dims[l], self.dims[l + 1]);
            let mut next = vec![0.0f32; fan_out];
            for j in 0..fan_out {
                let mut acc = b[j];
                for i in 0..fan_in {
                    acc += cur[i] * w[j * fan_in + i];
                }
                next[j] = acc;
            }
            let is_last = l == self.weights.len() - 1;
            for v in next.iter_mut() {
                *v = if is_last {
                    1.0 / (1.0 + (-*v).exp()) // sigmoid ∈ (0,1), matches BCE training
                } else {
                    v.max(0.0) // ReLU (NOT ClippedReLU)
                };
            }
            cur = next;
        }
        cur[0]
    }

    /// Serialize to a compact little-endian byte buffer (f32 weights).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.dims.len() as u32).to_le_bytes());
        for d in &self.dims {
            out.extend_from_slice(&(*d as u32).to_le_bytes());
        }
        for l in 0..self.weights.len() {
            for v in &self.weights[l] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in &self.biases[l] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Deserialize from `to_bytes` output.
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
        let n_dims = read_u32(&mut i)? as usize;
        if n_dims < 2 {
            return None;
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u32(&mut i)? as usize);
        }
        let mut net = Mlp::new(&dims);
        for l in 0..net.weights.len() {
            for w in net.weights[l].iter_mut() {
                *w = read_f32(&mut i)?;
            }
            for b in net.biases[l].iter_mut() {
                *b = read_f32(&mut i)?;
            }
        }
        Some(net)
    }
}

/// Evaluator adapter: evaluates a position with a trained Mlp value net.
/// Raw output is a win-probability sigmoid ∈ (0,1); for minimax we map it to
/// [-1,1] (0.5 = draw) and scale to score-ish units by `scale`.
pub struct NnueEvaluator {
    pub net: Mlp,
    pub scale: f32,
}

impl NnueEvaluator {
    pub fn new(net: Mlp, scale: f32) -> Self {
        NnueEvaluator { net, scale }
    }

    /// Raw sigmoid output ∈ (0,1).
    pub fn value(&self, st: &FastState) -> f32 {
        let f = crate::features::extract(st);
        self.net.forward(&f)
    }
}

impl azul_search::eval::Evaluator for NnueEvaluator {
    fn eval(&self, st: &FastState) -> f32 {
        // Map sigmoid∈(0,1) to roughly score-difference units so it composes
        // sanely with search values (which are in points). scale ≈ typical
        // points swing; 25.0 matches the training target mapping (value/25).
        (self.value(st) * 2.0 - 1.0) * self.scale
    }
}

/// Hybrid evaluator (2026-10-14, worklog v2 analysis): h0 settle-score diff at
/// FULL strength (in points) plus an α-scaled NNUE correction. With α=0 this
/// is exactly h0 — the net only supplies the residual it learned, instead of
/// having to reproduce h0's strength from noisy outcome labels.
pub struct HybridEvaluator {
    pub nn: NnueEvaluator,
    pub alpha: f32,
}

impl HybridEvaluator {
    pub fn new(net: Mlp, scale: f32, alpha: f32) -> Self {
        HybridEvaluator { nn: NnueEvaluator::new(net, scale), alpha }
    }
}

impl azul_search::eval::Evaluator for HybridEvaluator {
    fn eval(&self, st: &FastState) -> f32 {
        use azul_search::eval::Evaluator as _;
        azul_search::eval::Heuristic0.eval(st) + self.alpha * self.nn.eval(st)
    }
}
