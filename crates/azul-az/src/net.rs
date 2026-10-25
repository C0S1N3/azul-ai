//! policy+value 双头网络的 Rust 推理（AlphaZero 轨道，plan stage 5）。
//!
//! 结构：
//!   trunk  : 171 -> 256 -> 256，层间 ReLU
//!   policy : 256 -> 180，输出 logits（不做 softmax，训练侧对 logits 做
//!            交叉熵；MCTS 里再做先验归一化）
//!   value  : 256 -> 1，tanh，输出 ∈ (-1, 1)，"当前待走方"视角的胜负值
//!
//! 观察向量复用 `azul_nnue::features::extract`（171 维，side-to-move 视角）。
//!
//! # 二进制布局（小端，Python 训练侧 train/train_az.py 必须严格对齐）
//!
//! ```text
//! [u32 n_dims = 2]
//! [u32 in_dim = 171][u32 trunk_hidden = 256]
//! [u32 policy_out = 180][u32 value_out = 1]
//! trunk 层 0 : [f32 × (trunk_hidden * in_dim) 权重][f32 × trunk_hidden 偏置]
//! trunk 层 1 : [f32 × (trunk_hidden * trunk_hidden) 权重][f32 × trunk_hidden 偏置]
//! policy 头  : [f32 × (policy_out * trunk_hidden) 权重][f32 × policy_out 偏置]
//! value 头   : [f32 × (value_out * trunk_hidden) 权重][f32 × value_out 偏置]
//! ```
//!
//! 权重均为 row-major (out, in)：`w[j * fan_in + i]` = 输出 j、输入 i 的权重，
//! 与 PyTorch `nn.Linear` 的 `weight` (out, in) 布局一致，可直接 tobytes 写出。
//! 头部顺序固定为 policy 在前、value 在后。

use azul_core::fast::FastState;
use azul_nnue::features::FEATURE_DIM;
use crate::action::ACTION_DIM;

/// trunk 隐藏层宽度（两层同宽）。
pub const TRUNK_HIDDEN: usize = 256;
/// value 头输出维度（单标量）。
pub const VALUE_OUT: usize = 1;

/// policy+value 双头网络（纯 f32，无量化）。
#[derive(Clone, Debug)]
pub struct PvNet {
    /// [in_dim, trunk_hidden, trunk_hidden]，固定 3 段 trunk。
    pub dims: [usize; 3],
    /// trunk 权重（2 层），row-major (out, in)。
    pub trunk_w: Vec<Vec<f32>>,
    /// trunk 偏置（2 层）。
    pub trunk_b: Vec<Vec<f32>>,
    /// policy 头权重 [policy_out * trunk_hidden]，row-major (out, in)。
    pub policy_w: Vec<f32>,
    /// policy 头偏置 [policy_out]。
    pub policy_b: Vec<f32>,
    /// value 头权重 [trunk_hidden]（单输出，row-major 退化为一维）。
    pub value_w: Vec<f32>,
    /// value 头偏置 [1]。
    pub value_b: Vec<f32>,
}

impl PvNet {
    /// 默认网络：171 -> 256 -> 256 trunk + policy 180 + value 1。
    pub fn new() -> Self {
        PvNet::with_dims(FEATURE_DIM, TRUNK_HIDDEN, TRUNK_HIDDEN)
    }

    /// 指定 trunk 隐藏层宽度构建（in 固定 FEATURE_DIM=171，两层同宽）。
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

    /// 确定性随机初始化（Xavier 风格，与 azul_nnue::net::Mlp::init_random 同源）。
    /// 用 `seed` 驱动 azul_core::rng::Rnd，逐层按 fan_in 缩放。
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

    /// 前向：输入 171 维观察向量，返回 (policy logits[180], value ∈ (-1,1))。
    /// value 为 tanh 输出，从"当前待走方"视角。
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

    /// 观察向量（side-to-move 视角）+ 局面前向的便捷封装。
    pub fn forward_state(&self, st: &FastState) -> ([f32; ACTION_DIM], f32) {
        self.forward(&azul_nnue::features::extract(st))
    }

    /// 序列化为小端字节流（布局见模块文档）。trunk 隐藏层宽度编码在 dims 里。
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

    /// 反序列化 `to_bytes` 的输出。任何不一致（维度、长度、越界）返回 None。
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

    /// 序列化往返恒等：to_bytes -> from_bytes -> to_bytes 逐字节一致。
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

    /// 随机初始化确定性：同 seed 逐字节一致，异 seed 不同。
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

    /// value 输出在 tanh 范围内，logits 有限。
    #[test]
    fn net_output_ranges() {
        let mut net = PvNet::new();
        net.init_random(1);
        let (logits, v) = net.forward(&[0.0f32; FEATURE_DIM]);
        assert!(v.abs() <= 1.0);
        assert!(logits.iter().all(|l| l.is_finite()));
    }
}
