//! replay buffer（plan stage 5，本期极简）。
//!
//! 当前口径：selfplay 直接产出的单批数据即数据集（train/train_az.py 直接读
//! 二进制文件），不做缓冲/采样。
//!
//! TODO（garden 教训："无 replay buffer"失败——但本期先用单批数据验证管线
//! 通，之后再补）：
//!   - 多代网络自对弈数据的混合缓冲（按局/按代采样，防止对最新权重过拟合）
//!   - 优先级/均衡采样（按对局结果、按 value target 分布）
//!   - 磁盘多文件索引 + 随机 batch 读取

use crate::selfplay::{Sample, RECORD_BYTES};
use azul_nnue::features::FEATURE_DIM;
use crate::action::ACTION_DIM;

/// 单批数据集：selfplay 输出文件的内存镜像。本期唯一的数据载体。
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// 从 selfplay 二进制文件加载（校验记录对齐）。
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

    /// selfplay 输出 -> Dataset::load 往返一致。
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
