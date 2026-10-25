//! 动作编码：Azul 着法 <--> 180 维策略动作空间（garden 180 动作空间对齐）。
//!
//! 编码公式：`idx = src * 30 + colour * 6 + dest`
//!   - `src`   ∈ 0..6  ：0-4 = 工厂 Factory(f)，5 = 中央 Centre
//!   - `colour`∈ 0..5  ：砖色 0..4
//!   - `dest`  ∈ 0..6  ：0-4 = 图案行 Row(r)，5 = 地板 Floor
//!
//! 共 6 × 5 × 6 = 180 个动作。训练侧（train/train_az.py）的 policy 头输出
//! 180 个 logits，必须使用同一编码顺序（colour 维最内层、dest 维最内层）。

use azul_core::types::{Dest, Move, Source, NUM_COLOURS, NUM_FACTORIES, PATTERN_LINES};

/// 动作空间大小（garden 180）。
pub const ACTION_DIM: usize = 180;

/// 源数量：5 工厂 + 1 中央。
pub const NUM_SOURCES: usize = NUM_FACTORIES + 1;
/// 目的地数量：5 图案行 + 1 地板。
pub const NUM_DESTS: usize = PATTERN_LINES + 1;

/// 着法 -> 动作索引。非法着法（越界字段）不在此处校验，由调用方保证合法。
#[inline]
pub fn to_index(m: &Move) -> usize {
    let src = match m.source {
        Source::Factory(f) => f,
        Source::Centre => NUM_FACTORIES,
    };
    let dest = match m.dest {
        Dest::Row(r) => r,
        Dest::Floor => PATTERN_LINES,
    };
    src * (NUM_COLOURS * NUM_DESTS) + m.colour as usize * NUM_DESTS + dest
}

/// 动作索引 -> 着法（`to_index` 的逆）。索引必须 < 180。
#[inline]
pub fn from_index(idx: usize) -> Move {
    debug_assert!(idx < ACTION_DIM, "action index out of range: {}", idx);
    let src = idx / (NUM_COLOURS * NUM_DESTS);
    let rest = idx % (NUM_COLOURS * NUM_DESTS);
    let colour = rest / NUM_DESTS;
    let dest = rest % NUM_DESTS;
    Move {
        source: if src < NUM_FACTORIES { Source::Factory(src) } else { Source::Centre },
        colour: colour as u8,
        dest: if dest < PATTERN_LINES { Dest::Row(dest) } else { Dest::Floor },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 往返恒等：index -> Move -> index 必须对全部 180 个索引成立。
    #[test]
    fn index_roundtrip_identity() {
        for idx in 0..ACTION_DIM {
            let m = from_index(idx);
            assert_eq!(to_index(&m), idx, "roundtrip failed at index {}", idx);
        }
    }

    /// Move -> index -> Move 往返恒等（覆盖全部 src/colour/dest 组合）。
    #[test]
    fn move_roundtrip_identity() {
        for src in 0..NUM_SOURCES {
            for colour in 0..NUM_COLOURS {
                for dest in 0..NUM_DESTS {
                    let m = Move {
                        source: if src < NUM_FACTORIES {
                            Source::Factory(src)
                        } else {
                            Source::Centre
                        },
                        colour: colour as u8,
                        dest: if dest < PATTERN_LINES { Dest::Row(dest) } else { Dest::Floor },
                    };
                    assert_eq!(from_index(to_index(&m)), m);
                }
            }
        }
    }
}
