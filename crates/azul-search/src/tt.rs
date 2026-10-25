//! Transposition table (plan stage 1 §1.1): stores best move (hash move) and
//! value, probed first in move ordering and search.

use azul_core::types::Move;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub enum TtFlag {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
pub struct TtEntry {
    pub depth: u32,
    pub value: f32,
    pub flag: TtFlag,
    pub best: Option<Move>,
}

pub struct TranspositionTable {
    map: HashMap<u64, TtEntry>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        TranspositionTable { map: HashMap::with_capacity(1 << 20) }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn store(&mut self, key: u64, depth: u32, value: f32, flag: TtFlag, best: Option<Move>) {
        // Replace only if at least as deep (simple replacement scheme).
        if let Some(e) = self.map.get(&key) {
            if e.depth > depth {
                return;
            }
        }
        self.map.insert(key, TtEntry { depth, value, flag, best });
    }

    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        self.map.get(&key).copied()
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}
