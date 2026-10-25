//! azul-wasm: browser play export layer.
//!
//! Single-threaded wasm, no wasm-bindgen — JS calls extern "C" functions
//! directly:
//! `game_new` / `game_json`(+`game_json_len`) / `game_play` / `game_ai`.
//! Action encoding `src*30 + colour*6 + dest`: src 0..4=factory, 5=centre;
//! dest 0..4=pattern line, 5=floor line. JSON output lands in a global buffer;
//! JS reads memory via ptr+len.

use std::cell::RefCell;
use std::sync::OnceLock;

use azul_az::mcts::{Mcts, MctsConfig};
use azul_az::net::PvNet;
use azul_core::fast::{FastState, MoveBuf};
use azul_core::rng::Rnd;
use azul_core::slow::FloorItem;
use azul_core::types::*;

/// The champion net is compiled straight into the wasm (~625KB).
const NET_BYTES: &[u8] = include_bytes!("../../../out/az_net7.bin");

fn net() -> &'static PvNet {
    static NET: OnceLock<PvNet> = OnceLock::new();
    NET.get_or_init(|| PvNet::from_bytes(NET_BYTES).expect("az_net7.bin parse"))
}

thread_local! {
    static STATE: RefCell<FastState> = RefCell::new(FastState::new(0));
    static RNG: RefCell<Rnd> = RefCell::new(Rnd::new(0x5EED));
    static OUT: RefCell<String> = RefCell::new(String::new());
}

fn pack(src: u8, colour: u8, dest: u8) -> u32 {
    (src as u32) * 30 + (colour as u32) * 6 + (dest as u32)
}

/// Start a new game. The seed is split into two u32s (JS numbers only have
/// 2^53 of precision, so receiving a u64 directly is unreliable).
/// With `first`=0 player 0 (human) moves first; with =1 player 1 (AI) moves
/// first.
#[no_mangle]
pub extern "C" fn game_new(seed_lo: u32, seed_hi: u32, first: u32) {
    let seed = ((seed_hi as u64) << 32) | seed_lo as u64;
    STATE.with(|s| {
        let mut st = FastState::new(seed);
        st.turn = if first != 0 { 1 } else { 0 };
        *s.borrow_mut() = st;
    });
    RNG.with(|r| *r.borrow_mut() = Rnd::new(seed ^ 0x9E37_79B9));
}

/// Play a move. Returns 0=success, 1=illegal, 2=game already over.
#[no_mangle]
pub extern "C" fn game_play(src: u8, colour: u8, dest: u8) -> u32 {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        if st.ended {
            return 2;
        }
        let m = Move {
            source: if src >= 5 { Source::Centre } else { Source::Factory(src as usize) },
            colour,
            dest: if dest >= 5 { Dest::Floor } else { Dest::Row(dest as usize) },
        };
        // 合法性校验（防 JS 侧乱传）
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        if !buf.iter().any(|x| x == m) {
            return 1;
        }
        RNG.with(|r| st.apply(m, &mut r.borrow_mut()));
        0
    })
}

/// AI move: MCTS(champion net). Returns the packed action;
/// 0xFFFF_FFFF=no move (game over).
#[no_mangle]
pub extern "C" fn game_ai(sims: u32) -> u32 {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        if st.ended {
            return 0xFFFF_FFFF;
        }
        let cfg = MctsConfig {
            sims: sims.max(16),
            dirichlet: false,
            ..Default::default()
        };
        let mut mcts = Mcts::new(net(), cfg);
        let policy = mcts.search(&st, 0x5EED);
        let mut buf = MoveBuf::new();
        st.legal_into(&mut buf);
        let best = buf
            .iter()
            .map(|m| (policy[azul_az::to_index(&m)], m))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, m)| m);
        match best {
            Some(m) => {
                let src = match m.source {
                    Source::Factory(f) => f as u8,
                    Source::Centre => 5,
                };
                let dest = match m.dest {
                    Dest::Row(r) => r as u8,
                    Dest::Floor => 5,
                };
                RNG.with(|r| st.apply(m, &mut r.borrow_mut()));
                pack(src, m.colour, dest)
            }
            None => 0xFFFF_FFFF,
        }
    })
}

/// Write the current state JSON into the global buffer and return the pointer;
/// get the length with game_json_len().
#[no_mangle]
pub extern "C" fn game_json() -> *const u8 {
    let json = STATE.with(|s| state_json(&s.borrow()));
    OUT.with(|o| {
        let mut o = o.borrow_mut();
        o.clear();
        o.push_str(&json);
        o.as_ptr()
    })
}

#[no_mangle]
pub extern "C" fn game_json_len() -> u32 {
    OUT.with(|o| o.borrow().len() as u32)
}

fn state_json(st: &FastState) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("{\"factories\":[");
    for f in 0..NUM_FACTORIES {
        if f > 0 {
            s.push(',');
        }
        s.push('[');
        for c in 0..NUM_COLOURS {
            if c > 0 {
                s.push(',');
            }
            s.push_str(&st.factory_count(f, c as Colour).to_string());
        }
        s.push(']');
    }
    s.push_str("],\"centre\":[");
    for c in 0..NUM_COLOURS {
        if c > 0 {
            s.push(',');
        }
        s.push_str(&st.centre_count(c as Colour).to_string());
    }
    s.push_str("],\"marker_in_centre\":");
    s.push_str(if st.marker_in_centre { "true" } else { "false" });
    s.push_str(",\"turn\":");
    s.push_str(&st.turn.to_string());
    s.push_str(",\"ended\":");
    s.push_str(if st.ended { "true" } else { "false" });

    // result: null | 0(玩家0胜) | 1 | -1(平)
    s.push_str(",\"result\":");
    match st.result() {
        Some(GameResult::Win(p)) => s.push_str(&p.to_string()),
        Some(GameResult::Draw) => s.push_str("-1"),
        None => s.push_str("null"),
    }

    s.push_str(",\"players\":[");
    for p in 0..2 {
        if p > 0 {
            s.push(',');
        }
        let pl = &st.players[p];
        s.push_str("{\"score\":");
        s.push_str(&pl.score.to_string());
        // 案子行
        s.push_str(",\"lines\":[");
        for r in 0..PATTERN_LINES {
            if r > 0 {
                s.push(',');
            }
            let (col, cnt) = pl.line(r);
            match col {
                Some(c) => {
                    s.push('[');
                    s.push_str(&c.to_string());
                    s.push(',');
                    s.push_str(&cnt.to_string());
                    s.push(']');
                }
                None => s.push_str("[0,0]"),
            }
        }
        // 墙：输出 [r][c] 0/1（c=墙列，固定色 wall_colour(r,c)）
        s.push_str("],\"wall\":[");
        for r in 0..PATTERN_LINES {
            if r > 0 {
                s.push(',');
            }
            s.push('[');
            for c in 0..PATTERN_LINES {
                if c > 0 {
                    s.push(',');
                }
                s.push(if pl.wall_has(r, wall_colour(r, c)) { '1' } else { '0' });
            }
            s.push(']');
        }
        // 罚分条
        s.push_str("],\"floor\":[");
        for i in 0..FLOOR_CAPACITY {
            if i > 0 {
                s.push(',');
            }
            match pl.floor_slot(i) {
                Some(FloorItem::Marker) => s.push('7'), // 7=先手标记，0..4=花色
                Some(FloorItem::Tile(c)) => s.push_str(&(c + 1).to_string()),
                None => s.push('0'),
            }
        }
        s.push_str("],\"holds_marker\":");
        s.push_str(if pl.holds_marker { "true" } else { "false" });
        s.push('}');
    }
    s.push_str("],\"legal\":[");
    let mut buf = MoveBuf::new();
    st.legal_into(&mut buf);
    let mut first = true;
    for m in buf.iter() {
        if !first {
            s.push(',');
        }
        first = false;
        let src = match m.source {
            Source::Factory(f) => f as u8,
            Source::Centre => 5,
        };
        let dest = match m.dest {
            Dest::Row(r) => r as u8,
            Dest::Floor => 5,
        };
        s.push_str(&pack(src, m.colour, dest).to_string());
    }
    s.push_str("]}");
    s
}
