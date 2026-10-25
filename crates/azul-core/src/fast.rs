//! Fast bitpacked engine — the performance implementation, verified against
//! `slow` by differential tests (plan stage 0 §2 / M0-d).
//!
//! Packing scheme (Rzepecki-style compact state; hot fields only):
//! - `factories[f]`: u32, 3 bits per colour (max 4 tiles), offset c*3
//! - `centre`:       u32, 5 bits per colour (0..20), offset c*5
//! - player `wall`:  u32, 5 bits per row = colour-bitmask, offset row*5
//! - player `lines`: u32, 6 bits per line = 3b colour(+1) | 3b count, offset line*6
//! - player `floor`: u32, 3 bits per slot (0=empty,1..=5=colour+1,6=marker), left-packed
//! Bag/lid stay as [u8;5] arrays — only touched at round boundaries (cold).
//!
//! Move generation writes into a caller-provided buffer (no allocation).
//! Hash: state split into 16-bit chunks, xored through a static Zobrist table.

use crate::rng::{draw_tile, Rnd};
use crate::slow::{FloorItem, SlowState};
use crate::types::*;

/// Upper bound on legal moves: 6 sources (5 factories + centre) × 5 colours ×
/// 6 destinations (5 rows + floor) = 180. Matches the garden 180 action space.
pub const MAX_MOVES: usize = 180;

#[derive(Clone, Copy, Debug)]
pub struct MoveBuf {
    pub moves: [MaybeMove; MAX_MOVES],
    pub len: usize,
}

impl Default for MoveBuf {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MaybeMove {
    pub inner: Option<Move>,
}

impl MoveBuf {
    pub fn new() -> Self {
        MoveBuf { moves: [MaybeMove { inner: None }; MAX_MOVES], len: 0 }
    }
    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES, "move buffer overflow");
        if self.len < MAX_MOVES {
            self.moves[self.len].inner = Some(m);
            self.len += 1;
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().filter_map(|m| m.inner)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FastPlayer {
    pub score: i32,
    pub wall: u32,
    pub lines: u32,
    pub floor: u32,
    pub floor_len: u8,
    pub holds_marker: bool,
}

impl FastPlayer {
    pub fn new() -> Self {
        FastPlayer { score: 0, wall: 0, lines: 0, floor: 0, floor_len: 0, holds_marker: false }
    }

    #[inline]
    pub fn line(&self, row: usize) -> (Option<Colour>, u8) {
        let raw = (self.lines >> (row * 6)) & 0b11_1111;
        let colour = ((raw >> 3) & 0b111) as u8;
        let count = (raw & 0b111) as u8;
        if count == 0 {
            (None, 0)
        } else {
            (Some(colour - 1), count)
        }
    }

    #[inline]
    pub fn set_line(&mut self, row: usize, colour: Option<Colour>, count: u8) {
        let raw = if count == 0 {
            0
        } else {
            (((colour.unwrap() + 1) as u32) << 3) | count as u32
        };
        self.lines = (self.lines & !(0b11_1111 << (row * 6))) | (raw << (row * 6));
    }

    #[inline]
    pub fn wall_has(&self, row: usize, colour: Colour) -> bool {
        (self.wall >> (row * 5)) & (1 << colour) != 0
    }

    #[inline]
    pub fn wall_set(&mut self, row: usize, colour: Colour) {
        self.wall |= 1u32 << (row * 5 + colour as usize);
    }

    #[inline]
    pub fn floor_slot(&self, i: usize) -> Option<FloorItem> {
        let v = (self.floor >> (i * 3)) & 0b111;
        match v {
            0 => None,
            1..=5 => Some(FloorItem::Tile((v - 1) as Colour)),
            6 => Some(FloorItem::Marker),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn floor_push(&mut self, item: FloorItem) {
        let v = match item {
            FloorItem::Tile(c) => (c + 1) as u32,
            FloorItem::Marker => 6,
        };
        self.floor |= v << (self.floor_len as usize * 3);
        self.floor_len += 1;
    }

    pub fn completed_rows(&self) -> u32 {
        (0..PATTERN_LINES)
            .filter(|&r| (self.wall >> (r * 5)) & 0b1_1111 == WALL_ROW_FULL as u32)
            .count() as u32
    }

    /// Complete wall COLUMN c = every cell (r, c) filled; the cell holds the
    /// colour `wall_colour(r, c)`, so check that colour-bit per row.
    pub fn completed_cols(&self) -> u32 {
        (0..NUM_COLOURS)
            .filter(|&c| {
                (0..PATTERN_LINES).all(|r| {
                    (self.wall >> (r * 5)) & (1u32 << wall_colour(r, c)) != 0
                })
            })
            .count() as u32
    }

    /// Full colour = all 5 tiles of colour c placed (one per row in the fixed
    /// layout), i.e. colour-bit c set in every row.
    pub fn full_colours(&self) -> u32 {
        (0..NUM_COLOURS)
            .filter(|&c| (0..PATTERN_LINES).all(|r| self.wall_has(r, c as Colour)))
            .count() as u32
    }

    pub fn end_bonus(&self) -> i32 {
        self.completed_rows() as i32 * 2 + self.completed_cols() as i32 * 7 + self.full_colours() as i32 * 10
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FastState {
    pub factories: [u32; NUM_FACTORIES],
    pub centre: u32,
    pub bag: [u8; NUM_COLOURS],
    pub lid: [u8; NUM_COLOURS],
    pub marker_in_centre: bool,
    pub marker_owner: u8,
    pub turn: u8,
    pub ended: bool,
    pub players: [FastPlayer; 2],
}

impl FastState {
    pub fn new(seed: u64) -> Self {
        let mut st = FastState {
            factories: [0; NUM_FACTORIES],
            centre: 0,
            bag: [TILES_PER_COLOUR; NUM_COLOURS],
            lid: [0; NUM_COLOURS],
            marker_in_centre: true,
            marker_owner: 0,
            turn: 0,
            ended: false,
            players: [FastPlayer::new(), FastPlayer::new()],
        };
        let mut rng = Rnd::new(seed);
        st.fill_factories(&mut rng);
        st
    }

    fn fill_factories(&mut self, rng: &mut Rnd) {
        for f in 0..NUM_FACTORIES {
            for _ in 0..TILES_PER_FACTORY {
                if let Some(c) = draw_tile(&mut self.bag, &mut self.lid, rng) {
                    self.factories[f] += 1 << (c as usize * 3);
                }
            }
        }
    }

    #[inline]
    pub fn factory_count(&self, f: usize, c: Colour) -> u8 {
        ((self.factories[f] >> (c as usize * 3)) & 0b111) as u8
    }

    #[inline]
    pub fn centre_count(&self, c: Colour) -> u8 {
        ((self.centre >> (c as usize * 5)) & 0b1_1111) as u8
    }

    #[inline]
    fn centre_add(&mut self, c: Colour, n: u8) {
        self.centre += (n as u32) << (c as usize * 5);
    }

    pub fn legal_into(&self, buf: &mut MoveBuf) {
        *buf = MoveBuf::new();
        if self.ended {
            return;
        }
        let p = &self.players[self.turn as usize];
        for f in 0..NUM_FACTORIES {
            for c in 0..NUM_COLOURS {
                if self.factory_count(f, c as Colour) > 0 {
                    self.push_dests(buf, p, Source::Factory(f), c as Colour);
                }
            }
        }
        for c in 0..NUM_COLOURS {
            if self.centre_count(c as Colour) > 0 {
                self.push_dests(buf, p, Source::Centre, c as Colour);
            }
        }
    }

    fn push_dests(&self, buf: &mut MoveBuf, p: &FastPlayer, src: Source, colour: Colour) {
        buf.push(Move { source: src, colour, dest: Dest::Floor });
        for row in 0..PATTERN_LINES {
            if line_accepts(p, row, colour) {
                buf.push(Move { source: src, colour, dest: Dest::Row(row) });
            }
        }
    }

    pub fn apply(&mut self, m: Move, rng: &mut Rnd) {
        debug_assert!(!self.ended);
        let me = self.turn as usize;

        // 1. Take tiles.
        let taken = match m.source {
            Source::Factory(f) => {
                let n = self.factory_count(f, m.colour);
                self.factories[f] &= !(0b111 << (m.colour as usize * 3));
                // Sweep remainder into the centre.
                for c in 0..NUM_COLOURS {
                    let t = self.factory_count(f, c as Colour);
                    if t > 0 {
                        self.centre_add(c as Colour, t);
                    }
                }
                self.factories[f] = 0;
                n
            }
            Source::Centre => {
                let n = self.centre_count(m.colour);
                self.centre &= !(0b1_1111 << (m.colour as usize * 5));
                n
            }
        };

        // 2. Marker from the centre.
        if m.source == Source::Centre && self.marker_in_centre {
            self.marker_in_centre = false;
            self.players[me].holds_marker = true;
            self.floor_push_or_lid(me, 6); // 6 = marker slot code
            self.marker_owner = me as u8;
        }

        // 3. Place.
        let mut remaining = taken;
        if let Dest::Row(row) = m.dest {
            let (_, count) = self.players[me].line(row);
            let free = (row + 1) as u8 - count;
            let placed = remaining.min(free);
            self.players[me].set_line(row, Some(m.colour), count + placed);
            remaining -= placed;
        }
        for _ in 0..remaining {
            self.floor_push_or_lid(me, m.colour + 1);
        }

        // 4. Round end?
        let mut new_round = false;
        if self.centre == 0 && self.factories.iter().all(|&f| f == 0) {
            self.wall_tiling_and_floor();
            if self.players.iter().any(|p| p.completed_rows() > 0) {
                self.ended = true;
                for pi in 0..2 {
                    let b = self.players[pi].end_bonus();
                    self.players[pi].score += b;
                }
            } else {
                self.fill_factories(rng);
                self.marker_in_centre = true;
                self.turn = self.marker_owner;
                new_round = true;
            }
        }

        if !self.ended && !new_round {
            self.turn = 1 - self.turn;
        }
    }

    fn floor_push_or_lid(&mut self, me: usize, code: u8) {
        // code: 0 unused, 1..=5 = colour+1, 6 = marker. Overflow -> lid (tiles
        // only; the marker keeps its slot semantics but also can't overflow).
        if (self.players[me].floor_len as usize) < FLOOR_CAPACITY {
            let item = if code == 6 { FloorItem::Marker } else { FloorItem::Tile(code - 1) };
            self.players[me].floor_push(item);
        } else if code != 6 {
            self.lid[(code - 1) as usize] += 1;
        }
    }

    fn wall_tiling_and_floor(&mut self) {
        for me in 0..2 {
            for row in 0..PATTERN_LINES {
                let (colour, count) = self.players[me].line(row);
                if count == (row + 1) as u8 {
                    let c = colour.expect("complete line has colour");
                    self.players[me].set_line(row, None, 0);
                    self.lid[c as usize] += count - 1;
                    self.place_on_wall(me, row, c);
                }
            }
            let mut penalty = 0i32;
            for i in 0..self.players[me].floor_len as usize {
                penalty += FLOOR_PENALTY[i];
            }
            let p = &mut self.players[me];
            p.score = (p.score + penalty).max(0);
            for i in 0..p.floor_len as usize {
                match p.floor_slot(i) {
                    Some(FloorItem::Tile(c)) => self.lid[c as usize] += 1,
                    Some(FloorItem::Marker) => p.holds_marker = true,
                    None => {}
                }
            }
            p.floor = 0;
            p.floor_len = 0;
        }
    }

    fn place_on_wall(&mut self, me: usize, row: usize, colour: Colour) {
        let col = wall_col_for_colour(row, colour);
        let w = self.players[me].wall;
        let filled = |r: usize, c: usize| -> bool {
            let colour_at = wall_colour(r, c);
            (w >> (r * 5)) & (1 << colour_at) != 0
        };
        let h_adj = (col > 0 && filled(row, col - 1)) || (col + 1 < NUM_COLOURS && filled(row, col + 1));
        let v_adj = (row > 0 && filled(row - 1, col)) || (row + 1 < PATTERN_LINES && filled(row + 1, col));

        let mut gained = 1i32;
        if h_adj || v_adj {
            // Rulebook: count a direction's run ONLY if that direction has links.
            gained = 0;
            if h_adj {
                let mut h = 1i32;
                let mut c = col;
                while c > 0 && filled(row, c - 1) {
                    h += 1;
                    c -= 1;
                }
                let mut c = col;
                while c + 1 < NUM_COLOURS && filled(row, c + 1) {
                    h += 1;
                    c += 1;
                }
                gained += h;
            }
            if v_adj {
                let mut v = 1i32;
                let mut r = row;
                while r > 0 && filled(r - 1, col) {
                    v += 1;
                    r -= 1;
                }
                let mut r = row;
                while r + 1 < PATTERN_LINES && filled(r + 1, col) {
                    v += 1;
                    r += 1;
                }
                gained += v;
            }
        }
        self.players[me].wall_set(row, colour);
        self.players[me].score += gained;
    }

    pub fn result(&self) -> Option<GameResult> {
        if !self.ended {
            return None;
        }
        let (s0, s1) = (self.players[0].score, self.players[1].score);
        if s0 > s1 {
            Some(GameResult::Win(0))
        } else if s1 > s0 {
            Some(GameResult::Win(1))
        } else {
            let (r0, r1) = (self.players[0].completed_rows(), self.players[1].completed_rows());
            if r0 > r1 {
                Some(GameResult::Win(0))
            } else if r1 > r0 {
                Some(GameResult::Win(1))
            } else {
                Some(GameResult::Draw)
            }
        }
    }

    /// Zobrist-style hash: state -> 16-bit chunks -> static table xor.
    pub fn hash(&self) -> u64 {
        let mut all = [0u64; 5];
        all[0] = self.factories[0] as u64 | ((self.factories[1] as u64) << 32);
        all[1] = self.factories[2] as u64 | ((self.factories[3] as u64) << 32);
        all[2] = self.factories[4] as u64 | ((self.centre as u64) << 32);
        all[3] = (self.players[0].wall as u64)
            | ((self.players[0].lines as u64) << 32)
            | ((self.players[0].floor as u64) << 24);
        all[4] = (self.players[1].wall as u64)
            | ((self.players[1].lines as u64) << 32)
            | ((self.players[1].floor as u64) << 24);
        let mut key = 0u64;
        for (i, w) in all.iter().enumerate() {
            for s in 0..4 {
                let chunk = ((w >> (s * 16)) & 0xFFFF) as u64;
                key ^= mix(chunk ^ ((i as u64) << 8) ^ (s as u64));
            }
        }
        // Bag + lid contents MUST be in the key: they decide future refills, so
        // omitting them made TT entries collide across different bag states.
        let bag_packed = self.bag.iter().enumerate().fold(0u64, |a, (i, &n)| a | ((n as u64) << (i * 8)));
        let lid_packed = self.lid.iter().enumerate().fold(0u64, |a, (i, &n)| a | ((n as u64) << (i * 8)));
        key ^= mix(bag_packed ^ 0xA5A5_A5A5_A5A5_A5A5);
        key ^= mix(lid_packed ^ 0x5A5A_5A5A_5A5A_5A5A);
        key ^= mix(self.turn as u64 | ((self.marker_owner as u64) << 8));
        key ^= mix(self.marker_in_centre as u64 | ((self.ended as u64) << 1));
        key
    }
}

/// splitmix64 finalizer — cheap, well-mixed 64-bit hash (Zobrist substitute).
#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Placement constraints (mirrors slow::line_accepts, independent implementation).
#[inline]
pub fn line_accepts(p: &FastPlayer, row: usize, colour: Colour) -> bool {
    let (cur, count) = p.line(row);
    if count == (row + 1) as u8 {
        return false;
    }
    if count > 0 && cur != Some(colour) {
        return false;
    }
    if p.wall_has(row, colour) {
        return false;
    }
    true
}

/// Fast<->Slow conversion for differential tests.
impl From<&SlowState> for FastState {
    fn from(s: &SlowState) -> Self {
        let mut fs = FastState {
            factories: [0; NUM_FACTORIES],
            centre: 0,
            bag: s.bag,
            lid: s.lid,
            marker_in_centre: s.marker_in_centre,
            marker_owner: s.marker_owner as u8,
            turn: s.turn as u8,
            ended: s.ended,
            players: [FastPlayer::new(); 2],
        };
        for f in 0..NUM_FACTORIES {
            for c in 0..NUM_COLOURS {
                fs.factories[f] += (s.factories[f][c] as u32) << (c * 3);
            }
        }
        for c in 0..NUM_COLOURS {
            fs.centre += (s.centre[c] as u32) << (c * 5);
        }
        for pi in 0..2 {
            let sp = &s.players[pi];
            let fp = &mut fs.players[pi];
            fp.score = sp.score;
            for r in 0..PATTERN_LINES {
                if sp.wall[r] != 0 {
                    fp.wall |= (sp.wall[r] as u32) << (r * 5);
                }
                let (col, cnt) = sp.lines[r];
                fp.set_line(r, col, cnt);
            }
            for i in 0..sp.floor_len {
                if let Some(item) = sp.floor[i] {
                    fp.floor_push(item);
                }
            }
            fp.holds_marker = sp.holds_marker;
        }
        fs
    }
}
