//! Slow reference engine — clarity first, correctness above all.
//! Authority: official Plan B Games rulebook (plan stage 0 §5 conclusions).
//!
//! Key rulebook points implemented here (see docs/worklog.md "规则口径快查"):
//! - Game ends AFTER the full round (wall-tiling + floor scoring) in which
//!   someone completes a wall row. Bonuses then awarded to all players.
//! - Complete pattern lines: rightmost tile -> wall (score immediately),
//!   REMAINING tiles of that line -> lid. INCOMPLETE lines keep their tiles.
//! - Floor: penalty per occupied slot {1,1,2,2,2,3,3}; the starter marker
//!   occupies a slot and is penalised like a tile but kept by the player.
//! - Score never drops below 0.
//! - Scoring a placed wall tile: +1 if no orth. adjacency, else
//!   (horizontal run len) + (vertical run len), counting the new tile twice
//!   (once per direction).
//! - Bonuses: +2/complete row, +7/complete column, +10/full colour.
//! - Tie-break: score > completed rows > shared victory.
//! - 2p: 5 factories, bag 100 (20 per colour); bag empties -> lid refills bag.

use crate::rng::{draw_tile, Rnd};
use crate::types::*;

pub const LID_EXTRA_TO_LID: bool = true; // document intent of overflow handling

/// Rulebook placement constraints for pattern line `row` and `colour`:
/// 1. if the line already holds tiles, they must be the same colour;
/// 2. the corresponding wall row must NOT already hold that colour;
/// 3. the line must not be full.
#[inline]
pub fn line_accepts(p: &Player, row: usize, colour: Colour) -> bool {
    let (cur, count) = p.lines[row];
    if count == (row + 1) as u8 {
        return false; // full
    }
    if count > 0 && cur != Some(colour) {
        return false; // mismatched colour already in line
    }
    if p.wall[row] & (1 << colour) != 0 {
        return false; // wall row already has this colour
    }
    true
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloorItem {
    Tile(Colour),
    Marker,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub score: i32,
    /// wall[row] bit c set = colour `c` placed on that wall row.
    pub wall: [u8; PATTERN_LINES],
    /// pattern line i: (colour if any, count) with count <= i+1.
    pub lines: [(Option<Colour>, u8); PATTERN_LINES],
    pub floor: [Option<FloorItem>; FLOOR_CAPACITY],
    pub floor_len: usize,
    /// Does the player hold the starter marker in front of them?
    pub holds_marker: bool,
}

impl Player {
    pub fn new() -> Self {
        Player {
            score: 0,
            wall: [0; PATTERN_LINES],
            lines: [(None, 0); PATTERN_LINES],
            floor: [None; FLOOR_CAPACITY],
            floor_len: 0,
            holds_marker: false,
        }
    }

    pub fn completed_rows(&self) -> u32 {
        self.wall.iter().filter(|&&w| w == WALL_ROW_FULL).count() as u32
    }

    fn completed_cols(&self) -> u32 {
        (0..NUM_COLOURS)
            .filter(|&c| (0..PATTERN_LINES).all(|r| wall_filled_at(&self.wall, r, c)))
            .count() as u32
    }

    fn full_colours(&self) -> u32 {
        (0..NUM_COLOURS)
            .filter(|&c| (0..PATTERN_LINES).all(|r| self.wall[r] & (1 << c) != 0))
            .count() as u32
    }
}

#[derive(Clone, Debug)]
pub struct SlowState {
    pub bag: [u8; NUM_COLOURS],
    pub lid: [u8; NUM_COLOURS],
    pub centre: [u8; NUM_COLOURS],
    pub marker_in_centre: bool,
    pub factories: [[u8; NUM_COLOURS]; NUM_FACTORIES],
    pub players: [Player; 2],
    pub turn: PlayerId,
    pub marker_owner: PlayerId,
    pub ended: bool,
}

impl SlowState {
    /// New game from a seed. Setup per rulebook: bag shuffled (implicit in
    /// draws), each factory filled with 4 tiles, marker owner starts.
    pub fn new(seed: u64) -> Self {
        let mut st = SlowState {
            bag: [TILES_PER_COLOUR; NUM_COLOURS],
            lid: [0; NUM_COLOURS],
            centre: [0; NUM_COLOURS],
            marker_in_centre: true,
            factories: [[0; NUM_COLOURS]; NUM_FACTORIES],
            players: [Player::new(), Player::new()],
            turn: 0,
            marker_owner: 0,
            ended: false,
        };
        let mut rng = Rnd::new(seed);
        st.fill_factories(&mut rng);
        st
    }

    /// Round-start factory refill, drawing through the canonical RNG path.
    fn fill_factories(&mut self, rng: &mut Rnd) {
        for f in 0..NUM_FACTORIES {
            for _ in 0..TILES_PER_FACTORY {
                if let Some(c) = draw_tile(&mut self.bag, &mut self.lid, rng) {
                    self.factories[f][c as usize] += 1;
                }
            }
        }
    }

    // ------------------------------------------------------------------ moves

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::with_capacity(40);
        if self.ended {
            return out;
        }
        let p = &self.players[self.turn];
        let push_source = |src: Source, counts: &[u8; NUM_COLOURS], out: &mut Vec<Move>| {
            for c in 0..NUM_COLOURS {
                if counts[c] == 0 {
                    continue;
                }
                // Floor is always allowed.
                out.push(Move { source: src, colour: c as Colour, dest: Dest::Floor });
                for row in 0..PATTERN_LINES {
                    if line_accepts(p, row, c as Colour) {
                        out.push(Move { source: src, colour: c as Colour, dest: Dest::Row(row) });
                    }
                }
            }
        };
        for f in 0..NUM_FACTORIES {
            push_source(Source::Factory(f), &self.factories[f], &mut out);
        }
        push_source(Source::Centre, &self.centre, &mut out);
        out
    }

    /// Apply one player's turn. Handles round end (wall-tiling, floor scoring),
    /// game end (bonuses, tie-break) and next-round preparation internally.
    /// The turn then passes to the opponent (or to the marker owner on a new round).
    pub fn apply(&mut self, m: Move, rng: &mut Rnd) {
        assert!(!self.ended, "apply on ended game");
        let me = self.turn;

        // 1. Take tiles from source.
        let taken = self.take_tiles(m.source, m.colour);

        // 2. First take from centre in a round also takes the marker.
        if m.source == Source::Centre && self.marker_in_centre {
            self.marker_in_centre = false;
            self.give_marker(me);
        }

        // 3. Place tiles.
        self.place_tiles(me, m.colour, taken, m.dest);

        // 4. Round end? Centre and all factories must be empty of tiles.
        let mut new_round = false;
        if self.centre_empty() && self.factories_empty() {
            self.wall_tiling_and_floor();
            if self.players.iter().any(|p| p.completed_rows() > 0) {
                self.ended = true;
                // Bonuses are awarded to everyone (rulebook "End of the game").
                for pi in 0..2 {
                    let bonus = Self::end_bonus(&self.players[pi]);
                    self.players[pi].score += bonus;
                }
            } else {
                self.prepare_next_round(rng);
                new_round = true;
            }
        }

        // 5. Pass the turn. (On a fresh round, prepare_next_round already set
        //    turn = marker_owner; otherwise it alternates.)
        if self.ended {
            return;
        }
        if !new_round {
            self.turn = 1 - me;
        }
    }

    fn take_tiles(&mut self, src: Source, colour: Colour) -> u8 {
        let c = colour as usize;
        match src {
            Source::Factory(f) => {
                let n = self.factories[f][c];
                self.factories[f][c] = 0;
                // Remainder of the display goes to the centre.
                for oc in 0..NUM_COLOURS {
                    self.centre[oc] += self.factories[f][oc];
                    self.factories[f][oc] = 0;
                }
                n
            }
            Source::Centre => {
                let n = self.centre[c];
                self.centre[c] = 0;
                n
            }
        }
    }

    fn give_marker(&mut self, me: PlayerId) {
        let p = &mut self.players[me];
        p.holds_marker = true;
        // Marker occupies the leftmost free floor slot (rulebook).
        if p.floor_len < FLOOR_CAPACITY {
            p.floor[p.floor_len] = Some(FloorItem::Marker);
            p.floor_len += 1;
        }
        self.marker_owner = me;
    }

    fn place_tiles(&mut self, me: PlayerId, colour: Colour, taken: u8, dest: Dest) {
        let mut remaining = taken;
        match dest {
            Dest::Row(row) => {
                let slots_free = (row + 1) as u8 - self.players[me].lines[row].1;
                let placed = remaining.min(slots_free);
                let line = &mut self.players[me].lines[row];
                line.0 = Some(colour);
                line.1 += placed;
                remaining -= placed;
            }
            Dest::Floor => {}
        }
        // Excess (or all, for Dest::Floor) to the floor line; overflow to lid.
        for _ in 0..remaining {
            self.push_floor(me, FloorItem::Tile(colour));
        }
    }

    fn push_floor(&mut self, me: PlayerId, item: FloorItem) {
        let p = &mut self.players[me];
        if p.floor_len < FLOOR_CAPACITY {
            p.floor[p.floor_len] = Some(item);
            p.floor_len += 1;
        } else {
            // Rulebook: further fallen tiles go to the lid.
            debug_assert!(LID_EXTRA_TO_LID);
            if let FloorItem::Tile(c) = item {
                self.lid[c as usize] += 1;
            }
        }
    }

    fn centre_empty(&self) -> bool {
        self.centre.iter().all(|&n| n == 0)
    }

    fn factories_empty(&self) -> bool {
        self.factories.iter().flatten().all(|&n| n == 0)
    }

    /// Wall-tiling for both players, then floor scoring, then cleanup.
    fn wall_tiling_and_floor(&mut self) {
        for me in 0..2 {
            // A) complete lines, top to bottom: rightmost tile -> wall, score,
            //    remaining tiles of that line -> lid.
            for row in 0..PATTERN_LINES {
                let (colour, count) = self.players[me].lines[row];
                if count == (row + 1) as u8 {
                    let c = colour.expect("complete line has a colour");
                    self.players[me].lines[row] = (None, 0);
                    // leftovers of the complete line -> lid (count-1 tiles)
                    self.lid[c as usize] += count - 1;
                    self.place_on_wall(me, row, c);
                }
            }
            // Floor scoring: penalty per occupied slot, never below 0.
            let mut penalty = 0i32;
            for i in 0..self.players[me].floor_len {
                penalty += FLOOR_PENALTY[i];
            }
            let p = &mut self.players[me];
            p.score = (p.score + penalty).max(0);
            // Cleanup: tiles -> lid, marker -> kept by player.
            for i in 0..p.floor_len {
                match p.floor[i].take() {
                    Some(FloorItem::Tile(c)) => self.lid[c as usize] += 1,
                    Some(FloorItem::Marker) => p.holds_marker = true,
                    None => {}
                }
            }
            p.floor_len = 0;
        }
        // NOTE: incomplete pattern lines keep their tiles (rulebook).
    }

    pub(crate) fn place_on_wall(&mut self, me: PlayerId, row: usize, colour: Colour) {
        let col = wall_col_for_colour(row, colour);
        let w = self.players[me].wall; // copy (tiny) to avoid borrow conflicts
        let filled = |wall: &[u8; PATTERN_LINES], r: usize, c: usize| wall_filled_at(wall, r, c);

        // Adjacency & run lengths in WALL COORDINATES (col = position in row).
        let h_neighbours = (col > 0 && filled(&w, row, col - 1))
            || (col + 1 < NUM_COLOURS && filled(&w, row, col + 1));
        let v_neighbours = (row > 0 && filled(&w, row - 1, col))
            || (row + 1 < PATTERN_LINES && filled(&w, row + 1, col));

        let mut gained = 1i32;
        if h_neighbours || v_neighbours {
            // Rulebook: count linked tiles per direction ONLY if that direction
            // has links; each run counts the new tile (h and v each include it).
            gained = 0;
            if h_neighbours {
                let mut h = 1i32;
                let mut c = col;
                while c > 0 && filled(&w, row, c - 1) {
                    h += 1;
                    c -= 1;
                }
                let mut c = col;
                while c + 1 < NUM_COLOURS && filled(&w, row, c + 1) {
                    h += 1;
                    c += 1;
                }
                gained += h;
            }
            if v_neighbours {
                let mut v = 1i32;
                let mut r = row;
                while r > 0 && filled(&w, r - 1, col) {
                    v += 1;
                    r -= 1;
                }
                let mut r = row;
                while r + 1 < PATTERN_LINES && filled(&w, r + 1, col) {
                    v += 1;
                    r += 1;
                }
                gained += v;
            }
        }
        self.players[me].wall[row] |= 1 << colour;
        self.players[me].score += gained;
    }

    fn prepare_next_round(&mut self, rng: &mut Rnd) {
        // Rulebook: bag empty -> refill from lid; both empty -> partial factories ok.
        self.fill_factories(rng);
        self.marker_in_centre = true;
        self.turn = self.marker_owner;
    }

    // ------------------------------------------------------------- accessors

    pub fn result(&self) -> Option<GameResult> {
        if !self.ended {
            return None;
        }
        let s0 = self.players[0].score;
        let s1 = self.players[1].score;
        if s0 > s1 {
            Some(GameResult::Win(0))
        } else if s1 > s0 {
            Some(GameResult::Win(1))
        } else {
            let r0 = self.players[0].completed_rows();
            let r1 = self.players[1].completed_rows();
            if r0 > r1 {
                Some(GameResult::Win(0))
            } else if r1 > r0 {
                Some(GameResult::Win(1))
            } else {
                Some(GameResult::Draw)
            }
        }
    }

    /// End-of-game bonuses (rulebook): +2/row, +7/column, +10/full colour.
    pub fn end_bonus(p: &Player) -> i32 {
        (p.completed_rows() as i32) * 2
            + (p.completed_cols() as i32) * 7
            + (p.full_colours() as i32) * 10
    }
}

#[cfg(test)]
mod tests;
