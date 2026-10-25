//! Shared types and rule constants (rulebook-authoritative).

pub const NUM_COLOURS: usize = 5;
/// 2-player game uses 5 factory displays (rulebook).
pub const NUM_FACTORIES: usize = 5;
pub const TILES_PER_FACTORY: usize = 4;
/// Bag: 20 tiles of each colour, 100 total.
pub const TILES_PER_COLOUR: u8 = 20;
/// Pattern line i has i+1 slots (line 0 = 1 slot ... line 4 = 5 slots).
pub const PATTERN_LINES: usize = 5;
/// Floor line capacity; penalties per occupied slot, left to right (rulebook).
pub const FLOOR_PENALTY: [i32; 7] = [-1, -1, -2, -2, -2, -3, -3];
pub const FLOOR_CAPACITY: usize = 7;

pub type Colour = u8; // 0..NUM_COLOURS
pub type PlayerId = usize; // 0 | 1

/// Wall colour layout: fixed arrangement, wall slot (row, col) expects colour
/// `(col + NUM_COLOURS - row) % NUM_COLOURS`  (equivalently (col - row) mod 5).
/// Row 0 = 0 1 2 3 4, row 1 = 4 0 1 2 3, ...
#[inline]
pub fn wall_colour(row: usize, col: usize) -> Colour {
    ((col + NUM_COLOURS - row) % NUM_COLOURS) as Colour
}

/// Column on the wall row where a given colour must be placed.
#[inline]
pub fn wall_col_for_colour(row: usize, colour: Colour) -> usize {
    (colour as usize + row) % NUM_COLOURS
}

/// Is the wall cell (row, col) filled, given `wall[row]` = colour-bitmask?
#[inline]
pub fn wall_filled_at(wall: &[u8; PATTERN_LINES], row: usize, col: usize) -> bool {
    wall[row] & (1 << wall_colour(row, col)) != 0
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Factory(usize), // 0..NUM_FACTORIES
    Centre,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dest {
    /// Pattern line 0..PATTERN_LINES (line i holds i+1 tiles).
    Row(usize),
    /// Floor line (excess or deliberate discard).
    Floor,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub source: Source,
    pub colour: Colour,
    pub dest: Dest,
}

/// End-of-game result from player 0's perspective? No — from absolute view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    /// Winning player (strictly more points after tie-breaks).
    Win(PlayerId),
    /// Shared victory (rulebook: score tie -> more completed rows -> shared).
    Draw,
}

impl GameResult {
    /// Value from `p`'s perspective for training targets {-1, 0, +1}.
    pub fn value_for(&self, p: PlayerId) -> i8 {
        match self {
            GameResult::Win(w) if *w == p => 1,
            GameResult::Win(_) => -1,
            GameResult::Draw => 0,
        }
    }
}

/// Rows on the wall are "complete" when all 5 colours are placed (0b11111).
pub const WALL_ROW_FULL: u8 = 0b1_1111;
