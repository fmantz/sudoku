/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.md file in the top level folder.
 * Copyright (C) 2020 Florian Mantz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
pub const PUZZLE_SIZE: usize = 9;
pub const SQUARE_SIZE: usize = 3;
pub const NEW_SUDOKU_SEPARATOR: &str = "Grid";
pub const EMPTY_CHAR: char = '0';
pub const QQWING_EMPTY_CHAR: char = '.';
pub const CHECK_BITS: u16 = !0 >> (16 - PUZZLE_SIZE); //binary: Size times "1"
pub const PARALLELIZATION_COUNT: u16 = 1024;
