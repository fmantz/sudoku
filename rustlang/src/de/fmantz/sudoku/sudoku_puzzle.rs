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
use crate::sudoku_constants::{PUZZLE_SIZE, CELL_COUNT};
use crate::sudoku_constants::SQUARE_SIZE;

pub trait SudokuPuzzle {
    fn new() -> Self;
    fn get(&self, row: usize, col: usize) -> u8;
    fn set(&mut self, row: usize, col: usize, value: u8) -> ();
    fn init(&mut self) -> ();
    fn is_solvable(&self) -> bool;
    fn is_solved(&self) -> bool;
    fn solve(&mut self) -> ();
    fn to_pretty_string(&self) -> String;
    fn to_string(&self) -> String;
}

pub struct SudokuPuzzleData {
    my_is_solvable: bool,
    my_is_solved: bool,
    puzzle: [u8; CELL_COUNT],
    puzzle_sorted: [u8; CELL_COUNT],
    indices: [usize; CELL_COUNT],
    col_nums: [u16; PUZZLE_SIZE],
    row_nums: [u16; PUZZLE_SIZE],
    square_nums: [u16; PUZZLE_SIZE],
}

impl SudokuPuzzle for SudokuPuzzleData {

    fn new() -> Self {
        SudokuPuzzleData {
            my_is_solvable: true,
            my_is_solved: false,
            puzzle: [0; CELL_COUNT],
            puzzle_sorted: [0; CELL_COUNT],
            indices: [0; CELL_COUNT],
            col_nums: [0; PUZZLE_SIZE],
            row_nums: [0; PUZZLE_SIZE],
            square_nums: [0; PUZZLE_SIZE],
        }
    }

    fn get(&self, row: usize, col: usize) -> u8 {
       self.puzzle[SudokuPuzzleData::get_single_array_index(row, col)]
    }

    fn set(&mut self, row: usize, col: usize, value: u8) -> () {
        self.puzzle[SudokuPuzzleData::get_single_array_index(row, col)] = value;
    }

    fn is_solved(&self) -> bool {
        return self.my_is_solved;
    }

    fn is_solvable(&self) -> bool {
        return self.my_is_solvable;
    }

    fn init(&mut self) -> () {
        //TODO
    }

    /**
     * solves the sudoku by a simple non-recursive backtracking algorithm (brute force)
     * (own simple solution, its an algorithm which may be ported to CUDA or OpenCL)
     * to get a faster result use e.g. https://github.com/Emerentius/sudoku
     */
    fn solve(&mut self) -> () {

    }

    fn to_string(&self) -> String {
        let mut buffer: Vec<String> = Vec::new();
        for row in 0..PUZZLE_SIZE {
            let from = row * PUZZLE_SIZE;
            let until = from + PUZZLE_SIZE;
            let current_row: &[u8] = &self.puzzle[from .. until];
            buffer.push(std::str::from_utf8(current_row).unwrap().to_string());
        }
        return buffer.join("\n");
    }

    fn to_pretty_string(&self) -> String {
        let dotted_line: String = (0..(PUZZLE_SIZE * 3 + SQUARE_SIZE - 1)).map(|_| "-").collect::<String>();
        let empty = "*";
        let mut buffer: Vec<String> = Vec::new();
        for row in 0..PUZZLE_SIZE {
            let from = row * PUZZLE_SIZE;
            let until = from + PUZZLE_SIZE;
            let current_row = &self.puzzle[from .. until];
            let mut formatted_row: String = String::with_capacity(PUZZLE_SIZE);
            for col in 0..PUZZLE_SIZE {
                let col_value: u8 = current_row[col];
                let rs: String = if col_value == 0 { format!(" {} ", empty) } else { format!(" {} ", col_value) };
                formatted_row.push_str(&rs);
                if col + 1 < PUZZLE_SIZE && col % SQUARE_SIZE == 2 {
                    formatted_row.push_str("|");
                }
            }
            buffer.push(formatted_row);
            if row < (PUZZLE_SIZE - 1) && (row + 1) % SQUARE_SIZE == 0 {
                buffer.push(dotted_line.clone());
            }
        }
        return buffer.join("\n");
    }
}

impl SudokuPuzzleData {

    fn get_single_array_index(row: usize, col: usize) -> usize {
        row * PUZZLE_SIZE + col
    }

}