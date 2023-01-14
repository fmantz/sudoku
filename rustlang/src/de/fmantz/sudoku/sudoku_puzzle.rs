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
use std::fmt::{Display, Formatter};
use crate::sudoku_constants::SQUARE_SIZE;
use crate::sudoku_constants::{BITSET_ARRAY, BITSET_LENGTH, CELL_COUNT, PUZZLE_SIZE};

pub struct SudokuPuzzle {
    my_is_solvable: bool,
    my_is_solved: bool,
    puzzle: [u8; CELL_COUNT],
}

impl SudokuPuzzle {
    pub fn new() -> Self {
        SudokuPuzzle {
            my_is_solvable: true,
            my_is_solved: false,
            puzzle: [0; CELL_COUNT],
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.puzzle[SudokuPuzzle::get_single_array_index(row, col)]
    }

    pub fn set(&mut self, row: usize, col: usize, value: u8) {
        self.puzzle[SudokuPuzzle::get_single_array_index(row, col)] = value;
    }

    fn get_single_array_index(row: usize, col: usize) -> usize {
        row * PUZZLE_SIZE + col
    }

    /**
     * solves the sudoku by a simple non-recursive backtracking algorithm (brute force)
     * (own simple solution, its an algorithm which may be ported to CUDA or OpenCL)
     * to get a faster result use e.g. https://github.com/Emerentius/sudoku
     */
    pub fn solve(&mut self) -> bool {
        // Early out:
        if !self.my_is_solvable || self.my_is_solved {
            return self.my_is_solved;
        }

        // Temporary memory to compute solution:
        let mut puzzle_sorted: [u8; CELL_COUNT] = [0; CELL_COUNT];
        let mut indices: [u8; CELL_COUNT] = [0; CELL_COUNT];
        let mut row_nums: [u16; PUZZLE_SIZE] = [0; PUZZLE_SIZE];
        let mut col_nums: [u16; PUZZLE_SIZE] = [0; PUZZLE_SIZE];
        let mut square_nums: [u16; PUZZLE_SIZE] = [0; PUZZLE_SIZE];

        self.find_all_possible_values_for_each_empty_cell(
            &mut row_nums,
            &mut col_nums,
            &mut square_nums,
        );
        self.prepare_puzzle_for_solving(
            &mut puzzle_sorted,
            &mut indices,
            &mut row_nums,
            &mut col_nums,
            &mut square_nums,
        );

        if self.my_is_solvable && !self.my_is_solved {
            self.find_solution_non_recursively(
                &mut puzzle_sorted,
                &indices,
                &mut row_nums,
                &mut col_nums,
                &mut square_nums,
            );
        }

        self.my_is_solved
    }

    fn find_all_possible_values_for_each_empty_cell(
        &mut self,
        row_nums: &mut [u16; PUZZLE_SIZE],
        col_nums: &mut [u16; PUZZLE_SIZE],
        square_nums: &mut [u16; PUZZLE_SIZE],
    ) {
        for i in 0..CELL_COUNT {
            let cur_value = self.puzzle[i];
            if cur_value > 0 {
                self.save_value_for_cell_and_check_is_solvable(
                    cur_value,
                    i,
                    row_nums,
                    col_nums,
                    square_nums,
                );
            }
        }
    }

    fn prepare_puzzle_for_solving(
        &mut self,
        puzzle_sorted: &mut [u8],
        indices: &mut [u8],
        row_nums: &mut [u16],
        col_nums: &mut [u16],
        square_nums: &mut [u16],
    ) {
        let mut number_off_sets: [u8; PUZZLE_SIZE + 2] = [0; PUZZLE_SIZE + 2]; // counts 0 - 9 + 1 offset = puzzleSize + 2 (9 + 2)
        for i in 0..CELL_COUNT {
            let count_of_index = self.get_possible_counts(i, row_nums, col_nums, square_nums);
            number_off_sets[count_of_index + 1] += 1;
        }
        self.my_is_solved = number_off_sets[1] as usize == CELL_COUNT; // all cells have already a solution!
        for i in 1..PUZZLE_SIZE + 2 {
            // correct offsets
            number_off_sets[i] += number_off_sets[i - 1];
        }
        for i in 0..CELL_COUNT {
            let count_of_index = self.get_possible_counts(i, row_nums, col_nums, square_nums);
            let off_set = number_off_sets[count_of_index] as usize;
            indices[off_set] = i as u8;
            number_off_sets[count_of_index] += 1;
        }
        self.sort_puzzle(puzzle_sorted, indices); // avoid jumping in the puzzle array
    }

    fn find_solution_non_recursively(
        &mut self,
        puzzle_sorted: &mut [u8],
        indices: &[u8],
        row_nums: &mut [u16],
        col_nums: &mut [u16],
        square_nums: &mut [u16],
    ) {
        let mut indices_current: [i8; CELL_COUNT] = [-1; CELL_COUNT];
        let mut i = 0;
        while i < CELL_COUNT {
            let cur_value = puzzle_sorted[i]; //kind of stack
            if cur_value == 0 {
                // Is not given?

                // Is there a current guess possible?
                let puzzle_index: usize = indices[i] as usize;
                let row_index: usize = SudokuPuzzle::calculate_row_index(puzzle_index);
                let col_index: usize = SudokuPuzzle::calculate_col_index(puzzle_index);
                let square_index: usize =
                    SudokuPuzzle::calculate_square_index(row_index, col_index);
                let possible_number_index =
                    row_nums[row_index] | col_nums[col_index] | square_nums[square_index];
                let next_number_index: u8 = (indices_current[i] + 1) as u8;

                if next_number_index < BITSET_LENGTH[possible_number_index as usize] {
                    // next possible number to try found:
                    let next_numbers: &[u8] = BITSET_ARRAY[possible_number_index as usize];
                    let next_number: u8 = next_numbers[next_number_index as usize];
                    puzzle_sorted[i] = next_number;

                    // save value for cell:
                    let check_bit: u16 = 1 << (next_number - 1);
                    row_nums[row_index] |= check_bit;
                    col_nums[col_index] |= check_bit;
                    square_nums[square_index] |= check_bit;

                    indices_current[i] = next_number_index as i8; // success
                    i += 1; // go to next cell
                } else {
                    // backtrack:
                    indices_current[i] = -1; // forget last index for position i
                    i -= 1; // not given values are in the head of myIndices, there we can simply go one step back!
                    let last_invalid_try = puzzle_sorted[i];
                    let last_puzzle_index: usize = indices[i] as usize;
                    puzzle_sorted[i] = 0; // find in the next step a new solution for i

                    // revert last value:
                    let last_row_index: usize =
                        SudokuPuzzle::calculate_row_index(last_puzzle_index);
                    let last_col_index: usize =
                        SudokuPuzzle::calculate_col_index(last_puzzle_index);
                    let last_square_index: usize =
                        SudokuPuzzle::calculate_square_index(last_row_index, last_col_index);
                    let last_check_bit: u16 = 1 << (last_invalid_try - 1);
                    row_nums[last_row_index] ^= last_check_bit;
                    col_nums[last_col_index] ^= last_check_bit;
                    square_nums[last_square_index] ^= last_check_bit;
                }
            } else {
                i += 1;
            }
        }
        self.fill_positions(puzzle_sorted, indices);
        self.my_is_solved = true;
    }

    fn sort_puzzle(&mut self, puzzle_sorted: &mut [u8], indices: &[u8]) {
        for i in 0..CELL_COUNT {
            puzzle_sorted[i] = self.puzzle[indices[i] as usize];
        }
    }

    fn fill_positions(&mut self, puzzle_sorted: &mut [u8], indices: &[u8]) {
        for i in 0..CELL_COUNT {
            self.puzzle[indices[i] as usize] = puzzle_sorted[i];
        }
    }

    fn save_value_for_cell_and_check_is_solvable(
        &mut self,
        value: u8,
        index: usize,
        row_nums: &mut [u16],
        col_nums: &mut [u16],
        square_nums: &mut [u16],
    ) {
        let row_index: usize = SudokuPuzzle::calculate_row_index(index);
        let col_index: usize = SudokuPuzzle::calculate_col_index(index);
        let square_index: usize = SudokuPuzzle::calculate_square_index(row_index, col_index);
        let check_bit: u16 = 1 << (value - 1);
        self.my_is_solvable &= SudokuPuzzle::set_and_check_bit(check_bit, row_nums, row_index);
        self.my_is_solvable &= SudokuPuzzle::set_and_check_bit(check_bit, col_nums, col_index);
        self.my_is_solvable &=
            SudokuPuzzle::set_and_check_bit(check_bit, square_nums, square_index);
    }

    fn set_and_check_bit(check_bit: u16, array: &mut [u16], index: usize) -> bool {
        let old_value = array[index];
        array[index] |= check_bit;
        old_value != array[index]
    }

    fn calculate_row_index(index: usize) -> usize {
        index / PUZZLE_SIZE
    }

    fn calculate_col_index(index: usize) -> usize {
        index % PUZZLE_SIZE
    }

    fn calculate_square_index(row_index: usize, col_index: usize) -> usize {
        row_index / SQUARE_SIZE * SQUARE_SIZE + col_index / SQUARE_SIZE //attention: int arithmetic
    }

    fn get_possible_counts(
        &self,
        index: usize,
        row_nums: &[u16],
        col_nums: &[u16],
        square_nums: &[u16],
    ) -> usize {
        if self.puzzle[index] == 0 {
            let row_index: usize = SudokuPuzzle::calculate_row_index(index);
            let col_index: usize = SudokuPuzzle::calculate_col_index(index);
            let square_index: usize = SudokuPuzzle::calculate_square_index(row_index, col_index);
            let possible_number_index =
                row_nums[row_index] | col_nums[col_index] | square_nums[square_index];
            BITSET_LENGTH[possible_number_index as usize] as usize
        } else {
            0
        }
    }

    pub fn to_pretty_string(&self) -> String {
        let dotted_line: String = (0..(PUZZLE_SIZE * 3 + SQUARE_SIZE - 1))
            .map(|_| "-")
            .collect::<String>();
        let empty = "_";
        let mut buffer: Vec<String> = Vec::new();
        for row in 0..PUZZLE_SIZE {
            let from = row * PUZZLE_SIZE;
            let until = from + PUZZLE_SIZE;
            let current_row = &self.puzzle[from..until];
            let mut formatted_row: String = String::with_capacity(3 * PUZZLE_SIZE + 2);
            for (col, col_value) in current_row.iter().enumerate().take(PUZZLE_SIZE) {
                let rs: String = if *col_value == 0 {
                    format!(" {} ", empty)
                } else {
                    format!(" {} ", *col_value)
                };
                formatted_row.push_str(&rs);
                if col + 1 < PUZZLE_SIZE && col % SQUARE_SIZE == 2 {
                    formatted_row.push('|');
                }
            }
            buffer.push(formatted_row);
            if row < (PUZZLE_SIZE - 1) && (row + 1) % SQUARE_SIZE == 0 {
                buffer.push(dotted_line.clone());
            }
        }
        buffer.join("\n")
    }

}

impl Display for SudokuPuzzle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut buffer: Vec<String> = Vec::new();
        for row in 0..PUZZLE_SIZE {
            let from = row * PUZZLE_SIZE;
            let until = from + PUZZLE_SIZE;
            let current_row = self.puzzle[from..until]
                .iter()
                .map(|i| i.to_string())
                .collect::<String>();
            buffer.push(current_row);
        }
        let rs : String = buffer.join("\n");
        write!(f, "{}", rs)
    }
}