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
use crate::sudoku_bit_set::SudokuBitSet;
use crate::sudoku_constants::{PUZZLE_SIZE, SQUARE_SIZE};
use crate::sudoku_puzzle::{SudokuPuzzleData, SudokuPuzzle};

pub struct SudokuTurbo {
    //Save how many values are preset:
    col_counts: [u8; PUZZLE_SIZE],
    row_counts: [u8; PUZZLE_SIZE],

    //Save values set:
    col_nums: [u16; PUZZLE_SIZE],
    row_nums: [u16; PUZZLE_SIZE],
    square_nums: [u16; PUZZLE_SIZE],

    //Store optimized sort order:
    col_indices: [usize; PUZZLE_SIZE + 1],
    row_indices: [usize; PUZZLE_SIZE + 1],
    my_is_solvable: bool,
}

impl SudokuTurbo {

    fn save_value_and_check_is_solvable(&mut self, row: usize, col: usize, value: u8) -> () {
        if value != 0 {

            //save col data:
            self.col_counts[col] += 1;
            let old_col_num_value = self.col_nums[col];
            let new_col_num_value = SudokuTurbo::store_value_as_bit(old_col_num_value, value);
            self.col_nums[col] = new_col_num_value;

            //save row data:
            self.row_counts[row] += 1;
            let old_row_num_value = self.row_nums[row];
            let new_row_num_value = SudokuTurbo::store_value_as_bit(old_row_num_value, value);
            self.row_nums[row] = new_row_num_value;

            //save square data:
            let square_index = SudokuTurbo::calculate_square_index(row, col);
            let old_square_num_value = self.square_nums[square_index];
            let new_square_num_value = SudokuTurbo::store_value_as_bit(old_square_num_value, value);
            self.square_nums[square_index] = new_square_num_value;

            //If old and new value is equal the same value
            //has already been stored before:
            self.my_is_solvable = self.my_is_solvable &&
                old_col_num_value != new_col_num_value &&
                old_row_num_value != new_row_num_value &&
                old_square_num_value != new_square_num_value;
        }
    }

    pub fn save_value(&mut self, row: usize, col: usize, value: u8) -> () {
        if value != 0 {
            //save col data:
            self.col_counts[col] += 1;
            self.col_nums[col] = SudokuTurbo::store_value_as_bit(self.col_nums[col], value);
            //save row data:
            self.row_counts[row] += 1;
            self.row_nums[row] = SudokuTurbo::store_value_as_bit(self.row_nums[row], value);
            //save square data:
            let square_index = SudokuTurbo::calculate_square_index(row, col);
            self.square_nums[square_index] = SudokuTurbo::store_value_as_bit(self.square_nums[square_index], value);
        }
    }

    pub fn revert_value(&mut self, row: usize, col: usize, value: u8) -> () {
        if value != 0 {
            //save col data:
            self.col_counts[col] -= 1;
            self.col_nums[col] = SudokuTurbo::revert_value_as_bit(self.col_nums[col], value);
            //save row data:
            self.row_counts[row] -= 1;
            self.row_nums[row] = SudokuTurbo::revert_value_as_bit(self.row_nums[row], value);
            //save square data:
            let square_index = SudokuTurbo::calculate_square_index(row, col);
            self.square_nums[square_index] = SudokuTurbo::revert_value_as_bit(self.square_nums[square_index], value);
        }
    }

    pub fn create_solution_space(&mut self, row: usize, col: usize) -> SudokuBitSet {
        let square_index = SudokuTurbo::calculate_square_index(row, col);
        let bits: u16 = self.col_nums[col] | self.row_nums[row] | self.square_nums[square_index];
        SudokuBitSet::new_with_data(bits)
    }

    pub fn is_solvable(& self) -> bool {
        self.my_is_solvable
    }

    pub fn is_solved(& self) -> bool {
        for col in 0..PUZZLE_SIZE {
            //Does not ensure the solution is correct, but the algorithm will!
            if self.col_nums[col] != (PUZZLE_SIZE as u16) {
                return false;
            }
        }
        return true;
    }

    pub fn create(puzzle_data: SudokuPuzzleData) -> Self {

        //Save how many values are preset:
        let mut col_counts: [u8; PUZZLE_SIZE] = [0; PUZZLE_SIZE];
        let mut row_counts: [u8; PUZZLE_SIZE] = [0; PUZZLE_SIZE];

        //Save values set:
        let mut col_nums: [u16; PUZZLE_SIZE] = [0; PUZZLE_SIZE];
        let mut row_nums: [u16; PUZZLE_SIZE] = [0; PUZZLE_SIZE];
        let mut square_nums: [u16; PUZZLE_SIZE] = [0; PUZZLE_SIZE];

        //Store optimized sort order:
        let mut col_indices: [usize; PUZZLE_SIZE + 1] = [0; PUZZLE_SIZE + 1];
        let mut row_indices: [usize; PUZZLE_SIZE + 1] = [0; PUZZLE_SIZE + 1];
        let mut my_is_solvable : bool = true;

        let mut rs = SudokuTurbo {
            col_counts,
            row_counts: row_counts,
            col_nums: col_nums,
            row_nums: row_nums,
            square_nums: square_nums,
            col_indices: col_indices,
            row_indices: row_indices,
            my_is_solvable: my_is_solvable
        };

        for row in 0..PUZZLE_SIZE {
            let row_data = puzzle_data.puzzle_data()[row];
            for col in 0..PUZZLE_SIZE {
                rs.save_value_and_check_is_solvable(row, col, row_data[col]);
            }
        }

        SudokuTurbo::create_sorted_indices(col_counts, rs.col_indices);
        SudokuTurbo::create_sorted_indices(row_counts, rs.row_indices);

        rs
    }

    fn calculate_square_index(row: usize, col: usize) -> usize {
        col / SQUARE_SIZE + row / SQUARE_SIZE * SQUARE_SIZE
    }

    /**
     * Save a value
     */
    fn store_value_as_bit(container: u16, value: u8) -> u16 {
        let check_bit = 1 << (value - 1); //set for each number a bit by index from left, number 1 has index zero
        container | check_bit
    }

    /**
     * Revert a value
     */
    fn revert_value_as_bit(container: u16, value: u8) -> u16 {
        let check_bit = 1 << (value - 1); //set for each number a bit by index from left, number 1 has index zero
        container ^ check_bit
    }

    fn create_sorted_indices(num: [u8; PUZZLE_SIZE], mut rs: [usize; PUZZLE_SIZE + 1]) -> () {
        let mut nums_with_indices: Vec<(usize, &u8)> = num.iter().enumerate().collect();
        nums_with_indices.sort_by(|(a_num,a_index), (b_num, b_index)| b_num.cmp(a_num));
        for col in 0..PUZZLE_SIZE {
            rs[col] = nums_with_indices[col].0;
        }
        rs[PUZZLE_SIZE] = 0;
    }

}
