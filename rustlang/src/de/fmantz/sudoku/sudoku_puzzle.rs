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
use crate::sudoku_constants::{BITSET_ARRAY, CELL_COUNT, PUZZLE_SIZE};
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

    fn init(&mut self) -> () {
        self.find_all_possible_values_for_each_empty_cell();
        self.prepare_puzzle_for_solving();
    }

    fn is_solvable(&self) -> bool {
        return self.my_is_solvable;
    }

    fn is_solved(&self) -> bool {
        return self.my_is_solved;
    }

    /**
     * solves the sudoku by a simple non-recursive backtracking algorithm (brute force)
     * (own simple solution, its an algorithm which may be ported to CUDA or OpenCL)
     * to get a faster result use e.g. https://github.com/Emerentius/sudoku
     */
    fn solve(&mut self) -> () {
        if self.is_solvable() && !self.is_solved() {
            self.find_solution_non_recursively();
        }
    }

    fn to_pretty_string(&self) -> String {
        let dotted_line: String = (0..(PUZZLE_SIZE * 3 + SQUARE_SIZE - 1)).map(|_| "-").collect::<String>();
        let empty = "*";
        let mut buffer: Vec<String> = Vec::new();
        for row in 0..PUZZLE_SIZE {
            let from = row * PUZZLE_SIZE;
            let until = from + PUZZLE_SIZE;
            let current_row = &self.puzzle[from..until];
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

    fn to_string(&self) -> String {
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
        return buffer.join("\n");
    }
}

impl SudokuPuzzleData {
    fn find_all_possible_values_for_each_empty_cell(&mut self) -> () {
        for i in 0..CELL_COUNT {
            let cur_value = self.puzzle[i];
            if cur_value > 0 {
                self.save_value_for_cell_and_check_is_solvable(cur_value, i);
            }
        }
    }

    fn prepare_puzzle_for_solving(&mut self) -> () {
        let mut number_off_sets: [u8; PUZZLE_SIZE + 2] = [0; PUZZLE_SIZE + 2]; //counts 0 - 9 + 1 offset = puzzleSize + 2 (9 + 2)
        for i in 0..CELL_COUNT {
            let count_of_index = self.get_possible_counts(i);
            number_off_sets[count_of_index + 1] += 1;
        }
        self.my_is_solved = number_off_sets[1] as usize == CELL_COUNT; //all cells have already a solution!
        for i in 1..number_off_sets.len() { //correct offsets
            number_off_sets[i] += number_off_sets[i - 1];
        }
        for i in 0..CELL_COUNT {
            let count_of_index = self.get_possible_counts(i);
            let off_set = number_off_sets[count_of_index] as usize;
            self.indices[off_set] = i;
            number_off_sets[count_of_index] += 1;
        }
        self.sort_puzzle(); //avoid jumping in the puzzle array
    }

    fn find_solution_non_recursively(&mut self) -> () {
        let mut last_invalid_try: u8 = 0;
        let mut i = 0;
        while i < CELL_COUNT {
            let cur_value = self.puzzle_sorted[i]; //kind of stack
            if cur_value == 0 { //Is not given?

                //Is there a current guess possible?
                let puzzle_index = self.indices[i];
                let row_index: usize = SudokuPuzzleData::calculate_row_index(puzzle_index);
                let col_index: usize = SudokuPuzzleData::calculate_col_index(puzzle_index);
                let square_index: usize = SudokuPuzzleData::calculate_square_index(row_index, col_index);
                let possible_number_index = self.row_nums[row_index] | self.col_nums[col_index] | self.square_nums[square_index];
                let next_numbers = BITSET_ARRAY[possible_number_index as usize];
                let next_number_index = if last_invalid_try == 0 {
                    0
                } else {
                    SudokuPuzzleData::fast_index_of(next_numbers, &last_invalid_try) + 1
                };

                if next_number_index < next_numbers.len() {
                    //next possible number to try found:
                    let next_number = next_numbers[next_number_index];
                    self.puzzle_sorted[i] = next_number;
                    self.save_value_for_cell(next_number, row_index, col_index, square_index);
                    last_invalid_try = 0; //0 since success
                    i += 1; //go to next cell
                } else {
                    i -= 1; //backtrack, note not given values are in the head of myIndices, we can simply go one step back!
                    last_invalid_try = self.puzzle_sorted[i];
                    self.puzzle_sorted[i] = 0; //forget
                    let last_puzzle_index = self.indices[i];
                    self.revert_value_for_cell(last_invalid_try, last_puzzle_index);
                }
            } else {
                i += 1;
            }
        }
        self.fill_positions();
        self.my_is_solved = true;
    }

    fn sort_puzzle(&mut self) -> () {
        for i in 0..CELL_COUNT {
            self.puzzle_sorted[i] = self.puzzle[self.indices[i]];
        }
    }

    fn fill_positions(&mut self) -> () {
        for i in 0..CELL_COUNT {
            self.puzzle[self.indices[i]] = self.puzzle_sorted[i];
        }
    }

    fn get_single_array_index(row: usize, col: usize) -> usize {
        row * PUZZLE_SIZE + col
    }

    fn fast_index_of(array: &[u8], number: &u8) -> usize {
        let mut index = 0;
        for number_in_array in array {
            if number_in_array != number {
                index += 1;
            } else {
                break;
            }
        }
        index
    }

    fn save_value_for_cell_and_check_is_solvable(&mut self, value: u8, index: usize) -> () {
        let row_index: usize = SudokuPuzzleData::calculate_row_index(index);
        let col_index: usize = SudokuPuzzleData::calculate_col_index(index);
        let square_index: usize = SudokuPuzzleData::calculate_square_index(row_index, col_index);
        let check_bit: u16 = 1 << (value - 1);
        self.my_is_solvable &= SudokuPuzzleData::set_and_check_bit(check_bit, &mut self.row_nums, row_index);
        self.my_is_solvable &= SudokuPuzzleData::set_and_check_bit(check_bit, &mut self.col_nums, col_index);
        self.my_is_solvable &= SudokuPuzzleData::set_and_check_bit(check_bit, &mut self.square_nums, square_index);
    }

    fn set_and_check_bit(check_bit: u16, array: &mut [u16], index: usize) -> bool {
        let old_value = array[index];
        array[index] |= check_bit;
        old_value != array[index]
    }

    fn save_value_for_cell(&mut self, value: u8, row_index: usize, col_index: usize, square_index: usize) -> () {
        let check_bit: u16 = 1 << (value - 1);
        self.row_nums[row_index] |= check_bit;
        self.col_nums[col_index] |= check_bit;
        self.square_nums[square_index] |= check_bit;
    }

    fn revert_value_for_cell(&mut self, value: u8, index: usize) -> () {
        let row_index: usize = SudokuPuzzleData::calculate_row_index(index);
        let col_index: usize = SudokuPuzzleData::calculate_col_index(index);
        let square_index: usize = SudokuPuzzleData::calculate_square_index(row_index, col_index);
        let check_bit: u16 = 1 << (value - 1);
        self.row_nums[row_index] ^= check_bit;
        self.col_nums[col_index] ^= check_bit;
        self.square_nums[square_index] ^= check_bit;
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

    fn get_possible_numbers(&self, index: usize) -> &[u8] {
        let row_index: usize = SudokuPuzzleData::calculate_row_index(index);
        let col_index: usize = SudokuPuzzleData::calculate_col_index(index);
        let square_index: usize = SudokuPuzzleData::calculate_square_index(row_index, col_index);
        let possible_number_index = self.row_nums[row_index] | self.col_nums[col_index] | self.square_nums[square_index];
        BITSET_ARRAY[possible_number_index as usize]
    }

    fn get_possible_counts(&self, index: usize) -> usize {
        if self.puzzle[index] == 0 {
            self.get_possible_numbers(index).len()
        } else {
            0
        }
    }
}
