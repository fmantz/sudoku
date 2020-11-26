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
use std::env;
use std::path::{MAIN_SEPARATOR, Path};
use std::time::Instant;
use rayon::prelude::*;
use rayon::iter::FromParallelIterator;

use crate::sudoku_io::SudokuIO;
use crate::sudoku_iterator::{SudokuIterator, SudokuGroupedIterator};
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_puzzle::SudokuPuzzleData;
use crate::sudoku_constants::PARALLELIZATION_COUNT;

mod sudoku_bit_set;
mod sudoku_puzzle;
mod sudoku_io;
mod sudoku_iterator;
mod sudoku_constants;
mod sudoku_turbo;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!(">SudokuSolver inputFile [outputFile]");
        println!("-First argument must be path to sudoku puzzles!");
        println!("-Second argument can be output path for sudoku puzzles solution!");
    } else {
        let start: Instant = Instant::now();
        let input_file_name: &String = args.get(1).unwrap();
        let output_file_name: String = if args.len() > 2 {
            let second_argument: String = args.get(2).unwrap().to_string();
            second_argument
        } else {
            let path = Path::new(input_file_name);
            let parent = path.parent();
            let generated_file_name: String = if parent.is_some() {
                let simple_file_name: String = path.file_name().unwrap().to_str().unwrap().to_string();
                let new_file_name: String = format!("SOLUTION_{}", simple_file_name);
                parent.unwrap().join(new_file_name).to_str().unwrap().to_string()
            } else {
                format!(".{}sudoku_solution.txt", MAIN_SEPARATOR).to_string()
            };
            generated_file_name
        };
        let puzzles: Result<SudokuIterator, String> = SudokuIO::read(input_file_name);
        match puzzles {
            Ok(puzzles) => {

                let grouped_iterator = SudokuGroupedIterator::grouped(puzzles, PARALLELIZATION_COUNT);
                let mut counter : usize = 0;
                for mut puzzle_buffer in grouped_iterator {

                    //Assign numbers to sudokus for better error messages:
                    let mut indexed_unsolved_sudokus : Vec<(usize, SudokuPuzzleData)> = puzzle_buffer
                        .into_iter()
                        .enumerate()
                        .map(|(index, unsolved_sudoku)| {counter+=1; (index + counter, unsolved_sudoku)})
                        .collect();

                    let solved_sudokus: Vec<SudokuPuzzleData> = indexed_unsolved_sudokus
                        .par_iter() //solve in parallel
                        .map(|(index, mut unsolved_sudoku)| {
                            solve_current_sudoku(index, unsolved_sudoku);
                        })
                        .collect();

                    let write_rs : Result<(), String> = SudokuIO::write_qqwing(&output_file_name, solved_sudokus);
                    match write_rs {
                        Ok(()) => { /* do nothing */ }
                        Err(error) => {
                            panic!("Problem with saving solved puzzle: {:?}", error);
                        }
                    };
                }
                let duration = start.elapsed();
                println!("output: {} ", Path::new(&output_file_name).to_str().unwrap());
                println!("All sudoku puzzles solved by simple backtracking algorithm in {:?}", duration);

/*
//TODO implement grouped method in iterator
//https://github.com/rayon-rs/rayon
use rayon::prelude::*;
fn sum_of_squares(input: &[i32]) -> i32 {
    input.par_iter() // <-- just change that!
         .map(|&i| i * i)
         .sum()
}
 */
/*
                let write_rs : Result<(), String> = SudokuIO::write_qqwing(&output_file_name, puzzles, solve_current_sudoku);
                match write_rs {
                    Ok(()) => { /* do nothing */ }
                    Err(error) => {
                        panic!("Problem with saving solved puzzle: {:?}", error);
                    }
                };


 */
            }
            Err(error) => {
                panic!("Problem opening the file: {:?}", error);
            }
        }
    }
}

fn solve_current_sudoku(index: &usize, mut sudoku: SudokuPuzzleData) -> SudokuPuzzleData {
    sudoku.init_turbo();
    if sudoku.is_solved() {
        println!("Sudoku {} is already solved!", index);
    } else if sudoku.is_solvable() {
        sudoku.solve();
    } else {
        println!("Sudoku {} is unsolvable:\n {}", index, sudoku.to_pretty_string());
    }
    return sudoku;
}

#[cfg(test)]
mod tests {

    use std::path::MAIN_SEPARATOR;
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::sudoku_constants::{EMPTY_CHAR, QQWING_EMPTY_CHAR, PUZZLE_SIZE, SQUARE_SIZE};
    use crate::sudoku_io::SudokuIO;
    use crate::sudoku_iterator::SudokuIterator;
    use crate::sudoku_puzzle::{SudokuPuzzle, SudokuPuzzleData};
    use crate::sudoku_bit_set::SudokuBitSet;

    #[test]
    fn solve_should_solve_50_sudokus_from_project_euler_by_simple_backtracking_algorithm() -> () {
        check_solve("p096_sudoku.txt");
    }

    #[test]
    fn solve_should_solve_10_sudokus_generated_with_qqwing_by_simple_backtracking_algorithm() -> () {
        check_solve("sudoku.txt");
    }

    pub fn check_solve(filename: &str) -> () {
        let mut dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push(format!("test{}resources{}{}", MAIN_SEPARATOR, MAIN_SEPARATOR, filename).to_string());
        let filename_with_path: &str = dir.as_os_str().to_str().unwrap();
        let start: Instant = Instant::now();
        let rs: SudokuIterator = match SudokuIO::read(filename_with_path) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles
        };
        for (index, mut sudoku) in rs.enumerate() {
            let sudoku_number: usize = index + 1;
            let input: String = sudoku.to_string();
            sudoku.init_turbo();
            assert_eq!(sudoku.is_solvable(), true, "Sudoku {} is not well-defined:\n {}", sudoku_number, sudoku.to_pretty_string());
            sudoku.solve();
            let output = sudoku.to_string();
            assert_eq!(check_solution(&sudoku), true, "Sudoku {} is not solved:\n {}", sudoku_number, sudoku.to_pretty_string());
            assert_eq!(sudoku.is_solved(), true, "Sudoku {} is solved but isSolved() return false", sudoku_number);
            assert_eq!(input.len(), output.len(), "sudoku strings have not same length");
            let output_char_vec: Vec<char> = output.chars().collect();
            for (i, in_char) in input.char_indices() {
                let out_char = output_char_vec[i];
                if !is_blank(in_char) {
                    assert_eq!(in_char, out_char) //puzzle should not be changed!
                }
            }
        }
        let duration = start.elapsed();
        println!("All sudoku puzzles solved by simple backtracking algorithm in {:?}", duration);
    }

    pub fn is_blank(c: char) -> bool {
        return c == EMPTY_CHAR || c == QQWING_EMPTY_CHAR;
    }

    /**
     * @param row in [0,9]
     */
    fn is_row_ok(sudoku: &SudokuPuzzleData, row: usize) -> bool {
        let mut bits: SudokuBitSet = SudokuBitSet::new();
        check_row(sudoku, row, &mut bits);
        return bits.is_found_numbers_unique() && bits.is_all_numbers_found();
    }

    #[inline]
    fn check_row(sudoku: &SudokuPuzzleData, row: usize, bits: &mut SudokuBitSet) -> () {
        let selected_row: [u8; PUZZLE_SIZE] = sudoku.puzzle[row];
        for col in 0..PUZZLE_SIZE {
            let value: u8 = selected_row[col];
            bits.save_value(value);
        }
    }

    /**
     * @param col in [0,9]
     */
    fn is_col_ok(sudoku: &SudokuPuzzleData, row: usize) -> bool {
        let mut bits: SudokuBitSet = SudokuBitSet::new();
        check_col(sudoku,row, &mut bits);
        return bits.is_found_numbers_unique() && bits.is_all_numbers_found();
    }

    #[inline]
    fn check_col(sudoku: &SudokuPuzzleData, col: usize, bits: &mut SudokuBitSet) -> () {
        for row in 0..PUZZLE_SIZE {
            let value: u8 = sudoku.puzzle[row][col];
            bits.save_value(value);
        }
    }

    /**
     * @param rowSquareIndex in [0,2]
     * @param colSquareIndex in [0,2]
     */
    fn is_square_ok(sudoku: &SudokuPuzzleData, row_square_index: usize, col_square_index: usize) -> bool {
        let mut bits: SudokuBitSet = SudokuBitSet::new();
        check_square(sudoku, row_square_index, col_square_index, &mut bits);
        return bits.is_found_numbers_unique() && bits.is_all_numbers_found();
    }

    #[inline]
    fn check_square(sudoku: &SudokuPuzzleData, row_square_index: usize, col_square_index: usize, bits: &mut SudokuBitSet) -> () {
        let row_square_offset: usize = row_square_index * SQUARE_SIZE;
        let col_square_offset: usize = col_square_index * SQUARE_SIZE;
        for row in 0..SQUARE_SIZE {
            for col in 0..SQUARE_SIZE {
                let value: u8 = sudoku.puzzle[row + row_square_offset][col + col_square_offset];
                bits.save_value(value);
            }
        }
    }

    #[inline]
    fn check_solution(sudoku: &SudokuPuzzleData) -> bool {
        for row in 0..PUZZLE_SIZE {
            if !is_row_ok(sudoku,row) {
                return false;
            }
            for col in 0..PUZZLE_SIZE {
                if !is_col_ok(sudoku,col) {
                    return false;
                }
                for i in 0..PUZZLE_SIZE {
                    if !is_square_ok(sudoku,i / SQUARE_SIZE, i % SQUARE_SIZE) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
