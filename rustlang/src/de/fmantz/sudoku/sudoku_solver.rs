/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.txt file in the top level folder.
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

use crate::sudoku_io::SudokuIO;
use crate::sudoku_iterator::SudokuIterator;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_puzzle::SudokuPuzzleData;

mod sudoku_bit_set;
mod sudoku_puzzle;
mod sudoku_io;
mod sudoku_iterator;
mod sudoku_constants;

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
            Err(error) => {
                panic!("Problem opening the file: {:?}", error);
            }
            Ok(puzzles) => {
                let write_rs : Result<(), String> = SudokuIO::write_qqwing(&output_file_name, puzzles, solve_current_sudoku);
                match write_rs {
                    Err(error) => {
                        panic!("Problem with saving solved puzzle: {:?}", error);
                    }
                    Ok(()) => { /* do nothing */ }
                };
                let duration = start.elapsed();
                println!("output: {} ", Path::new(&output_file_name).to_str().unwrap());
                println!("All sudoku puzzles solved by simple backtracking algorithm in {:?}", duration);
            }
        }
    }
}

fn solve_current_sudoku(index: &u32, sudoku: &mut SudokuPuzzleData) -> () {
    if sudoku.is_solved() {
        println!("Sudoku {} is already solved!", index);
    } else if sudoku.is_solvable() {
        sudoku.solve();
        if !sudoku.is_solved() {
            println!("ERROR: Sudoku {} is not correctly solved!", index);
        }
    } else {
        println!("Sudoku {} is unsolvable:\n {}", index, sudoku.to_pretty_string());
    }
}


#[cfg(test)]
mod tests {

    use std::path::MAIN_SEPARATOR;
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::sudoku_constants::{EMPTY_CHAR, QQWING_EMPTY_CHAR};
    use crate::sudoku_io::SudokuIO;
    use crate::sudoku_iterator::SudokuIterator;
    use crate::sudoku_puzzle::SudokuPuzzle;

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
            assert_eq!(sudoku.is_solvable(), true, "Sudoku {} is not well-defined:\n {}", sudoku_number, sudoku.to_pretty_string());
            sudoku.solve();
            let output = sudoku.to_string();
            assert_eq!(sudoku.is_solved(), true, "Sudoku {} is not solved:\n {}", sudoku_number, sudoku.to_pretty_string());
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
}
