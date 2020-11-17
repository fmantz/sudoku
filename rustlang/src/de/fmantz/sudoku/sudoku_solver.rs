mod sudoku_bit_set;
mod sudoku_puzzle;
mod sudoku_io;
mod sudoku_iterator;
mod sudoku_constants;

use crate::sudoku_bit_set::SudokuBitSet;
use crate::sudoku_puzzle::SudokuPuzzleData;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_iterator::SudokuIterator;
use crate::sudoku_io::SudokuIO;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::{Path, MAIN_SEPARATOR};
use std::time::{Duration, Instant};
use std::iter::Map;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.is_empty() {
        println!(">SudokuSolver inputFile [outputFile]");
        println!("-First argument must be path to sudoku puzzles!");
        println!("-Second argument can be output path for sudoku puzzles solution!");
    } else {
        let start: Instant = Instant::now();
        let input_file_name : &String = args.get(1).unwrap();
        let output_file_name: String = if args.len() > 2 {
            let second_argument: String = args.get(2).unwrap().to_string();
            second_argument
        } else {
            let path = Path::new(input_file_name);
            let parent = path.parent();
            let generated_file_name : String  = if parent.is_some() {
                let simple_file_name : String = path.file_name().unwrap().to_str().unwrap().to_string();
                let new_file_name : String = format!("SOLUTION_{}", simple_file_name);
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
            },
            Ok(puzzles) => {
                SudokuIO::write_qqwing(&output_file_name, puzzles, solve_current_sudoku);
                let duration = start.elapsed();
                println!("output: {} ", Path::new(&output_file_name).to_str().unwrap());
                println!("All sudoku puzzles solved by simple backtracking algorithm in {:?}", duration);
            }
        }
    }
}

fn solve_current_sudoku(index: &u32, sudoku : & mut SudokuPuzzleData) -> () {
    if sudoku.is_solved() {
        println!("Sudoku {} is already solved!", index);
    } else if sudoku.is_solvable() {
        sudoku.solve();
        if !sudoku.is_solved() {
            println!("ERROR: Sudoku {} is not correctly solved!", index);
        }
    } else {
        println!("Sudoku {} is unsolvable:\n {}" , index, sudoku.to_pretty_string());
    }
}


