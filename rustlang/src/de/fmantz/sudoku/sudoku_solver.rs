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
use std::path::Path;
use std::time::{Duration, Instant};
use std::iter::Map;

fn main() {

    let start : Instant = Instant::now();
    let puzzles: Result<SudokuIterator, String>  = SudokuIO::read("/home/florian/temp/sudoku2.txt");

    match puzzles {
        Err(error ) => {
            panic!("Problem opening the file: {:?}", error);
        },
        Ok(puzzles )  => {
            SudokuIO::write_qqwing("/home/florian/temp/sudoku2_solution.txt", puzzles, solve_current_sudoku);
            let duration = start.elapsed();
            //println!("output:" + new File(outputFileName).getAbsolutePath);
            println!("All sudoku puzzles solved by simple backtracking algorithm in {:?}", duration);

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


