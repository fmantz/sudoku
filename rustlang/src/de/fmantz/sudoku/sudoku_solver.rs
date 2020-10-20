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

    let start = Instant::now();
    let puzzles = SudokuIO::read("/home/florian/temp/sudoku2.txt");

    match puzzles {
        Err(error ) => {
            panic!("Problem opening the file: {:?}", error);
        },
        Ok(puzzles )  => {
            SudokuIO::write_qqwing("/home/florian/temp/sudoku2_solution.txt",puzzles, solveSudoku);

            //SudokuIO::write_qqwing("/home/florian/temp/sudoku2.txt", puzzels_solved);
            let duration = start.elapsed();
            println!("Time elapsed in expensive_function() is: {:?}", duration);
        }
    }
}

fn solveSudoku( s : & mut SudokuPuzzleData) -> () {
    s.solve();
}


