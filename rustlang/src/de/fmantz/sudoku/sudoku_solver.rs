mod sudoku_bit_set;
mod sudoku_puzzle;
mod sudoku_io;
mod sudoku_iterator;
mod sudoku_constants;

use crate::sudoku_bit_set::SudokuBitSet;
use crate::sudoku_puzzle::SudokuPuzzleData;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_iterator::Puzzles;
use crate::sudoku_io::SudokuIO;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {

    let start = Instant::now();
    let puzzles = SudokuIO::read("/home/florian/temp/sudoku2.txt");

    match puzzles {
        Err(error ) => {
            panic!("Problem opening the file: {:?}", error);
        },
        Ok(puzzleLines ) => {
            for mut puzzle in puzzleLines {
                println!("{}\n\n", puzzle.to_pretty_string());
                puzzle.solve();
                println!("{}\n\n", puzzle.to_pretty_string());
            }
            let duration = start.elapsed();
            println!("Time elapsed in expensive_function() is: {:?}", duration);
        }
    }
}
