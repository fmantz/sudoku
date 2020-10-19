mod sudoku_bit_set;
mod sudoku_puzzle;
mod sudoku_io;
mod sudoku_iterator;

use crate::sudoku_bit_set::SudokuBitSet;
use crate::sudoku_puzzle::SudokuPuzzleData;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_iterator::PuzzleLines;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn main() {
    println!("Hello, world!");
    let mut bit_set: SudokuBitSet = SudokuBitSet::new();
    bit_set.save_value(2);
    println!("{}", bit_set.to_string());
    println!("Hello, world!");
    let x:u16 = !0 >> (16 - 9);
    println!("{:#018b}", x);

    let mut test_object: SudokuBitSet = SudokuBitSet::new();
    for i in 1..9 {
        test_object.save_value(i);
        println!("{}, {}", i, test_object.to_string())
    }

    let test:SudokuPuzzleData = SudokuPuzzle::new();
    println!("{}\n\n", test.to_string());

    let file_data = read_lines("/home/florian/temp/sudoku2.txt");
    if let Ok(lines) = file_data {
        // Consumes the iterator, returns an (Optional) String
        let mut puzzles = PuzzleLines::new(lines);
        let mut puzzle = puzzles.next().unwrap();
        println!("{}\n\n", puzzle.to_pretty_string());
    } else if let Err(myerror) = file_data {
        println!("{}\n\n", myerror);
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}