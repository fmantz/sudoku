mod sudoku_bit_set;
mod sudoku_puzzle;
mod sudoku_io;
mod sudoku_iterator;

use crate::sudoku_bit_set::SudokuBitSet;
use crate::sudoku_puzzle::SudokuPuzzleData;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_iterator::PuzzleLines;

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
    println!("{}\n\n", test.to_pretty_string());

    /*
    if let Ok(lines) = read_lines("./hosts") {
        lines
    }
    */

}
