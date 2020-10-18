mod sudoku_bit_set;
mod sudoku_puzzle;
use crate::sudoku_bit_set::SudokuBitSet;

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

}
