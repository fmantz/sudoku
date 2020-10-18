use crate::sudoku_bit_set::SudokuBitSet;

pub const PUZZLE_SIZE: usize = 9;
const SQUARE_SIZE: usize = 3;

pub trait SudokuPuzzle {
   fn new() -> Self;
   fn set(& mut self, row:usize, col:usize, value:u8) -> ();
   fn is_empty(&self, row: usize, col: usize) -> bool;
   fn is_solved(&self) -> bool;
   fn is_solvable(&self) -> bool;
   fn solve(& mut self)-> ();
//   fn to_pretty_string(&self) -> String;
//   fn to_string(&self) -> String;
}

pub struct SudokuPuzzleData{
   puzzle: [[u8; PUZZLE_SIZE as usize]; PUZZLE_SIZE as usize],
   is_open: bool,
   is_empty: bool
}

impl SudokuPuzzle for SudokuPuzzleData {

    fn new() -> Self {
        SudokuPuzzleData {
            puzzle: [[0; PUZZLE_SIZE as usize]; PUZZLE_SIZE as usize],
            is_open: true,
            is_empty: true
        }
    }

    fn set(& mut self, row:usize, col:usize, value:u8) -> () {
        if self.is_open{
            self.puzzle[row][col] = value;
            self.is_empty = false;
        }
    }

    fn is_empty(&self, row: usize, col: usize) -> bool {
        return self.puzzle[row][col] == 0;
    }

    fn is_solved(&self) -> bool {
        return self.check_conditions(false);
    }

    fn is_solvable(&self) -> bool {
        return self.check_conditions(true);
    }

    fn solve(& mut self)-> () {
        fn go(puzzle : & mut SudokuPuzzleData) -> () {
            let mut run : bool = true;
            'outer: for row in 0.. PUZZLE_SIZE {
                for col in 0.. PUZZLE_SIZE {
                    if puzzle.is_empty(row, col) {
                        let solution_space : SudokuBitSet = puzzle.create_solution_space(row, col);
                        for n in 1..(PUZZLE_SIZE + 1) as u8 {
                            if solution_space.is_solution(n) {
                                puzzle.set(row, col, n);
                                go(puzzle);
                                puzzle.set(row, col, 0); //back track
                            }
                        }
                        //solution found for slot!
                        run = false;
                        break 'outer;
                    }
                }
            }
            //solution found for all slots:
            if run {
                puzzle.is_open = false;
            }
        }
        go(self);
        self.is_open = true;
    }

/*
    fn to_string(&self) -> String {
        return "";
    }
*/

}

//private functions here:
impl SudokuPuzzleData {

    /**
     * @param row in [0,9]
     * @param relaxed true means it is still solvable, false it contains all possible numbers once
    */
    fn is_row_ok(&self, row:usize, relaxed: bool) -> bool {
        let bits: SudokuBitSet = self.check_row_2p(row);
        return bits.is_found_numbers_unique() && (relaxed || bits.is_all_numbers_found());
    }

    #[inline]
    fn check_row_3p(&self, row: usize, mut bits: SudokuBitSet) -> SudokuBitSet {
        let selected_row: [u8; PUZZLE_SIZE as usize] = self.puzzle[row];
        for col in 0.. PUZZLE_SIZE {
            let value: u8 = selected_row[col];
            bits.save_value(value);
        }
        return bits;
    }

    #[inline]
    fn check_row_2p(&self, row: usize) -> SudokuBitSet {
        return self.check_row_3p( row, SudokuBitSet::new());
    }

    /**
     * @param col in [0,9]
     * @param relaxed true means it is still solvable, false it contains all possible numbers once
     */
    fn is_col_ok(&self, row:usize, relaxed: bool) -> bool {
        let bits: SudokuBitSet = self.check_col_2p(row);
        return bits.is_found_numbers_unique() && (relaxed || bits.is_all_numbers_found());
    }

    #[inline]
    fn check_col_3p(&self, col: usize, mut bits: SudokuBitSet) -> SudokuBitSet {
        for row in 0.. PUZZLE_SIZE {
            let value: u8 = self.puzzle[row][col];
            bits.save_value(value);
        }
        return bits;
    }

    #[inline]
    fn check_col_2p(&self, col: usize) -> SudokuBitSet {
        return self.check_col_3p( col, SudokuBitSet::new());
    }

    /**
     * @param rowSquareIndex in [0,2]
     * @param colSquareIndex in [0,2]
     * @param relaxed true means it is still solvable, false it contains all possible numbers once
    */
    fn is_square_ok(&self, row_square_index: usize, col_square_index: usize, relaxed: bool) -> bool {
        let bits: SudokuBitSet = self.check_square_2p(row_square_index, col_square_index);
        return bits.is_found_numbers_unique() && (relaxed || bits.is_all_numbers_found());
    }

    #[inline]
    fn check_square_3p(&self, row_square_index: usize, col_square_index: usize, mut bits: SudokuBitSet) -> SudokuBitSet {
        let row_square_offset: usize = row_square_index * SQUARE_SIZE;
        let col_square_offset: usize = col_square_index * SQUARE_SIZE;
        for row in 0.. SQUARE_SIZE {
            for col in 0.. SQUARE_SIZE {
                let value: u8 = self.puzzle[row + row_square_offset][col + col_square_offset];
                bits.save_value(value);
            }
        }
        return bits;
    }

    #[inline]
    fn check_square_2p(&self, row_square_index: usize, col_square_index: usize) -> SudokuBitSet {
        return self.check_square_3p( row_square_index, col_square_index, SudokuBitSet::new());
    }

    #[inline]
    fn check_conditions(&self, relaxed: bool) -> bool {
        for row in 0.. PUZZLE_SIZE {
            if !self.is_row_ok(row, relaxed){
                return false;
            }
            for col in 0.. PUZZLE_SIZE {
                if !self.is_col_ok(col, relaxed){
                    return false;
                }
                for i in 0.. PUZZLE_SIZE {
                    if !self.is_square_ok(i / SQUARE_SIZE, i % SQUARE_SIZE, relaxed){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * The method returns a bit set containing all numbers already used
     */
    #[inline]
    fn create_solution_space(&self, row : usize, col: usize) -> SudokuBitSet {
        let bits_row: SudokuBitSet = self.check_row_2p(row);
        let bits_row_col: SudokuBitSet = self.check_col_3p(col, bits_row);
        let bits_row_col_square: SudokuBitSet = self.check_square_3p(
            row / SQUARE_SIZE,
            col / SQUARE_SIZE,
            bits_row_col
        );
        return bits_row_col_square;
    }

}