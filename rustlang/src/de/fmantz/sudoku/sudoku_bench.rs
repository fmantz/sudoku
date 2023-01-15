// bench.rs

extern crate test;

#[cfg(test)]
pub mod bench {

    use super::*;
    use crate::sudoku_constants::PUZZLE_SIZE;
    use crate::sudoku_puzzle::SudokuPuzzle;
    use test::Bencher;

    const PUZZLE: &[&[u8]] = &[
        &[0, 0, 0, 0, 0, 0, 6, 7, 0],
        &[7, 0, 0, 0, 0, 0, 0, 0, 3],
        &[0, 9, 0, 6, 0, 0, 2, 0, 4],
        &[9, 2, 0, 0, 4, 0, 3, 0, 0],
        &[5, 7, 0, 3, 2, 0, 0, 6, 0],
        &[6, 3, 0, 9, 0, 0, 0, 0, 5],
        &[0, 0, 0, 0, 0, 0, 7, 4, 9],
        &[0, 4, 0, 0, 0, 0, 5, 1, 2],
        &[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    #[bench]
    fn bench_solve(b: &mut Bencher) {
        b.iter(|| {
            let mut sudoku = SudokuPuzzle::new();
            for row in 0..PUZZLE_SIZE {
                for col in 0..PUZZLE_SIZE {
                    sudoku.set(row, col, PUZZLE[row][col]);
                }
            }
            sudoku.solve();
        })
    }
}
