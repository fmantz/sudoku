/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.md file in the top level folder.
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
