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
use std::fs::File;
use std::io::{self};

use crate::sudoku_constants::NEW_SUDOKU_SEPARATOR;
use crate::sudoku_constants::PUZZLE_SIZE;
use crate::sudoku_puzzle::SudokuPuzzle;

pub struct SudokuIterator {
    lines: io::Lines<io::BufReader<File>>,
}

pub struct SudokuGroupedIterator {
    sudoku_iterator: SudokuIterator,
    buffer_size: u32,
}

impl Iterator for SudokuIterator {
    type Item = SudokuPuzzle;

    fn next(&mut self) -> Option<SudokuPuzzle> {
        //Find first line with data:
        let first_line = self.re_init();
        first_line.as_ref()?;

        //Allocate memory for new puzzle:
        let mut puzzle: SudokuPuzzle = SudokuPuzzle::new();

        //Read first line:
        let line_data: String = first_line.unwrap();
        SudokuIterator::read_line(&line_data, &mut puzzle, 0);

        //Read other lines:
        for row in 1..PUZZLE_SIZE {
            let next_line = self.lines.next();
            next_line.as_ref()?;
            let next_line_data = next_line.unwrap();
            if next_line_data.is_err() {
                return None;
            }
            SudokuIterator::read_line(&next_line_data.unwrap(), &mut puzzle, row);
        }
        Some(puzzle)
    }
}

impl SudokuIterator {
    pub fn new(lines: io::Lines<io::BufReader<File>>) -> Self {
        SudokuIterator { lines }
    }

    fn re_init(&mut self) -> Option<String> {
        let mut maybe_cur_line = self.lines.next();
        let mut rs = None;
        while maybe_cur_line.is_some() {
            match maybe_cur_line.unwrap() {
                Err(_) => {
                    break;
                }
                Ok(cur_line_string) => {
                    if cur_line_string.is_empty()
                        || cur_line_string.starts_with(NEW_SUDOKU_SEPARATOR)
                    {
                        maybe_cur_line = self.lines.next();
                    } else {
                        rs = Some(cur_line_string);
                        break;
                    }
                }
            };
        }
        rs
    }

    fn read_line(line_data: &str, puzzle: &mut SudokuPuzzle, row: usize) {
        //Read string into puzzle
        let mut chars_of_line = line_data.chars();
        for col in 0..PUZZLE_SIZE {
            let ch = chars_of_line.next();
            if let Some(char_unwrapped) = ch {
                let number: u8 = if '0' < char_unwrapped && char_unwrapped <= '9' {
                    let char_as_u8: u8 = (char_unwrapped as i32 - '0' as i32) as u8; //result is in [0 - 9]
                    char_as_u8
                } else {
                    0
                };
                puzzle.set(row, col, number);
            } else {
                break;
            }
        }
    }
}

impl Iterator for SudokuGroupedIterator {
    type Item = Vec<SudokuPuzzle>;

    fn next(&mut self) -> Option<Vec<SudokuPuzzle>> {
        let mut buffer: Vec<SudokuPuzzle> = Vec::new();
        for _index in 0..self.buffer_size {
            if let Some(sudoku) = self.sudoku_iterator.next() {
                buffer.push(sudoku);
            }
        }
        if buffer.is_empty() {
            None
        } else {
            Some(buffer)
        }
    }
}

impl SudokuGroupedIterator {
    pub fn grouped(sudoku_iterator: SudokuIterator, buffer_size: u32) -> Self {
        SudokuGroupedIterator {
            sudoku_iterator,
            buffer_size,
        }
    }
}
