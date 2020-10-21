use std::fs::File;
use std::io::{self, Error};
use crate::sudoku_puzzle::SudokuPuzzleData;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_constants::NEW_SUDOKU_SEPARATOR;
use crate::sudoku_constants::EMPTY_CHAR;
use crate::sudoku_constants::QQWING_EMPTY_CHAR;
use crate::sudoku_constants::PUZZLE_SIZE;
use std::str::Chars;

pub struct SudokuIterator {
    lines: io::Lines<io::BufReader<File>>
}

impl Iterator for SudokuIterator {

    type Item = SudokuPuzzleData;

    fn next(&mut self) -> Option<SudokuPuzzleData> {

        //Find first line with data:
        let first_line = self.re_init();
        if first_line.is_none() {
            return None
        }

        //Allocate memory for new puzzle:
        let mut puzzle: SudokuPuzzleData = SudokuPuzzleData::new();

        //Read first line:
        let mut line_data : String  = first_line.unwrap();
        SudokuIterator::read_line(& mut line_data, &mut puzzle, 0);

        //Read other lines:
        for row in 1.. (PUZZLE_SIZE - 1) {
            let next_line = self.lines.next();
            if next_line.is_none() {
                return None;
            }
            let next_line_data = next_line.unwrap();
            if next_line_data.is_err() {
                return None;
            }
            SudokuIterator::read_line(&mut next_line_data.unwrap(), &mut puzzle, row);
        }
        return Some(puzzle);
    }
}

impl SudokuIterator {

    pub fn new(lines: io::Lines<io::BufReader<File>>) -> Self {
        SudokuIterator {
            lines: lines
        }
    }

    fn re_init(&mut self) -> Option<String> {
        let mut maybe_cur_line = self.lines.next();
        let mut rs = None;
        while maybe_cur_line.is_some() {
            match maybe_cur_line.unwrap() {
                Err(_) => {
                    maybe_cur_line = None;
                    break;
                },
                Ok(cur_line_string) => {
                    if cur_line_string.is_empty() || cur_line_string.starts_with(NEW_SUDOKU_SEPARATOR) {
                        maybe_cur_line = self.lines.next();
                    } else {
                        rs = Some(cur_line_string);
                        break;
                    }
                }
            };
        }
        return rs;
    }

    fn read_line(line_data: &mut String, puzzle: &mut SudokuPuzzleData, row: usize) {
        //Read string into puzzle
        let mut chars_of_line: Chars = line_data.chars();
        for col in 0..PUZZLE_SIZE {
            let ch = chars_of_line.next();
            if ch.is_some() {
                let char_unwrapped = ch.unwrap();
                let number: u8 = if char_unwrapped == QQWING_EMPTY_CHAR {
                    0
                } else {
                    (char_unwrapped as i32 - EMPTY_CHAR as i32) as u8 //result is in [0 - 9]
                };
                puzzle.set(row, col, number);
            }
        }
    }
}
