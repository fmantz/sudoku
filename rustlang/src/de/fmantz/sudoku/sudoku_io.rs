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
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;

use crate::sudoku_iterator::SudokuIterator;
use crate::sudoku_puzzle::SudokuPuzzle;

pub struct SudokuIO {} //no data!

impl SudokuIO {
    /**
     * Read usual 9x9 Suduko from text file
     */
    pub fn read(path: &str) -> Result<SudokuIterator, String> {
        let my_path = Path::new(&path);
        let display = my_path.display();
        let file_data = match File::open(&my_path) {
            Err(why) => return Err(format!("couldn't read {}: {}", display, why)),
            Ok(file) => io::BufReader::new(file).lines(),
        };
        let puzzles: SudokuIterator = SudokuIterator::new(file_data);
        Ok(puzzles)
    }

    pub fn write(path: &str, puzzles: Vec<SudokuPuzzle>) -> Result<(), String> {
        let my_path = Path::new(path);
        let display = my_path.display();
        let write_file = match OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&my_path)
        {
            Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
            Ok(file) => file,
        };
        let mut writer = BufWriter::new(&write_file);
        for puzzle in puzzles {
            let write_rs = writeln!(&mut writer, "{}\n", puzzle);
            match write_rs {
                Ok(()) => { /* do nothing */ }
                Err(error) => {
                    panic!("problem with saving solved puzzle: {:?}", error);
                }
            };
            match writer.flush() {
                Ok(()) => (), /*do nothing */
                Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{self, BufRead};
    use std::path::PathBuf;
    use std::path::{Path, MAIN_SEPARATOR};

    use crate::sudoku_constants::NEW_SUDOKU_SEPARATOR;
    use crate::sudoku_io::SudokuIO;
    use crate::sudoku_puzzle::SudokuPuzzle;

    #[test]
    fn read_should_correctly_parse_sudokus() {
        let mut dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push(format!(
            "test{}resources{}{}",
            MAIN_SEPARATOR, MAIN_SEPARATOR, "p096_sudoku.txt"
        ));
        let filename: &str = dir.as_os_str().to_str().unwrap();
        let expected_rs: Vec<String> = match read_file(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles,
        };
        let rs: Vec<SudokuPuzzle> = match SudokuIO::read(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles.collect(),
        };
        for (index, read) in rs.iter().enumerate() {
            assert_eq!(read.to_string(), expected_rs[index]);
        }
        assert_eq!(expected_rs.len(), 51);
    }

    #[test]
    fn read_should_read_correct_number_of_sudokus() {
        let mut dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push(format!(
            "test{}resources{}{}",
            MAIN_SEPARATOR, MAIN_SEPARATOR, "sudoku.txt"
        ));
        let filename: &str = dir.as_os_str().to_str().unwrap();
        let expected_length: usize = match read_file(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles.len(),
        };
        let read_length: usize = match SudokuIO::read(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles.count(),
        };
        assert_eq!(read_length, expected_length);
    }

    pub fn read_file(path: &str) -> Result<Vec<String>, String> {
        let mut rs: Vec<String> = Vec::new();
        let mut buffer: Vec<String> = Vec::new();
        let my_path = Path::new(&path);
        let display = my_path.display();
        let file_data = match File::open(&my_path) {
            Err(why) => return Err(format!("couldn't read {}: {}", display, why)),
            Ok(file) => io::BufReader::new(file).lines(),
        };
        file_data.for_each(|maybe_line| {
            let line = maybe_line.unwrap();
            if line.is_empty() || line.starts_with(NEW_SUDOKU_SEPARATOR) {
                if !buffer.is_empty() {
                    let puzzle = buffer.join("\n");
                    rs.push(puzzle)
                }
                buffer.clear();
            } else {
                buffer.push(line.trim().to_string());
            }
        });
        if !buffer.is_empty() {
            let puzzle = buffer.join("\n");
            rs.push(puzzle)
        }
        Ok(rs)
    }
}
