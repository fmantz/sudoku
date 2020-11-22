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
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;

//use crate::sudoku_constants::NEW_SUDOKU_SEPARATOR;
use crate::sudoku_iterator::SudokuIterator;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_puzzle::SudokuPuzzleData;

pub struct SudokuIO {} //no data!

impl SudokuIO {

    /**
      * Read usual 9x9 Suduko from text file
      */
    pub fn read(filename: &str) -> Result<SudokuIterator, String> {
        let path = Path::new(&filename);
        let display = path.display();
        let file_data = match File::open(&path) {
            Err(why) => return Err(format!("couldn't read {}: {}", display, why)),
            Ok(file) => io::BufReader::new(file).lines()
        };
        let puzzles: SudokuIterator = SudokuIterator::new(file_data);
        return Ok(puzzles);
    }

    // /**
    //  * Read Suduko to text file
    //  */
    // pub fn write(
    //     filename: &str,
    //     puzzles: SudokuIterator,
    //     f: fn(&u32, &mut SudokuPuzzleData) -> (),
    // ) -> Result<(), String> {
    //     let path = Path::new(filename);
    //     let display = path.display();
    //     let write_file = match File::create(&path) {
    //         Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
    //         Ok(file) => file
    //     };
    //     let mut writer = BufWriter::new(&write_file);
    //     let mut i: u32 = 0;
    //     for mut puzzle in puzzles {
    //         i += 1;
    //         f(&i, &mut puzzle);
    //         writeln!(&mut writer, "{} {}", NEW_SUDOKU_SEPARATOR, i);
    //         writeln!(&mut writer, "{}\n", puzzle.to_string());
    //         match writer.flush() {
    //             Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
    //             Ok(()) => () /*do nothing */
    //         }
    //     }
    //     return Ok(());
    // }

    pub fn write_qqwing(
        filename: &str,
        puzzles: SudokuIterator,
        f: fn(&u32, &mut SudokuPuzzleData) -> (),
    ) -> Result<(), String> {
        let path = Path::new(filename);
        let display = path.display();
        let write_file = match File::create(&path) {
            Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
            Ok(file) => file
        };
        let mut writer = BufWriter::new(&write_file);
        let mut i: u32 = 0;
        for mut puzzle in puzzles {
            i +=1;
            f(&i, &mut puzzle); //do someting!
            let write_rs = writeln!(&mut writer, "{}\n", puzzle.to_string());
            match write_rs {
                Err(error) => {
                    panic!("Problem with saving solved puzzle: {:?}", error);
                }
                Ok(()) => { /* do nothing */ }
            };
            match writer.flush() {
                Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
                Ok(()) => ()/*do nothing */
            }
        }
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{self, BufRead};
    use std::path::{MAIN_SEPARATOR, Path};
    use std::path::PathBuf;

    use crate::sudoku_constants::NEW_SUDOKU_SEPARATOR;
    use crate::sudoku_io::SudokuIO;
    use crate::sudoku_puzzle::SudokuPuzzle;
    use crate::sudoku_puzzle::SudokuPuzzleData;

    #[test]
    fn read_should_correctly_parse_sudokus() -> () {
        let mut dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push(format!("test{}resources{}{}", MAIN_SEPARATOR, MAIN_SEPARATOR, "p096_sudoku.txt").to_string());
        let filename: &str = dir.as_os_str().to_str().unwrap();
        let expected_rs: Vec<String> = match read_file(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles
        };
        let rs: Vec<SudokuPuzzleData> = match SudokuIO::read(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles.collect()
        };
        for (index, read) in rs.iter().enumerate() {
            assert_eq!(read.to_string(), expected_rs[index]);
        }
        assert_eq!(expected_rs.len(), 51);
    }

    #[test]
    fn read_should_read_correct_number_of_documents() -> () {
        let mut dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push(format!("test{}resources{}{}", MAIN_SEPARATOR, MAIN_SEPARATOR, "sudoku.txt").to_string());
        let filename: &str = dir.as_os_str().to_str().unwrap();
        let expected_length: usize = match read_file(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles.len()
        };
        let read_length: usize = match SudokuIO::read(filename) {
            Err(why) => panic!("{}", why),
            Ok(puzzles) => puzzles.count()
        };
        assert_eq!(read_length, expected_length);
    }

    pub fn read_file(filename: &str) -> Result<Vec<String>, String> {
        let mut rs: Vec<String> = Vec::new();
        let mut buffer: Vec<String> = Vec::new();
        let path = Path::new(&filename);
        let display = path.display();
        let file_data = match File::open(&path) {
            Err(why) => return Err(format!("couldn't read {}: {}", display, why)),
            Ok(file) => io::BufReader::new(file).lines()
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
        return Ok(rs);
    }
}
