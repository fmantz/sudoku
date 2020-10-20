use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::{Duration, Instant};
use crate::sudoku_iterator::Puzzles;
use std::fmt::format;

pub struct SudokuIO{} //no data!

impl SudokuIO {

    /**
      * Read usual 9x9 Suduko from text file
      */
    pub fn read (filename: &str) -> Result<Puzzles, String> {
        let file_data = SudokuIO::read_lines(filename);
        match file_data {
            Err(myerror) => {
                return Err(format!("Could not read file {}, because '{}'\n", filename, myerror));
            }
            Ok(lines) => {
               let mut puzzles : Puzzles = Puzzles::new(lines);
               return Ok(puzzles);
            }
        }
    }

    fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
        where P: AsRef<Path>, {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
    }

}