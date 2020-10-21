use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use crate::sudoku_iterator::SudokuIterator;
use std::fmt::format;
use crate::sudoku_constants::NEW_SUDOKU_SEPARATOR;
use crate::sudoku_puzzle::SudokuPuzzle;
use crate::sudoku_puzzle::SudokuPuzzleData;
use std::iter::Map;

pub struct SudokuIO{} //no data!

impl SudokuIO {

    /**
      * Read usual 9x9 Suduko from text file
      */
    pub fn read (filename: &str) -> Result<SudokuIterator, String> {
        let path = Path::new(&filename);
        let display = path.display();
        let file_data = match File::open(&path) {
            Err(why) => return Err(format!("couldn't read {}: {}", display, why)),
            Ok(file) => io::BufReader::new(file).lines()
        };
        let mut puzzles : SudokuIterator = SudokuIterator::new(file_data);
        return Ok(puzzles);
    }

    /**
     * Read Suduko to text file
     */
    pub fn write(
        filename: &str,
        puzzles: SudokuIterator,
        f: fn(& mut SudokuPuzzleData) -> ()
    ) -> Result<(), String> {
        let path = Path::new(filename);
        let display = path.display();
        let write_file = match File::create(&path) {
            Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
            Ok(file) => file
        };
        let mut writer = BufWriter::new(&write_file);
        let mut i:u32 = 0;
        for mut puzzle in puzzles {
            i += 1;
            f(&mut puzzle);
            writeln!(& mut writer, "{} {}", NEW_SUDOKU_SEPARATOR, i);
            writeln!(& mut writer, "{}", puzzle.to_string());
            match writer.flush() {
                Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
                Ok(()) => () /*do nothing */
            }
        }
        return Ok(());
    }

    pub fn write_qqwing(
        filename: &str,
        puzzles: SudokuIterator,
        f: fn(&mut SudokuPuzzleData) -> ()
    ) -> Result<(), String> {
        let path = Path::new(filename);
        let display = path.display();
        let write_file = match File::create(&path) {
            Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
            Ok(file) => file
        };
        let mut writer = BufWriter::new(&write_file);
        for mut puzzle in puzzles {
            f(&mut puzzle); //do someting!
            println!("{}\n ", puzzle.to_string());
            //writeln!(& mut writer, "{}", &f(puzzle).to_string());
            /*
            match writer.flush() {
                Err(why) => return Err(format!("couldn't create {}: {}", display, why)),
                Ok(()) => ()/*do nothing */
            }
            */
        }
        return Ok(());
    }
}

#[cfg(test)]
mod tests {

    use crate::sudoku_io::SudokuIO;
    use crate::sudoku_puzzle::SudokuPuzzleData;
    use crate::sudoku_puzzle::SudokuPuzzle;

    #[test]
    fn test() -> () {
        let puzzles = SudokuIO::read("/home/florian/temp/sudoku2.txt");
        for p in puzzles.unwrap() {
            println!("{}", p.to_string());
        }
    }

}