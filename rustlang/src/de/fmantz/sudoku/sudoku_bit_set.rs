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
#[cfg(test)]
pub mod tests {

    use crate::sudoku_constants::{BITSET_ARRAY, CHECK_BITS};

    pub struct SudokuBitSet {
        bits: u16,
        not_found_before: bool,
    }

    #[allow(dead_code)] //unfortantely some of the methods to delete are still used in tests!
    impl SudokuBitSet {
        pub fn new() -> Self {
            SudokuBitSet {
                bits: 0,
                not_found_before: true,
            }
        }

        pub fn new_with_data(data: u16) -> Self {
            SudokuBitSet {
                bits: data,
                not_found_before: true,
            }
        }

        pub fn save_value(&mut self, value: u8) -> () {
            if value > 0 {
                let check_bit: u16 = 1 << (value - 1); //set for each number a bit by index from left, number 1 has index zero
                self.not_found_before = self.not_found_before && SudokuBitSet::first_match(self.bits, check_bit);
                self.bits = self.bits | check_bit;
            }
        }

        #[inline]
        fn first_match(bits: u16, check_bit: u16) -> bool {
            bits & check_bit == 0
        }

        pub fn is_all_numbers_found(&self) -> bool {
            self.bits == CHECK_BITS
        }

        // pub fn has_solution(&self) -> bool {
        //     return !self.is_all_numbers_found();
        // }

        pub fn is_found_numbers_unique(&self) -> bool {
            self.not_found_before
        }

        // pub fn is_solution(&self, sol: u8) -> bool {
        //     if sol > 0 {
        //         let check_bit: u16 = 1 << sol - 1;
        //         return (self.bits & check_bit) == 0;
        //     } else {
        //         return false;
        //     };
        // }

        // pub fn to_string(&self) -> String {
        //     return format!("BITS={:#011b}", self.bits);
        // }

        pub fn possible_numbers(&self) -> &[u8] {
            BITSET_ARRAY[self.bits as usize]
        }
    }


//    #[test]
//    fn to_string_should_work_correctly() -> () {
//        let mut test_object: SudokuBitSet = SudokuBitSet::new();
//        test_object.save_value(5);
//        assert_eq!(test_object.to_string(), "BITS=0b000010000")
//    }

    #[test]
    fn is_found_numbers_unique_should_work_correctly() -> () {
        let mut test_object: SudokuBitSet = SudokuBitSet::new();
        assert_eq!(test_object.is_found_numbers_unique(), true);
        test_object.save_value(5);
        assert_eq!(test_object.is_found_numbers_unique(), true);
        test_object.save_value(5);
        assert_eq!(test_object.is_found_numbers_unique(), false);
    }

    #[test]
    fn is_all_numbers_found_should_work_correctly() -> () {
        let mut test_object: SudokuBitSet = SudokuBitSet::new();

        assert_eq!(test_object.is_all_numbers_found(), false);

        for i in 1..9 { //1 until 9
            test_object.save_value(i);
            assert_eq!(test_object.is_all_numbers_found(), false);
        }

        test_object.save_value(9);
        assert_eq!(test_object.is_all_numbers_found(), true);
    }
}
