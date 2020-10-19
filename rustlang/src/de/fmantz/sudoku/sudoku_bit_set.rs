use crate::sudoku_puzzle::PUZZLE_SIZE;

const CHECK_BIT: u16 = !0 >> (16 - PUZZLE_SIZE); //binary: Size times "1"

pub struct SudokuBitSet {
    bits: u16,
    not_found_before: bool,
}

impl SudokuBitSet {
    pub fn new() -> Self {
        SudokuBitSet {
            bits: 0,
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
        return (bits & check_bit) == 0;
    }

    pub fn is_all_numbers_found(&self) -> bool {
        return self.bits == CHECK_BIT;
    }

    pub fn has_solution(&self) -> bool {
        return !SudokuBitSet::is_all_numbers_found(self);
    }

    pub fn is_found_numbers_unique(&self) -> bool {
        return self.not_found_before;
    }

    pub fn is_solution(&self, sol: u8) -> bool {
        if sol > 0 {
            let check_bit: u16 = 1 << sol - 1;
            return (self.bits & check_bit) == 0;
        } else {
            return false;
        };
    }

    pub fn to_string(&self) -> String {
        return format!("BITS={:#018b}", self.bits);
    }
}

#[cfg(test)]
mod tests {
    use crate::sudoku_bit_set::SudokuBitSet;

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