/*
 * sudoku - Sudoku solver for comparison Golang with Scala and Rust
 *        - The motivation is explained in the README.md file in the top level folder.
 * Copyright (C) 2023 Florian Mantz
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
package algo

import (
	"testing"
)

//func TestToStringShouldWorkCorrectly(t *testing.T) {
//	testObject := NewSudokuBitSet()
//	testObject.SaveValue(5)
//	rs := testObject.String()
//	expectedRs := "BITS=000010000"
//	if rs != expectedRs {
//		t.Errorf("String() isNot '%s' BUT '%s'", expectedRs, rs)
//	}
//}

func TestFoundNumbersUniqueShouldWorkCorrecly(t *testing.T) {
	testObject := NewSudokuBitSet()
	if !testObject.IsAllFoundNumbersUnique() {
		t.Errorf("Initially IsAllFoundNumbersUnique() should be true")
	}
	testObject.SaveValue(5)
	if !testObject.IsAllFoundNumbersUnique() {
		t.Errorf("After storing value for the first time, IsAllFoundNumbersUnique() should be true")
	}
	testObject.SaveValue(5)
	if testObject.IsAllFoundNumbersUnique() {
		t.Errorf("After storing value for the second time, IsAllFoundNumbersUnique() should be false")
	}
}

func TestAllNumbersFoundShouldWorkCorrecly(t *testing.T) {
	testObject := NewSudokuBitSet()
	if testObject.IsAllNumbersFound() {
		t.Errorf("Initially IsAllNumbersFound() should be false")
	}
	for i := 1; i < 9; i++ {
		testObject.SaveValue(uint8(i))
		if testObject.IsAllNumbersFound() {
			t.Errorf("IsAllNumbersFound() should be false, after step %d", i)
		}
	}
	testObject.SaveValue(9)
	if !testObject.IsAllNumbersFound() {
		t.Errorf("IsAllNumbersFound() should be true")
	}
}
