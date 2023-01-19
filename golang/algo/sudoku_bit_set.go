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

//import (
//	"fmt"
//)

const (
	CHECK_BITS = 511 // bit pattern 0000...111111111, 9 ones
)

type SudokuBitSet struct {
	bits           uint16
	notFoundBefore bool
}

func NewSudokuBitSet() *SudokuBitSet {
	return &SudokuBitSet{0, true}
}

func NewSudokuBitSetWithData(data uint16) *SudokuBitSet {
	return &SudokuBitSet{0, true}
}

func (b *SudokuBitSet) SaveValue(value uint8) {
	if value > 0 {
		checkBit := uint16(1) << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		b.notFoundBefore = b.notFoundBefore && firstMatch(b.bits, checkBit)
		b.bits |= checkBit
	}
}

func firstMatch(bits uint16, checkBit uint16) bool {
	return bits&checkBit == 0
}

func (b *SudokuBitSet) IsAllNumbersFound() bool {
	return b.bits == CHECK_BITS
}

//func (b *SudokuBitSet) HasSolution() bool {
//	return !b.IsAllNumbersFound()
//}

func (b *SudokuBitSet) IsAllFoundNumbersUnique() bool {
	return b.notFoundBefore
}

//func (b *SudokuBitSet) IsSolution(sol uint8) bool {
//	if sol > 0 {
//		checkBit := uint16(1) << (sol - 1)
//		return (b.bits & checkBit) == 0
//	} else {
//		return false
//	}
//}

//func (b *SudokuBitSet) String() string {
//	return fmt.Sprintf("BITS=%09b", b.bits)
//}

func (b *SudokuBitSet) PossibleNumbers() []uint8 {
	return pBITSET_ARRAY[b.bits]
}
