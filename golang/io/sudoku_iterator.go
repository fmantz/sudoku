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
package io

import (
	"fmt"
	"strings"

	"github.com/fmantz/sudoku/golang/algo"
)

const (
	pNEW_SUDOKU_SEPARATOR = "Grid"
)

type SudokuIterator struct {
	curPostion int
	len        int
	lines      []string
}

func NewSudokuIterator(lines []string) *SudokuIterator {
	return &SudokuIterator{0, len(lines), lines}
}

func (iter *SudokuIterator) reInit() {
	curLine := ""
	for iter.HasNext() && (len(curLine) == 0 || strings.HasPrefix(curLine, pNEW_SUDOKU_SEPARATOR)) {
		curLine = iter.lines[iter.curPostion]
		iter.curPostion++
	}
}

func (iter *SudokuIterator) HasNext() bool {
	return iter.curPostion < iter.len
}

func (iter *SudokuIterator) Next() *algo.SudokuPuzzle {
	currentSudoku := algo.NewSudokuPuzzle()
	for currentRow := 0; currentRow < algo.PUZZLE_SIZE; currentRow++ {
		curLine := iter.lines[iter.curPostion]
		readLine(currentSudoku, currentRow, curLine)
		if currentRow == algo.PUZZLE_SIZE {
			iter.reInit()
		} else {
			if !iter.HasNext() {
				panic("incomplete puzzle found!")
			}
			iter.curPostion++
		}
	}
	return currentSudoku
}

func readLine(p *algo.SudokuPuzzle, currentRow int, curLine string) {
	curLineAsSlice := []rune(curLine)
	for col := 0; col < min(algo.PUZZLE_SIZE, len(curLineAsSlice)); col++ {
		c := curLineAsSlice[col]
		var num uint8
		if '0' < c && c <= '9' {
			num = uint8(c) - uint8('0')
			fmt.Printf("FLO>%d\n", num)
		} else {
			num = 0
		}
		p.Set(currentRow, col, num)
	}
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
