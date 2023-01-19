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
	for iter.curPostion < iter.len-1 {
		curLine = iter.lines[iter.curPostion]
		if len(curLine) == 0 || strings.HasPrefix(curLine, pNEW_SUDOKU_SEPARATOR) {
			iter.curPostion++
		} else {
			break
		}
	}
}

func (iter *SudokuIterator) HasNext() bool {
	if iter.curPostion == 0 {
		iter.reInit()
	}
	return iter.curPostion < iter.len && len(iter.lines[iter.curPostion]) >= algo.PUZZLE_SIZE
}

func (iter *SudokuIterator) Next() *algo.SudokuPuzzle {
	if iter.curPostion == 0 {
		iter.reInit()
	}
	currentSudoku := algo.NewSudokuPuzzle()
	for currentRow := 0; currentRow < algo.PUZZLE_SIZE; currentRow++ {
		if currentRow < algo.PUZZLE_SIZE && iter.curPostion >= iter.len {
			panic("incomplete puzzle found!")
		}
		curLine := iter.lines[iter.curPostion]
		readLine(currentSudoku, currentRow, curLine)
		iter.curPostion++
	}
	iter.reInit()
	return currentSudoku
}

func readLine(p *algo.SudokuPuzzle, currentRow int, curLine string) {
	curLineAsSlice := []rune(curLine)
	for col := 0; col < min(algo.PUZZLE_SIZE, len(curLineAsSlice)); col++ {
		c := curLineAsSlice[col]
		var num uint8
		if '0' < c && c <= '9' {
			num = uint8(c) - uint8('0')
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
