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

var (
	pPUZZLE = [][]uint8{
		{0, 0, 0, 0, 0, 0, 6, 7, 0},
		{7, 0, 0, 0, 0, 0, 0, 0, 3},
		{0, 9, 0, 6, 0, 0, 2, 0, 4},
		{9, 2, 0, 0, 4, 0, 3, 0, 0},
		{5, 7, 0, 3, 2, 0, 0, 6, 0},
		{6, 3, 0, 9, 0, 0, 0, 0, 5},
		{0, 0, 0, 0, 0, 0, 7, 4, 9},
		{0, 4, 0, 0, 0, 0, 5, 1, 2},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
)

func BenchmarkSolve(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sudoku := NewSudokuPuzzle()
		for row := 0; row < PUZZLE_SIZE; row++ {
			for col := 0; col < PUZZLE_SIZE; col++ {
				sudoku.Set(row, col, pPUZZLE[row][col])
			}
		}
		sudoku.Solve()
	}
}
