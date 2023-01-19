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
package main

import (
	"testing"

	"github.com/fmantz/sudoku/golang/algo"
	"github.com/fmantz/sudoku/golang/io"
)

func TestSolveOneSudoku(t *testing.T) {
	checkSolve("./test/one_sudoku.txt", t)
}

func TestSolve50SudokusFromProjectEuler(t *testing.T) {
	checkSolve("./test/p096_sudoku.txt", t)
}

func TestSolve10SudokusGeneratedWithQQWing(t *testing.T) {
	checkSolve("./test/sudoku.txt", t)
}

func checkSolve(path string, t *testing.T) {
	puzzles, errRead := io.Read(path)
	if errRead != nil {
		t.Error("could not read sudokus")
	}
	for sudokuNumber := 1; puzzles.HasNext(); sudokuNumber++ {
		sudoku := puzzles.Next()
		input := sudoku.String()
		sudoku.Solve()
		output := sudoku.String()
		if !checkSolution(sudoku) {
			t.Errorf("Sudoku %d is not solved:\n%s", sudokuNumber, sudoku.ToPrettyString())
		}
		if len(input) != len(output) {
			t.Error("Sudoku strings have not same length")
		}
		inputRunes := []rune(input)
		outputRunes := []rune(output)
		for i := 0; i < len(input); i++ {
			inChar := inputRunes[i]
			outChar := outputRunes[i]
			if !isBlank(inChar) {
				if inChar != outChar {
					t.Errorf("Sudoku %d has been changed:\n%s", sudokuNumber, sudoku.ToPrettyString())
				}
			}
		}
	}
}

func isBlank(c rune) bool {
	return '0' <= c || c > '9'
}

/**
 * @param row in [0,9]
 */
func isRowOk(sudoku [][]uint8, row int) bool {
	bits := algo.NewSudokuBitSet()
	checkRow(sudoku, row, bits)
	return bits.IsAllFoundNumbersUnique() && bits.IsAllNumbersFound()
}

func checkRow(sudoku [][]uint8, row int, bits *algo.SudokuBitSet) {
	selectRow := sudoku[row]
	for col := 0; col < algo.PUZZLE_SIZE; col++ {
		value := selectRow[col]
		bits.SaveValue(value)
	}
}

/**
 * @param col in [0,9]
 */
func isColOk(sudoku [][]uint8, col int) bool {
	bits := algo.NewSudokuBitSet()
	checkCol(sudoku, col, bits)
	return bits.IsAllFoundNumbersUnique() && bits.IsAllNumbersFound()
}

func checkCol(sudoku [][]uint8, col int, bits *algo.SudokuBitSet) {
	for row := 0; row < algo.PUZZLE_SIZE; row++ {
		value := sudoku[row][col]
		bits.SaveValue(value)
	}
}

/**
 * @param rowSquareIndex in [0,2]
 * @param colSquareIndex in [0,2]
 */
func isSquareOk(sudoku [][]uint8, rowSquareIndex, colSquareIndex int) bool {
	bits := algo.NewSudokuBitSet()
	checkSquare(sudoku, rowSquareIndex, colSquareIndex, bits)
	return bits.IsAllFoundNumbersUnique() && bits.IsAllNumbersFound()
}

func checkSquare(sudoku [][]uint8, rowSquareIndex, colSquareIndex int, bits *algo.SudokuBitSet) {
	rowSquareOffset := rowSquareIndex * algo.SQUARE_SIZE
	colSquareOffset := colSquareIndex * algo.SQUARE_SIZE
	for row := 0; row < algo.SQUARE_SIZE; row++ {
		for col := 0; col < algo.SQUARE_SIZE; col++ {
			value := sudoku[row+rowSquareOffset][col+colSquareOffset]
			bits.SaveValue(value)
		}
	}
}

func checkSolution(sudokuPuzzle *algo.SudokuPuzzle) bool {
	sudoku := makeSudoku2DArray(sudokuPuzzle)
	for i := 0; i < algo.PUZZLE_SIZE; i++ {
		rs := isRowOk(sudoku, i) && isColOk(sudoku, i) && isSquareOk(sudoku, i/algo.SQUARE_SIZE, i%algo.SQUARE_SIZE)
		if !rs {
			return false
		}
	}
	return true
}

func makeSudoku2DArray(sudokuPuzzle *algo.SudokuPuzzle) [][]uint8 {
	sudoku := make([][]uint8, algo.PUZZLE_SIZE)
	for row := 0; row < algo.PUZZLE_SIZE; row++ {
		nextRow := make([]uint8, algo.PUZZLE_SIZE)
		sudoku[row] = nextRow
		for col := 0; col < algo.PUZZLE_SIZE; col++ {
			nextRow[col] = sudokuPuzzle.Get(row, col)
		}
	}
	return sudoku
}
