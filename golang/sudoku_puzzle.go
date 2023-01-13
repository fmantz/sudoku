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
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

type SudokuPuzzle struct {
	myIsSolvable bool
	myIsSolved   bool
	puzzle       [CELL_COUNT]uint8
}

func NewSudokuPuzzle() *SudokuPuzzle {
	var p SudokuPuzzle
	p.myIsSolvable = true
	return &p
}

func (p *SudokuPuzzle) Get(row, col int) uint8 {
	return p.puzzle[getSingleArrayIndex(row, col)]
}

func (p *SudokuPuzzle) Set(row, col int, value uint8) {
	p.puzzle[getSingleArrayIndex(row, col)] = value
}

func getSingleArrayIndex(row, col int) int {
	return row*PUZZLE_SIZE + col
}

/**
 * solves the sudoku by a simple non-recursive backtracking algorithm (brute force)
 * (own simple solution, its an algorithm which may be ported to CUDA or OpenCL)
 * to get a faster result use e.g. https://github.com/Emerentius/sudoku
 */
func (p *SudokuPuzzle) Solve() bool {
	// Early out:
	if !p.myIsSolvable || p.myIsSolved {
		return p.myIsSolved
	}

	// Temporary memory to compute solution:
	var puzzleSorted, indices [CELL_COUNT]uint8
	var rowNums, colNums, squareNums [PUZZLE_SIZE]uint16
	p.findAllPossibleValuesForEachEmptyCell(rowNums[:], colNums[:], squareNums[:])
	p.preparePuzzleForSolving(puzzleSorted[:], indices[:], rowNums[:], colNums[:], squareNums[:])

	if p.myIsSolvable && !p.myIsSolved {
		p.findSolutionNonRecursively(puzzleSorted[:], indices[:], rowNums[:], colNums[:], squareNums[:])
	}

	return p.myIsSolved
}

func (p *SudokuPuzzle) findAllPossibleValuesForEachEmptyCell(rowNums, colNums, squareNums []uint16) {
	for i := 0; i < CELL_COUNT; i++ {
		curValue := p.puzzle[i]
		if curValue > 0 {
			p.saveValueForCellAndCheckIsSolvable(curValue, i, rowNums, colNums, squareNums)
		}
	}
}

func (p *SudokuPuzzle) preparePuzzleForSolving(puzzleSorted, indices []uint8, rowNums, colNums, squareNums []uint16) {
	var numberOffSets [PUZZLE_SIZE + 2]uint8 // counts 0 - 9 + 1 offset = puzzleSize + 2 (9 + 2)
	for i := 0; i < CELL_COUNT; i++ {
		countOfIndex := p.getPossibleCounts(i, rowNums, colNums, squareNums)
		numberOffSets[countOfIndex+1]++
	}
	p.myIsSolved = numberOffSets[1] == CELL_COUNT // all cells have already a solution!
	for i := 1; i < (PUZZLE_SIZE + 2); i++ {
		// correct offsets
		numberOffSets[i] += numberOffSets[i-1]
	}
	for i := 0; i < CELL_COUNT; i++ {
		countOfIndex := p.getPossibleCounts(i, rowNums, colNums, squareNums)
		offSet := numberOffSets[countOfIndex]
		indices[offSet] = uint8(i)
		numberOffSets[countOfIndex]++
	}
	p.sortPuzzle(puzzleSorted, indices) // avoid jumping in the puzzle array
}

func (p *SudokuPuzzle) findSolutionNonRecursively(puzzleSorted, indices []uint8, rowNums, colNums, squareNums []uint16) {
	var indicesCurrent [CELL_COUNT]int8
	for i := 0; i < CELL_COUNT; i++ {
		indicesCurrent[i] = -1
	}
	i := 0
	for i < CELL_COUNT {
		curValue := puzzleSorted[i]
		if curValue == 0 {
			// Is not given?

			// Is there a current guess possible?
			puzzleIndex := indices[i]
			rowIndex := calculateRowIndex(int(puzzleIndex))
			colIndex := calculateColIndex(int(puzzleIndex))
			squareIndex := calculateSquareIndex(rowIndex, colIndex)
			possibleNumberIndex := rowNums[rowIndex] | colNums[colIndex] | squareNums[squareIndex]
			nextNumberIndex := uint8(indicesCurrent[i] + 1)

			if nextNumberIndex < BITSET_LENGTH[possibleNumberIndex] {
				// next possible number to try found:
				nextNumbers := BITSET_ARRAY[possibleNumberIndex]
				nextNumber := nextNumbers[nextNumberIndex]
				puzzleSorted[i] = nextNumber

				// save value for cell:
				checkBit := uint16(1) << (nextNumber - 1)
				rowNums[rowIndex] |= checkBit
				colNums[colIndex] |= checkBit
				squareNums[squareIndex] |= checkBit

				indicesCurrent[i] = int8(nextNumberIndex) // success
				i++                                       // go to next cell
			} else {
				// backtrack:
				indicesCurrent[i] = -1 // forget last index for position i
				i--                    // not given values are in the head of myIndices, there we can simply go one step back!
				lastInvalidTry := puzzleSorted[i]
				lastPuzzleIndex := int(indices[i])
				puzzleSorted[i] = 0 // find in the next step a new solution for i

				// revert last value:
				lastRowIndex := calculateRowIndex(lastPuzzleIndex)
				lastColIndex := calculateColIndex(lastPuzzleIndex)
				lastSquareIndex := calculateSquareIndex(rowIndex, colIndex)
				lastCheckBit := uint16(1) << (lastInvalidTry - 1)
				rowNums[lastRowIndex] ^= lastCheckBit
				colNums[lastColIndex] ^= lastCheckBit
				squareNums[lastSquareIndex] ^= lastCheckBit
			}
		} else {
			i++
		}
	}
}

func (p *SudokuPuzzle) sortPuzzle(puzzleSorted, indices []uint8) {
	for i := 0; i < CELL_COUNT; i++ {
		puzzleSorted[i] = p.puzzle[indices[i]]
	}
}

func (p *SudokuPuzzle) fillPositions(puzzleSorted, indices []uint8) {
	for i := 0; i < CELL_COUNT; i++ {
		p.puzzle[indices[i]] = puzzleSorted[i]
	}
}

func (p *SudokuPuzzle) saveValueForCellAndCheckIsSolvable(value uint8, index int, rowNums, colNums, squareNums []uint16) {
	rowIndex := calculateRowIndex(index)
	colIndex := calculateColIndex(index)
	squareIndex := calculateSquareIndex(rowIndex, colIndex)
	checkBit := uint16(1) << (value - 1)
	p.myIsSolvable = p.myIsSolvable && setAndCheckBit(checkBit, rowNums, rowIndex)
	p.myIsSolvable = p.myIsSolvable && setAndCheckBit(checkBit, colNums, colIndex)
	p.myIsSolvable = p.myIsSolvable && setAndCheckBit(checkBit, squareNums, squareIndex)
}

func setAndCheckBit(checkBit uint16, array []uint16, index int) bool {
	oldValue := array[index]
	array[index] |= checkBit
	return oldValue != array[index]
}

func calculateRowIndex(index int) int {
	return index / PUZZLE_SIZE
}

func calculateColIndex(index int) int {
	return index % PUZZLE_SIZE
}

func calculateSquareIndex(rowIndex, colIndex int) int {
	return rowIndex/SQUARE_SIZE*SQUARE_SIZE + colIndex/SQUARE_SIZE //attention: int arithmetic
}

func (p *SudokuPuzzle) getPossibleCounts(index int, rowNums, colNums, squareNums []uint16) uint8 {
	if p.puzzle[index] == 0 {
		rowIndex := calculateRowIndex(index)
		colIndex := calculateColIndex(index)
		squareIndex := calculateSquareIndex(rowIndex, colIndex)
		possibleNumberIndex := rowNums[rowIndex] | colNums[colIndex] | squareNums[squareIndex]
		return BITSET_LENGTH[possibleNumberIndex]
	} else {
		return 0
	}
}

func (p *SudokuPuzzle) ToPrettyString() string {
	var dottedLine = strings.Repeat("-", 4)
	const empty = "*"
	var buffer bytes.Buffer
	for row := 0; row < PUZZLE_SIZE; row++ {
		from := row * PUZZLE_SIZE
		until := from + PUZZLE_SIZE
		currentRow := p.puzzle[from:until]
		var formattedRow bytes.Buffer
		for col, colValue := range currentRow {
			var rs string
			if colValue == 0 {
				rs = fmt.Sprintf(" % ", empty)
			} else {
				rs = fmt.Sprintf(" % ", colValue)
			}
			formattedRow.WriteString(rs)
			if col+1 < PUZZLE_SIZE && col%SQUARE_SIZE == 2 {
				formattedRow.WriteRune('|')
			}
		}
		buffer.WriteString(formattedRow.String())
		buffer.WriteRune('\n')
		if row < (PUZZLE_SIZE-1) && (row+1)%SQUARE_SIZE == 0 {
			buffer.WriteString(dottedLine)
			buffer.WriteRune('\n')
		}
	}
	return buffer.String()
}

func (p *SudokuPuzzle) String() string {
	var buffer bytes.Buffer
	for row := 0; row < PUZZLE_SIZE; row++ {
		from := row * PUZZLE_SIZE
		until := from + PUZZLE_SIZE
		currentRow := p.puzzle[from:until]
		for _, colValue := range currentRow {
			buffer.WriteString(strconv.Itoa(int(colValue)))
		}
		buffer.WriteRune('\n')
	}
	return buffer.String()
}
