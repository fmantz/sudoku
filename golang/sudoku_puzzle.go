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

func (p *SudokuPuzzle) Solve() bool {
	return false //TODO
}

func (p *SudokuPuzzle) findAllPossibleValuesForEachEmptyCell(rowNums, colNums, squareNums []uint16) {
	for i := 0; i < CELL_COUNT; i++ {
		curValue := p.puzzle[i]
		if curValue > 0 {
			p.saveValueForCellAndCheckIsSolvable(curValue, i, rowNums, colNums, squareNums)
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
