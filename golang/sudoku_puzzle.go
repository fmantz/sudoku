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
