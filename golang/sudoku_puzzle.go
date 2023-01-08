package main

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
