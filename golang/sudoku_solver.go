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
	"fmt"
	"os"
	"path"
	"time"

	"github.com/fmantz/sudoku/golang/algo"
	"github.com/fmantz/sudoku/golang/io"
)

func main() {
	args := os.Args
	if len(args) < 2 {
		println(">SudokuSolver inputFile [outputFile]")
		println("-First argument must be path to sudoku puzzles!")
		println("-Second argument can be output path for sudoku puzzles solution!")
		os.Exit(0)
	}
	start := time.Now()
	inputFileName := args[1]
	var outputFileName string
	if len(args) > 2 {
		outputFileName = args[2]
	} else {
		outputFileName = generateOutputPath(inputFileName)
	}
	fmt.Printf("input: %s\n", inputFileName)
	solveSudokus(inputFileName, outputFileName)
	duration := time.Since(start)
	fmt.Printf("output: %s\n", outputFileName)
	fmt.Printf("All sudoku puzzles solved by simple backtracking algorithm in %s\n", &duration)
}

func generateOutputPath(inputFileName string) (outputFileName string) {
	defer func() {
		if r := recover(); r != nil {
			outputFileName = fmt.Sprintf(".%ssudoku_solution.txt", os.PathSeparator)
		}
	}()
	myPath := path.Dir(inputFileName)
	simpleFileName := path.Base(inputFileName)
	outputFileName = fmt.Sprintf("%s/SOLUTION_%s", myPath, simpleFileName)
	return outputFileName
}

func solveSudokus(inputFileName string, outputFileName string) {
	puzzles, errRead := io.Read(inputFileName)
	if errRead != nil {
		panic(errRead.Error())
	}
	var solvedPuzzles []algo.SudokuPuzzle
	for puzzles.HasNext() {
		puzzle := puzzles.Next()
		solveCurrentSudoku(puzzle)
		solvedPuzzles = append(solvedPuzzles, *puzzle)
	}
	errWrite := io.Write(outputFileName, solvedPuzzles)
	if errWrite != nil {
		panic(errWrite.Error())
	}
}

func solveCurrentSudoku(sudoku *algo.SudokuPuzzle) {
	solved := sudoku.Solve()
	if !solved {
		fmt.Printf("Sudoku is unsolvable:\n%s", sudoku.ToPrettyString())
	}
}
