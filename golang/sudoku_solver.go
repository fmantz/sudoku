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
	"sync"
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
	inputPath := args[1]
	var outputPath string
	if len(args) > 2 {
		outputPath = args[2]
	} else {
		outputPath = generateOutputPath(inputPath)
	}
	fmt.Printf("input: %s\n", inputPath)
	solveSudokus(inputPath, outputPath)
	duration := time.Since(start)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("All sudoku puzzles solved by simple backtracking algorithm in %s\n", &duration)
}

func generateOutputPath(inputPath string) (outputPath string) {
	defer func() {
		if r := recover(); r != nil {
			outputPath = fmt.Sprintf(".%ssudoku_solution.txt", string(os.PathSeparator))
		}
	}()
	myPath := path.Dir(inputPath)
	simpleFileName := path.Base(inputPath)
	outputPath = fmt.Sprintf("%s/SOLUTION_%s", myPath, simpleFileName)
	return outputPath
}

func solveSudokus(inputPath string, outputPath string) {
	puzzles, errRead := io.Read(inputPath)
	if errRead != nil {
		panic(errRead.Error())
	}
	// save in buffer:
	var puzzleBuffer []algo.SudokuPuzzle
	for puzzles.HasNext() {
		puzzle := puzzles.Next()
		puzzleBuffer = append(puzzleBuffer, *puzzle)
	}
	// solve in parallel:
	var wg sync.WaitGroup
	for i, _ := range puzzleBuffer {
		wg.Add(1)
		go solveCurrentSudoku(i, puzzleBuffer, &wg)
	}
	wg.Wait()
	errWrite := io.Write(outputPath, puzzleBuffer)
	if errWrite != nil {
		panic(errWrite.Error())
	}
}

func solveCurrentSudoku(index int, puzzles []algo.SudokuPuzzle, wg *sync.WaitGroup) {
	defer wg.Done()
	solved := puzzles[index].Solve()
	if !solved {
		fmt.Printf("Sudoku is unsolvable:\n%s", puzzles[index].ToPrettyString())
	}
}
