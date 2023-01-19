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
	"bufio"
	"fmt"
	"os"

	"github.com/fmantz/sudoku/golang/algo"
)

/**
 * Read usual 9x9 Suduko from text file
 */
func Read(path string) (*SudokuIterator, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't read %s: %s", path, err.Error())
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return NewSudokuIterator(lines), scanner.Err()
}

func Write(path string, puzzles []algo.SudokuPuzzle) error {
	// open output file
	fo, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("couldn't create %s: %s", path, err.Error())
	}

	// close fo on exit and check for its returned error
	defer func() {
		if err := fo.Close(); err != nil {
			panic(fmt.Errorf("couldn't create %s: %s", path, err.Error()))
		}
	}()

	wr := bufio.NewWriter(fo)
	for i := 0; i < len(puzzles); i++ {
		_, writeErr := wr.WriteString(fmt.Sprintln(puzzles[i].String()))
		if writeErr != nil {
			return fmt.Errorf("problem with saving solved puzzle: %s", writeErr.Error())
		}
	}
	wr.Flush()

	return nil
}
