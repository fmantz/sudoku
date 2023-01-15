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
	"os"
	"strings"
	"testing"
)

func TestReadShouldCorrectlyParseSudokus(t *testing.T) {
	fileName := "../test/p096_sudoku.txt"
	expectedRs := readFile(fileName)
	var rs []string
	puzzles, err := Read(fileName)
	if err != nil {
		t.Error(err.Error())
	}
	for puzzles.HasNext() {
		rs = append(rs, puzzles.Next().String())
	}
	expectedLen := 51
	if len(expectedRs) != expectedLen {
		t.Errorf("testsetup is broken, expected %d strings", len(expectedRs))
	}
	if len(rs) != expectedLen {
		t.Errorf("rs should has %d strings but has %d", expectedLen, len(rs))
	}
	for i := 0; i < expectedLen; i++ {
		if strings.Compare(rs[i], expectedRs[i]) == 0 {
			t.Errorf("sudokus not correctly read, \nread:\n%s\nexpected:\n%s", rs[i], expectedRs[i])
		}
	}
}

func TestReadShouldReadCorrectNumberOfSudokus(t *testing.T) {
	fileName := "../test/p096_sudoku.txt"
	expectedLength := len(readFile(fileName))
	puzzles, err := Read(fileName)
	if err != nil {
		t.Error(err.Error())
	}
	var rs []string
	for puzzles.HasNext() {
		rs = append(rs, puzzles.Next().String())
	}
	if len(rs) != expectedLength {
		t.Errorf("read %d sudokus but expected to read %d", len(rs), expectedLength)
	}
}

func readFile(path string) []string {
	var rs []string
	var buffer []string
	file, err := os.Open(path)
	if err != nil {
		panic(err.Error())
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		curLine := scanner.Text()
		if len(curLine) == 0 || strings.HasPrefix(curLine, pNEW_SUDOKU_SEPARATOR) {
			lenBuffer := len(buffer)
			if lenBuffer > 0 {
				rs = append(rs, strings.Join(buffer, "\n"))
			}
			buffer = nil
		} else {
			buffer = append(buffer, curLine)
		}
	}
	lenBuffer := len(buffer)
	if lenBuffer > 0 {
		rs = append(rs, strings.Join(buffer, "\n"))
	}
	return rs
}
