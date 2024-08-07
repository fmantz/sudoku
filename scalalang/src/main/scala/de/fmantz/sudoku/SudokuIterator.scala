//scalastyle:off
/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.md file in the top level folder.
 * Copyright (C) 2020 Florian Mantz
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
//scalastyle:on
package de.fmantz.sudoku

import de.fmantz.sudoku.SudokuConstants.NewSudokuSeparator

import scala.collection.AbstractIterator

class SudokuIterator(val source:Iterator[String]) extends AbstractIterator[SudokuPuzzle]{

	private var curLine: String = ""

	reInit()

	private def reInit():Unit = {
		curLine = ""
		while (source.hasNext && curLine.isEmpty) {
			val readLine = source.next()
			if (!readLine.startsWith(NewSudokuSeparator)) {
				curLine = readLine
			} else {
				curLine = ""
			}
		}
	}

	override def hasNext: Boolean = {
		curLine.nonEmpty
	}

	override def next(): SudokuPuzzle = {
		val currentSudoku = new SudokuPuzzle()
		var currentRow = 0
		while (currentRow < SudokuConstants.PuzzleSize) {
			readLine(currentSudoku, currentRow)
			currentRow += 1
			if(currentRow == SudokuConstants.PuzzleSize){ //Puzzle read!
				reInit()
			} else {
				if(!source.hasNext){
					throw new IllegalArgumentException("incomplete puzzle found!")
				}
				curLine = source.next()
			}
		}
		currentSudoku
	}

	private def readLine(currentSudoku: SudokuPuzzle, currentRow: Int): Unit = {
		val length = math.min(SudokuConstants.PuzzleSize, curLine.length)
		var col = 0
		while (col < length) {
			val c = curLine.charAt(col)
			val num = if ('0' < c && c <= '9') {
				c - '0'
			} else {
				0
			}
			currentSudoku.set(currentRow, col, num.toByte)
			col+=1
		}
	}

}
