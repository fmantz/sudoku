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

import de.fmantz.sudoku.SudokuConstants.{EmptyChar, QQWingEmptyChar, NewSudokuSeparator}

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
		val currentSudoku = new SudokuPuzzleImpl() //static sudoku used with iterator!
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

	private def readLine(currentSuduko: SudokuPuzzle, currentRow: Int): Unit = {
		val normalizedLine = curLine.replace(QQWingEmptyChar, EmptyChar).trim
		val rawValues = normalizedLine.toCharArray.map(_ - EmptyChar)
		for (col <- 0 until SudokuConstants.PuzzleSize) {
			currentSuduko.set(currentRow, col, rawValues(col))
		}
	}

}
