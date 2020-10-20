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

	private def readLine(currentSuko: SudokuPuzzle, currentRow: Int): Unit = {
		val normalizedLine = curLine.replace(QQWingEmptyChar, EmptyChar).trim
		val rawValues = normalizedLine.toCharArray.map(_ - EmptyChar)
		for (col <- 0 until SudokuConstants.PuzzleSize) {
			currentSuko.set(currentRow, col, rawValues(col))
		}
	}

}
