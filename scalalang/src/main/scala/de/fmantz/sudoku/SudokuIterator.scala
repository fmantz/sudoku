package de.fmantz.sudoku

import de.fmantz.sudoku.SudokuIO.{EmptyChar, QQWingEmptyChar}

import scala.collection.AbstractIterator

class SudokuIterator(val source:Iterator[String]) extends AbstractIterator[SudokuPuzzle]{

	private var curLine: String = ""

	reInit()

	private def reInit():Unit = {
		curLine = ""
		while (source.hasNext && curLine.isEmpty) {
			val readLine = source.next()
			if (!readLine.startsWith(SudokuIO.NewSudokuSeperator)) {
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
		val currentSudoku = SudokuPuzzle //static sudoku used with iterator!
		var currentRow = 0
		while (currentRow < SudokuPuzzle.Size) {
			readLine(currentSudoku, currentRow)
			currentRow += 1
			if(currentRow == SudokuPuzzle.Size){ //Puzzle read!
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
		for (col <- 0 until SudokuPuzzle.Size) {
			currentSuko.set(currentRow, col, rawValues(col))
		}
	}

}
