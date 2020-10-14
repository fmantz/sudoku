package de.fmantz.sudoku

import java.io.{File, PrintWriter}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object SudokuIO {

	final val NewSudokuSeperator: String = "Grid"
	final val EmptyChar: Char = '0'
	final val QQWingEmptyChar: Char = '.' //https://qqwing.com

	/**
	 * Read usual 9x9 Suduko from text file
	 */
	def read(fileName: String): IndexedSeq[SudokuPuzzle] = {
		val rs = ArrayBuffer.empty[SudokuPuzzle]
		var currentSuko: SudokuPuzzle = new SudokuPuzzle
		var currentRow = 0
		val source = Source.fromFile(fileName)
		for (line <- source.getLines) {
			if (line.isEmpty || line.startsWith(NewSudokuSeperator)) {
				if(currentSuko.nonEmpty){
					rs.append(currentSuko)
				}
				currentSuko = new SudokuPuzzle
				currentRow = 0
			} else {
				val normalizedLine = line.replace(QQWingEmptyChar, EmptyChar).trim
				val rawValues = normalizedLine.toCharArray.map(_ - EmptyChar)
				for (col <- 0 until SudokuPuzzle.Size) {
					currentSuko.set(currentRow, col, rawValues(col))
				}
				currentRow += 1
			}
		}
		if(currentSuko.nonEmpty){
			rs.append(currentSuko)
		}
		source.close
		rs.toIndexedSeq
	}

  /**
   * Read Suduko to text file
   */
  def write(fileName: String, puzzles: Seq[SudokuPuzzle]): Unit = {
		val writer = new PrintWriter(new File(fileName))
		try {
			val maxNumberLength = puzzles.length.toString.length
			val pattern = s"%0${maxNumberLength}d"
			puzzles.zipWithIndex.foreach({ case (sudoku, index) =>
        writer.println(s"$NewSudokuSeperator ${pattern.format(index + 1)}")
        writer.println(sudoku)
				writer.flush()
			})
		} finally {
			writer.close()
		}
	}

	def writeQQWing(fileName: String, puzzles: Seq[SudokuPuzzle]): Unit = {
		val writer = new PrintWriter(new File(fileName))
		try {
			puzzles.foreach({ sudoku =>
				writer.println(sudoku)
				writer.println()
				writer.flush()
			})
		} finally {
			writer.close()
		}
	}
}
