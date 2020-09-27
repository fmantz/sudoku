package de.fmantz.sudoku

import java.io.{File, PrintWriter}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object SudokuIO {

	private final val NewSudokuSeperator: String = "Grid"

	/**
	 * Read usual 9x9 Suduko from text file
	 */
	def read(fileName: String): IndexedSeq[SudokuPuzzle] = {
		val rs = ArrayBuffer.empty[SudokuPuzzle]
		var currentSuko: SudokuPuzzle = null
		var currentRow = 0
		val source = Source.fromFile(fileName)
		for (line <- source.getLines) {
			if (line.startsWith(NewSudokuSeperator)) {
				currentSuko = new SudokuPuzzle
				currentRow = 0
				rs.append(currentSuko)
			} else {
				val rowValues = line.trim.toCharArray.map(_ - '0')
				for (col <- 0 until SudokuPuzzle.Size) {
					currentSuko.set(currentRow, col, rowValues(col))
				}
				currentRow += 1
			}
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
			})
		} finally {
			writer.close()
		}
	}

}
