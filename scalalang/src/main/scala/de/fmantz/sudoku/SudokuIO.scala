package de.fmantz.sudoku

import java.io.{File, PrintWriter}

import scala.io.{BufferedSource, Source}

object SudokuIO {

	final val NewSudokuSeperator: String = "Grid"
	final val EmptyChar: Char = '0'
	final val QQWingEmptyChar: Char = '.' //https://qqwing.com

	/**
	 * Read usual 9x9 Suduko from text file
	 */
	def read(fileName: String): IndexedSeq[SudokuPuzzle] = {
		val source: BufferedSource = Source.fromFile(fileName)
		try {
			new SudokuIterator(source.getLines()).toIndexedSeq
		} finally {
			source.close()
		}
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
