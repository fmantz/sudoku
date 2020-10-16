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
	def read(fileName: String): (BufferedSource, Iterator[SudokuPuzzle]) = {
		val source: BufferedSource = Source.fromFile(fileName)
		val iter = new SudokuIterator(source.getLines())
		(source, iter)
	}

  /**
   * Read Suduko to text file
   */
  def write(fileName: String, puzzles: Iterator[SudokuPuzzle]): Unit = {
		val writer = new PrintWriter(new File(fileName))
		try {
			val pattern = s"$NewSudokuSeperator %d"
			puzzles.zipWithIndex.foreach({ case (sudoku, index) =>
        writer.println(pattern.format(index + 1))
        writer.println(sudoku)
				writer.flush()
			})
		} finally {
			writer.close()
		}
	}

	def writeQQWing(fileName: String, puzzles: Iterator[SudokuPuzzle]): Unit = {
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
