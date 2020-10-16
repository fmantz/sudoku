package de.fmantz.sudoku

import de.fmantz.sudoku.SudokuIO.read
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SudokuPuzzleTests extends AnyFlatSpec with Matchers {

  "solve" should "solve 50 sudokos from project euler by simple backtracking algorithm" in {
    checkSolve(fileName = "p096_sudoku.txt")
  }

  it should "solve 100 sudokos generated with www.qqwing.com by simple backtracking algorithm" in {
    checkSolve(fileName = "sudoku.txt")
  }

  private def checkSolve(fileName: String): Unit = {
    val path = this.getClass.getResource("/").getPath
    val startTotal = System.currentTimeMillis()
    val (source, puzzles) = read(fileName = s"$path/$fileName")
    try {
      puzzles.zipWithIndex.foreach({ case (sudoku, index) =>
        val sudokuNumber = index + 1
        val input = sudoku.toString
        require(sudoku.isSolvable, s"Sudoku $sudokuNumber is not well-defined:\n $sudoku")
        sudoku.solve()
        val output = sudoku.toString
        require(sudoku.isSolved, s"Sudoku $sudokuNumber is not solved:\n ${sudoku.toPrettyString}")
        require(input.length == output.length, "sudoku strings have not same length")
        for (i <- input.indices) {
          val in = input.charAt(i)
          val out = output.charAt(i)
          if (!isBlank(in) && in != out) {
            input shouldBe output
          }
        }
      })
      println(s"All sudokus solved by simple backtracking algorithm in ${System.currentTimeMillis() - startTotal} ms")
    } finally {
      source.close()
    }
  }

  def isBlank(c: Char): Boolean = {
    c == SudokuIO.EmptyChar || c == SudokuIO.QQWingEmptyChar
  }

}
