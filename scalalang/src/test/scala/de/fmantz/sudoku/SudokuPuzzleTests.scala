package de.fmantz.sudoku

import de.fmantz.sudoku.SudokuIO.read
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SudokuPuzzleTests extends AnyFlatSpec with Matchers {

  "solve" should "solve 50 sudokos from project euler by simple backtracking algorithm" in {

    val path = this.getClass.getResource("/").getPath
    val startTotal = System.currentTimeMillis()

    read(fileName = s"$path/p096_sudoku.txt").zipWithIndex.foreach({ case (sudoku, index) =>
      val sudokuNumber = index + 1
      require(sudoku.isSolvable, s"Sudoku $sudokuNumber is not well-defined:\n $sudoku")
      sudoku.solve()
      require(sudoku.isSolved, s"Sudoku $sudokuNumber is not solved:\n $sudoku")
    })

    println(s"All sudokus solved by simple backtracking algorithm in ${System.currentTimeMillis() - startTotal} ms")
  }
}
