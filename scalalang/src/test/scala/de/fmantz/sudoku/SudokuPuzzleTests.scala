//scalastyle:off
/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.txt file in the top level folder.
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

import de.fmantz.sudoku.SudokuIO.read
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SudokuPuzzleTests extends AnyFlatSpec with Matchers {

  "solve" should "solve 50 sudokus from project euler by simple backtracking algorithm" in {
    checkSolve(fileName = "p096_sudoku.txt")
  }

  it should "solve 10 sudokus generated with www.qqwing.com by simple backtracking algorithm" in {
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
        require(sudoku.isSolvable, s"Sudoku $sudokuNumber is not well-defined:\n ${sudoku.toPrettyString}")
        sudoku.solve()
        val output = sudoku.toString
        require(sudoku.isSolved, s"Sudoku $sudokuNumber is not solved:\n ${sudoku.toPrettyString}")
        require(input.length == output.length, "sudoku strings have not same length")
        for (i <- input.indices) {
          val inChar = input.charAt(i)
          val outChar = output.charAt(i)
          if (!isBlank(inChar)) {
            inChar shouldBe outChar //puzzle should not be changed!
          }
        }
      })
      println(s"All sudokus solved by simple backtracking algorithm in ${System.currentTimeMillis() - startTotal} ms")
    } finally {
      source.close()
    }
  }

  def isBlank(c: Char): Boolean = {
    c == SudokuConstants.EmptyChar || c == SudokuConstants.QQWingEmptyChar
  }

}
