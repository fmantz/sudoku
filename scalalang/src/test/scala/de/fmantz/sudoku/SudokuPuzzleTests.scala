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

import de.fmantz.sudoku.SudokuConstants.{PuzzleSize, SquareSize}
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
        sudoku.initTurbo()
        require(sudoku.isSolvable, s"Sudoku $sudokuNumber is not well-defined:\n ${sudoku.toPrettyString}")
        sudoku.solve()
        val output = sudoku.toString
        require(checkSolution(sudoku.asInstanceOf[SudokuPuzzleImpl]), s"Sudoku $sudokuNumber is not solved:\n ${sudoku.toPrettyString}")
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

  /**
   * @param row in [0,9]
   */
  private def isRowOK(sudoku: SudokuPuzzleImpl, row: Int): Boolean = {
    val bits: SudokuBitSet = checkRow(sudoku, row)
    bits.isFoundNumbersUnique && bits.isAllNumbersFound
  }

  @inline private def checkRow(
    sudoku: SudokuPuzzleImpl,
    row: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    val selectedRow = sudoku.puzzle(row)
    var col = 0
    while (col < PuzzleSize) {
      val value = selectedRow(col)
      bits.saveValue(value)
      col += 1
    }
    bits
  }

  /**
   * @param col in [0,9]
   */
  private def isColOK(sudoku: SudokuPuzzleImpl, col: Int): Boolean = {
    val bits: SudokuBitSet = checkCol(sudoku, col)
    bits.isFoundNumbersUnique && bits.isAllNumbersFound
  }

  @inline private def checkCol(
    sudoku: SudokuPuzzleImpl,
    col: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    var row = 0
    while (row < PuzzleSize) {
      val value = sudoku.puzzle(row)(col)
      bits.saveValue(value)
      row += 1
    }
    bits
  }

  /**
   * @param rowSquareIndex in [0,2]
   * @param colSquareIndex in [0,2]
   */
  private def isSquareOK(sudoku: SudokuPuzzleImpl, rowSquareIndex: Int, colSquareIndex: Int): Boolean = {
    val bits = checkSquare(sudoku, rowSquareIndex, colSquareIndex)
    bits.isFoundNumbersUnique && bits.isAllNumbersFound
  }

  @inline private def checkSquare(
    sudoku: SudokuPuzzleImpl,
    rowSquareIndex: Int,
    colSquareIndex: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    val rowSquareOffset = rowSquareIndex * SquareSize
    val colSquareOffset = colSquareIndex * SquareSize
    var row = 0
    while (row < SquareSize) {
      var col = 0
      while (col < SquareSize) {
        val value = sudoku.puzzle(row + rowSquareOffset)(col + colSquareOffset)
        bits.saveValue(value)
        col += 1
      }
      row += 1
    }
    bits
  }

  private def checkSolution(sudoku: SudokuPuzzleImpl): Boolean = {
    (0 until PuzzleSize).forall(row => isRowOK(sudoku, row)) &&
        (0 until PuzzleSize).forall(col => isColOK(sudoku, col)) &&
          (0 until PuzzleSize).forall(i => isSquareOK(sudoku, i / SquareSize, i % SquareSize))
  }

}
