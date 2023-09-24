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

  "solve" should "solve one sudoku by simple backtracking algorithm" in {
    checkSolve(fileName = "one_sudoku.txt")
  }

  it should "solve 50 sudokus from project euler by simple backtracking algorithm" in {
    checkSolve(fileName = "p096_sudoku.txt")
  }

  it should "solve 10 sudokus generated with www.qqwing.com by simple backtracking algorithm" in {
    checkSolve(fileName = "sudoku.txt")
  }

  private def checkSolve(fileName: String): Unit = {
    val path = this.getClass.getResource("/").getPath
    val startTotal = System.currentTimeMillis()
    val (source, puzzles) = read(path = s"$path/$fileName")
    try {
      puzzles
        .zipWithIndex.foreach({ case (sudoku, index) =>
        val sudokuNumber = index + 1
        val input = sudoku.toString
        sudoku.solve()
        val output = sudoku.toString
        require(checkSolution(sudoku), s"Sudoku $sudokuNumber is not solved:\n${sudoku.toPrettyString}")
        require(input.length == output.length, "Sudoku strings have not same length")
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
    '0' <= c || c > '9'
  }

  /**
   * @param row in [0,9]
   */
  private def isRowOK(sudoku: Array[Array[Byte]], row: Int): Boolean = {
    val bits: SudokuBitSet = checkRow(sudoku, row)
    bits.isFoundNumbersUnique && bits.isAllNumbersFound
  }

  @inline private def checkRow(
    sudoku: Array[Array[Byte]],
    row: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    val selectedRow = sudoku(row)
    for (col <- 0 until PuzzleSize) {
      val value = selectedRow(col)
      bits.saveValue(value)
    }
    bits
  }

  /**
   * @param col in [0,9]
   */
  private def isColOK(sudoku:  Array[Array[Byte]], col: Int): Boolean = {
    val bits: SudokuBitSet = checkCol(sudoku, col)
    bits.isFoundNumbersUnique && bits.isAllNumbersFound
  }

  @inline private def checkCol(
    sudoku: Array[Array[Byte]],
    col: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    for (row <- 0 until PuzzleSize) {
      val value = sudoku(row)(col)
      bits.saveValue(value)
    }
    bits
  }

  /**
   * @param rowSquareIndex in [0,2]
   * @param colSquareIndex in [0,2]
   */
  private def isSquareOK(sudoku: Array[Array[Byte]], rowSquareIndex: Int, colSquareIndex: Int): Boolean = {
    val bits = checkSquare(sudoku, rowSquareIndex, colSquareIndex)
    bits.isFoundNumbersUnique && bits.isAllNumbersFound
  }

  @inline private def checkSquare(
    sudoku: Array[Array[Byte]],
    rowSquareIndex: Int,
    colSquareIndex: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    val rowSquareOffset = rowSquareIndex * SquareSize
    val colSquareOffset = colSquareIndex * SquareSize
    for (row <- 0 until SquareSize) {
      for (col <- 0 until SquareSize) {
        val value = sudoku(row + rowSquareOffset)(col + colSquareOffset)
        bits.saveValue(value)
      }
    }
    bits
  }

  private def checkSolution(sudokuPuzzle: SudokuPuzzle): Boolean = {
    val sudoku = makeSudoku2DArray(sudokuPuzzle)
    (0 until PuzzleSize).forall(row => isRowOK(sudoku, row)) &&
      (0 until PuzzleSize).forall(col => isColOK(sudoku, col)) &&
        (0 until PuzzleSize).forall(i => isSquareOK(sudoku, i / SquareSize, i % SquareSize))
  }

  private def makeSudoku2DArray(sudokuPuzzle: SudokuPuzzle): Array[Array[Byte]] = {
    val sudoku = Array.ofDim[Byte](PuzzleSize, PuzzleSize)
    for (row <- 0 until PuzzleSize) {
      for (col <- 0 until PuzzleSize) {
        sudoku(row)(col) = sudokuPuzzle.get(row, col)
      }
    }
    sudoku
  }

}
