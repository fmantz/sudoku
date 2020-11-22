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

import scala.collection.mutable.ListBuffer

trait SudokuPuzzle {
  def set(row: Int, col: Int, value: Int): Unit
  def isEmpty(row: Int, col: Int): Boolean
  def isSolvable: Boolean
  def isSolved: Boolean
  def solve(): Unit
  def toPrettyString: String
}

class SudokuPuzzleImpl extends SudokuPuzzle {

  import SudokuConstants._

  //state:
  private val puzzle: Array[Array[Int]] = Array.ofDim[Int](PuzzleSize, PuzzleSize)
  private var isOpen: Boolean = true
  private var isEmpty: Boolean = true

  override def set(row: Int, col: Int, value: Int): Unit = {
    if(isOpen){
      puzzle(row)(col) = value
      isEmpty = false
    }
  }

  override def isEmpty(row: Int, col: Int): Boolean = {
    puzzle(row)(col) == 0
  }

  override def isSolved: Boolean = checkConditions(relaxed = false)

  override def isSolvable: Boolean = checkConditions(relaxed = true)

  /**
   * solves the sudoku by a simple backtracking algorithm (brute force)
   * inspired by https://www.youtube.com/watch?v=G_UYXzGuqvM
   */
  override def solve(): Unit = {
    def go(): Unit = {
      var row = 0
      var run = true
      while (run && row < PuzzleSize) {
        var col = 0
        while (run && col < PuzzleSize) {
          if (isEmpty(row, col)) {
            val solutionSpace = createSolutionSpace(row, col)
            for (n <- 1 to PuzzleSize) {
              if (solutionSpace.isSolution(n)) {
                set(row, col, n)
                go()
                set(row, col, value = 0) //backtrack!
              }
            }
            //solution found for slot!
            run = false
          }
          col += 1
        }
        row += 1
      }
      //solution found for all slots:
      if (run) {
        isOpen = false
      }
    }

    go()
    isOpen = true
  }

  override def toString: String = {
    val buffer = new ListBuffer[String]
    for (row <- 0 until PuzzleSize) {
      buffer.append(puzzle(row).mkString)
    }
    buffer.mkString("\n")
  }

  override def toPrettyString: String = {
    val dottedLine = "-" * (PuzzleSize * 3 + SquareSize - 1)
    val empty = "*"
    val buffer = new ListBuffer[String]
    for (row <- 0 until PuzzleSize) {
      val formattedRow = puzzle(row).zipWithIndex.map({ case (colValue, col) =>
        val rs = if (colValue == 0) s" $empty " else s" $colValue "
        if (col + 1 < PuzzleSize && col % SquareSize == 2) {
          rs + "|"
        } else {
          rs
        }
      }).mkString("")
      buffer.append(formattedRow)
      if (row < (PuzzleSize - 1) && (row + 1) % SquareSize == 0) {
        buffer.append(dottedLine)
      }
    }
    buffer.mkString("\n")
  }

  /**
   * @param row in [0,9]
   * @param relaxed true means it is still solvable, false it contains all possible numbers once
   */
  private def isRowOK(row: Int, relaxed: Boolean): Boolean = {
    val bits: SudokuBitSet = checkRow(row)
    bits.isFoundNumbersUnique && (relaxed || bits.isAllNumbersFound)
  }

  @inline private def checkRow(
    row: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    val selectedRow = puzzle(row)
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
   * @param relaxed true means it is still solvable, false it contains all possible numbers once
   */
  private def isColOK(col: Int, relaxed: Boolean): Boolean = {
    val bits: SudokuBitSet = checkCol(col)
    bits.isFoundNumbersUnique && (relaxed || bits.isAllNumbersFound)
  }

  @inline private def checkCol(
    col: Int,
    bits: SudokuBitSet = new SudokuBitSet()
  ): SudokuBitSet = {
    var row = 0
    while (row < PuzzleSize) {
      val value = puzzle(row)(col)
      bits.saveValue(value)
      row += 1
    }
    bits
  }

  /**
   * @param rowSquareIndex in [0,2]
   * @param colSquareIndex in [0,2]
   * @param relaxed true means it is still solvable, false it contains all possible numbers once
   */
  private def isSquareOK(rowSquareIndex: Int, colSquareIndex: Int, relaxed: Boolean): Boolean = {
    val bits = checkSquare(rowSquareIndex, colSquareIndex)
    bits.isFoundNumbersUnique && (relaxed || bits.isAllNumbersFound)
  }

  @inline private def checkSquare(
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
        val value = puzzle(row + rowSquareOffset)(col + colSquareOffset)
        bits.saveValue(value)
        col += 1
      }
      row += 1
    }
    bits
  }

  @inline private def checkConditions(relaxed: Boolean): Boolean = {
    (0 until PuzzleSize).forall(row => isRowOK(row, relaxed)) &&
      (0 until PuzzleSize).forall(col => isColOK(col, relaxed)) &&
        (0 until PuzzleSize).forall(i => isSquareOK(i / SquareSize, i % SquareSize, relaxed))
  }

  /**
   * The method returns a bit set containing all numbers already used
   */
  @inline private def createSolutionSpace(row: Int, col: Int): SudokuBitSet = {
    val bits = new SudokuBitSet()
    checkRow(row, bits)
    checkCol(col, bits)
    checkSquare(
      rowSquareIndex = row / SquareSize,
      colSquareIndex = col / SquareSize,
      bits = bits
    )
    bits
  }

}
