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

import scala.collection.mutable.ListBuffer

trait SudokuPuzzle {
  def set(row: Int, col: Int, value: Int): Unit
  def isEmpty(row: Int, col: Int): Boolean
  def isSolvable: Boolean
  def initTurbo(): Unit
  def isSolved: Boolean
  def solve(): Unit
  def toPrettyString: String
}

class SudokuPuzzleImpl extends SudokuPuzzle {

  import SudokuConstants._

  //state:
  val puzzle: Array[Array[Int]] = Array.ofDim[Int](PuzzleSize, PuzzleSize)
  private var isOpen: Boolean = true
  private var isEmpty: Boolean = true
  private var turbo: Option[SudokuTurbo] = None //TODO Remove Option

  override def set(row: Int, col: Int, value: Int): Unit = {
    if(isOpen){
      puzzle(row)(col) = value
      isEmpty = false
    }
  }

  override def isEmpty(row: Int, col: Int): Boolean = {
    puzzle(row)(col) == 0
  }

  override def initTurbo() : Unit = {
    this.turbo = Some(SudokuTurbo.create(this.puzzle)) //TODO should call init but turbo should already exist!
  }

  override def isSolved: Boolean = turbo.exists(_.isSolved)

  override def isSolvable: Boolean = turbo.map(_.isSolvable).getOrElse(sys.error("turbo must be initialized"))

  /**
   * solves the sudoku by a simple backtracking algorithm (brute force)
   * inspired by https://www.youtube.com/watch?v=G_UYXzGuqvM
   */
  override def solve(): Unit = {
    val myTurbo = turbo.getOrElse(sys.error("turbo must be initialized"))
    def go(): Unit = {
      var row = 0
      var rowIndex = myTurbo.rowIndices(row)
      var run = true
      while (run && row < PuzzleSize) {
        var col = 0
        var colIndex = myTurbo.colIndices(col)
        while (run && col < PuzzleSize) {
          if (isEmpty(rowIndex, colIndex)) {
            val solutionSpace = myTurbo.createSolutionSpace(rowIndex, colIndex)
            for (n <- 1 to PuzzleSize) {
              if (solutionSpace.isSolution(n)) {
                set(rowIndex, colIndex, n)
                myTurbo.saveValue(rowIndex, colIndex, n)
                go()
                set(rowIndex, colIndex, value = 0) //backtrack!
                myTurbo.revertValue(rowIndex, colIndex, n)
              }
            }
            //solution found for slot!
            run = false
          }
          col += 1
          colIndex=myTurbo.colIndices(col)
        }
        row += 1
        rowIndex=myTurbo.rowIndices(row)
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

}
