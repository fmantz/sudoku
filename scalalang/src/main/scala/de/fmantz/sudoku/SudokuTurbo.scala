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

class SudokuTurbo private() {

	import SudokuTurbo._

	private val colCounts: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val rowCounts: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)

	private val colNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val rowNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val squareNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)

	var rowIndices: Array[Int] = Array.emptyIntArray
	var colIndices: Array[Int] = Array.emptyIntArray

	private var myIsSolvable: Boolean = true

	private def saveValueAndCheckIsSolvable(row: Int, col: Int, value: Int): Unit = {
		if (value != 0) {

			//save col data:
			colCounts(col) += 1
			val oldColNumValue = colNums(col)
			val newColNumValue = storeValueAsBit(oldColNumValue, value)
			colNums(col) = newColNumValue

			//save row data:
			rowCounts(row) += 1
			val oldRowNumValue = rowNums(row)
			val newRowNumValue = storeValueAsBit(oldRowNumValue, value)
			rowNums(row) = newRowNumValue

			//save square data:
			val squareIndex = calculateSquareIndex(row, col)
			val oldSquareNumValue = squareNums(squareIndex)
			val newSquareNumValue = storeValueAsBit(oldSquareNumValue, value)
			squareNums(squareIndex) = newSquareNumValue

			//If old and new value is equal the same value
			//has already been stored before:
			myIsSolvable &&=
				oldColNumValue != newColNumValue &&
					oldRowNumValue != newRowNumValue &&
					oldSquareNumValue != newSquareNumValue
		}
	}

	def saveValue(row: Int, col: Int, value: Int): Unit = {
		if (value != 0) {
			//save col data:
			colCounts(col) += 1
			colNums(col) = storeValueAsBit(colNums(col), value)
			//save row data:
			rowCounts(row) += 1
			rowNums(row) = storeValueAsBit(rowNums(row), value)
			//save square data:
			val squareIndex = calculateSquareIndex(row, col)
			squareNums(squareIndex) = storeValueAsBit(squareNums(squareIndex), value)
		}
	}

	def revertValue(row: Int, col: Int, value: Int): Unit = {
		if (value != 0) {
			//save col data:
			colCounts(col) -= 1
			colNums(col) = revertValueAsBit(colNums(col), value)
			//save row data:
			rowCounts(row) -= 1
			rowNums(row) = revertValueAsBit(rowNums(row), value)
			//save square data:
			val squareIndex = calculateSquareIndex(row, col)
			squareNums(squareIndex) = revertValueAsBit(squareNums(squareIndex), value)
		}
	}

	def createSolutionSpace(row: Int, col: Int): SudokuBitSet = {
		val squareIndex = SudokuTurbo.calculateSquareIndex(row, col)
		val bits: Int = colNums(col) | rowNums(row) | squareNums(squareIndex)
		new SudokuBitSet(bits)
	}

	def isSolvable: Boolean = myIsSolvable

	def isSolved: Boolean = {
		this.colNums.forall(_ == SudokuConstants.PuzzleSize) //Does not ensure the solution is correct, but the algorithm will!
	}

}

object SudokuTurbo {

	def create(puzzleData: Array[Array[Int]]): SudokuTurbo = {
		val rs = new SudokuTurbo()
		var row, col = 0
		while (row < SudokuConstants.PuzzleSize) {
			val rowData = puzzleData(row)
			while (col < SudokuConstants.PuzzleSize) {
				rs.saveValueAndCheckIsSolvable(row, col, rowData(col))
				col += 1
			}
			col = 0
			row += 1
		}
		rs.rowIndices = createSortedIndices(rs.rowCounts)
		rs.colIndices = createSortedIndices(rs.colCounts)
		rs
	}

	private def calculateSquareIndex(row: Int, col: Int): Int = {
		col / SudokuConstants.SquareSize + row / SudokuConstants.SquareSize * SudokuConstants.SquareSize
	}

	/**
	 * Save a value
	 */
	private def storeValueAsBit(container: Int, value: Int): Int = {
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		container | checkBit
	}

	/**
	 * Revert a value
	 */
	private def revertValueAsBit(container: Int, value: Int): Int = {
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		container ^ checkBit
	}

	private def createSortedIndices(num: Array[Int]): Array[Int] = {
		num.zipWithIndex.sortBy(_._1).reverse.map(_._2) ++ Array(-1) //sort according to numbers heuristic, large numbers first
	}

}
