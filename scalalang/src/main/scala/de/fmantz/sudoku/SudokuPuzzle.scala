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
	def set(row: Int, col: Int, value: Byte): Unit

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
	val puzzle: Array[Array[Byte]] = Array.ofDim[Byte](PuzzleSize, PuzzleSize)
	private var isOpen: Boolean = true
	private var isEmpty: Boolean = true
	private var turbo: SudokuTurbo = SudokuTurbo.create()
	private var myIsSolved: Boolean = false

	//TODO:
	val CellCount = 81
	val myPuzzle = Array.ofDim[Byte](CellCount)
	val myIndices = Array.ofDim[Int](CellCount)
	val myPossibleNumbers = Array.ofDim[Int](CellCount) //Bitsets
	val myNumbersTried = Array.ofDim[Int](CellCount) //Bitsets

	//1. step go once through the puzzle and store which numbers are still possible in each cell
	//   note: in cells that are preset by the puzzle no numbers are valid (fill myPossibleNumbers)

	//2. store count possible numbers in myIndices (get possible numbers by PossibleCounts(i))
	//   zip possible numbers by index, and sort tuple array by counts (asc)
	//   then forget counts
	//   sort can be implemented very fast by only 2 scans:
	//   a. count possible numbers in an int array (since all counts must be between 0-9)
	//   b. have another int array for the current counter index
	//   c. go once again thorough all numbers and put each index to postion numberOffset + countNumberInCounterpostion

	//3. solve the puzzle by backtracking
	//   i = 0
	//   while(i < CellCount){
	//     puzzleIndex = myIndices(i)
	//     curMyPossibleNumbers = myPossibleNumbers(puzzelIndex) | myNumbersTried(puzzleIndex)
	//     onePossibleSolution = OnePossibleNumbers(curMyPossibleNumbers)  //array that contain one random equaly distributed number
	//     myNumbersTried(puzzelIndex) |= onePossibleSolution //tried numbers update
	//     if(onePossibleSolution == 0 &&  myPossibleNumbers(puzzelIndex) != 0){
	//     		//go backwords //also update myTriedNumbers(?)
	//				myTriedNumbers(puzzleIndex) = 0
	//        i-=1
	//     } else{
	//			myPuzzle(puzzelIndex) = onePossibleSolution
	//     	i+=1
	//     }
	//   }

	override def set(row: Int, col: Int, value: Byte): Unit = {
		if (isOpen) {
			myPuzzle(row * PuzzleSize + col) = value
			puzzle(row)(col) = value
			isEmpty = false
		}
	}

	override def isEmpty(row: Int, col: Int): Boolean = {
		myPuzzle(row * PuzzleSize + col)  == 0
	}

	override def initTurbo(): Unit = {
		this.turbo.init(this.puzzle)
		myIsSolved = this.turbo.isSolved
	}

	override def isSolved: Boolean = {
		myIsSolved
	}

	override def isSolvable: Boolean = {
		turbo.isSolvable
	}

	/**
	 * solves the sudoku by a simple backtracking algorithm (brute force)
	 * inspired by https://www.youtube.com/watch?v=G_UYXzGuqvM
	 */
	override def solve(): Unit = {
		def go(): Unit = {
			var row = 0
			var run = true
			while (run && row < PuzzleSize) {
				val rowIndex = turbo.rowIndices(row)
				var col = 0
				while (run && col < PuzzleSize) {
					val colIndex = turbo.colIndices(col)
					if (isEmpty(rowIndex, colIndex)) {
						val solutionSpace = turbo.createSolutionSpace(rowIndex, colIndex)
						val possibleNumbers = solutionSpace.possibleNumbers
						for (n <- possibleNumbers) {
							set(rowIndex, colIndex, n)
							turbo.saveValue(rowIndex, colIndex, n)
							go()
							set(rowIndex, colIndex, value = 0) //backtrack!
							turbo.revertValue(rowIndex, colIndex, n)
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
				myIsSolved = true
			}
		}

		go()
		isOpen = true
	}

	override def toString: String = {
		val buffer = new ListBuffer[String]
		for (row <- 0 until PuzzleSize) {
			val from = row * PuzzleSize
			val until = from + PuzzleSize
			buffer.append(myPuzzle.slice(from, until).mkString)
		}
		buffer.mkString("\n")
	}

	override def toPrettyString: String = {
		val dottedLine = "-" * (PuzzleSize * 3 + SquareSize - 1)
		val empty = "*"
		val buffer = new ListBuffer[String]
		for (row <- 0 until PuzzleSize) {
			val from = row * PuzzleSize
			val until = from + PuzzleSize
			val currentRow = myPuzzle.slice(from, until)
			val formattedRow = currentRow.zipWithIndex.map({ case (colValue, col) =>
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
