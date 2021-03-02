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
	val myPossibleNumbers = Array.ofDim[Int](CellCount) //TODO: change Bitsets (three Arrays wie in Turbo)
	val myLastPossibleNumberIndex = Array.ofDim[Int](CellCount) //next index to use in each array of possible numbers
	//TODO try possibleNo(myLastPossibleNumberIndex) , revert possibleNo(myLastPossibleNumberIndex - 1) iff myLastPossibleNumberIndex > 0

	override def set(row: Int, col: Int, value: Byte): Unit = {
		if (isOpen) {
			myPuzzle(getSingleArrayIndex(row, col)) = value
			puzzle(row)(col) = value
			isEmpty = false
		}
	}

	private def getSingleArrayIndex(row: Int, col: Int) = {
		row * PuzzleSize + col
	}

	override def isEmpty(row: Int, col: Int): Boolean = {
		myPuzzle(getSingleArrayIndex(row, col))  == 0
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
		//1. step go once through the puzzle and store which numbers are still possible in each cell
		//   note: in cells that are preset by the puzzle no numbers are valid (fill myPossibleNumbers)
		for(i <- myPuzzle.indices){
			val curValue = myPuzzle(i)
			if(curValue > 0) {
				saveValueForCell(curValue, myPossibleNumbers, i)
			}
		}

		//Test:
		for(i <- 0 until CellCount) {
			println(s"$i: " + SudokuConstants.BitsetPossibleNumbers(myPossibleNumbers(i)).toVector)
		}

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
		//     onePossibleSolution = OnePossibleNumbers(curMyPossibleNumbers)
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
	}

	private def saveValueForCell(value: Int, mem: Array[Int], index: Int) : Unit = {
		saveValueRow(value, mem, index)
		saveValueCol(value, mem, index)
		saveValueSquare(value, mem, index)
	}

	/**
	 * Save a value
	 */
	private def saveValue(value: Int, mem: Array[Int], index: Int): Unit = {
		//TODO nach aussen!
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		mem(index) |= checkBit
	}

	private def saveValueRow(value: Int, mem: Array[Int], index: Int): Unit = {
			val rowNumber = index / PuzzleSize
			val fromIndex = rowNumber * PuzzleSize
			val untilIndex = fromIndex + PuzzleSize
			for(i <- fromIndex until untilIndex){
				  //println(s"save $value in $i (row)")
					saveValue(value, mem, i)
			}
	}

	private def saveValueCol(value: Int, mem: Array[Int], index: Int): Unit = {
		val colNumber = index % PuzzleSize
		for(i <- colNumber until CellCount by PuzzleSize){
			//println(s"save $value in $i (col)")
			saveValue(value, mem, i)
		}
	}

	private def saveValueSquare(value: Int, mem: Array[Int], index: Int): Unit = {
		val squareColumn = index / SquareSize
		for(j <- 0 until SquareSize) {
			val rowNumber = index / PuzzleSize
			val rowOffset = (rowNumber / SquareSize) * (SquareSize * PuzzleSize)
			val colNumber = index % PuzzleSize
			val colOffset = (colNumber / SquareSize) * SquareSize
			val startIndex = rowOffset + colOffset + (j * PuzzleSize)
			for (i <- startIndex until startIndex + SquareSize) {
//				println(s"save $value in $i (square)")
				saveValue(value, mem, i)
			}
		}
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
