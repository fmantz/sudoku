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
	private var myIsSolved: Boolean = false

	private val puzzle = Array.ofDim[Byte](CellCount)
	private val puzzleSorted = Array.ofDim[Byte](CellCount)
	private val indices = Array.ofDim[Int](CellCount)

	private val rowNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val colNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val squareNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)

	override def set(row: Int, col: Int, value: Byte): Unit = {
		puzzle(getSingleArrayIndex(row, col)) = value
	}

	def get(row: Int, col: Int): Byte = {
		puzzle(getSingleArrayIndex(row, col))
	}

	private def getSingleArrayIndex(row: Int, col: Int) = {
		row * PuzzleSize + col
	}

	override def isEmpty(row: Int, col: Int): Boolean = {
		puzzle(getSingleArrayIndex(row, col))  == 0
	}

	override def initTurbo(): Unit = {
	}

	override def isSolved: Boolean = {
		myIsSolved
	}

	override def isSolvable: Boolean = {
		true //TODO
	}

	/**
	 * solves the sudoku by a simple backtracking algorithm (brute force)
	 * inspired by https://www.youtube.com/watch?v=G_UYXzGuqvM
	 */
	override def solve(): Unit = {

		//1. step go once through the puzzle and store which numbers are still possible in each cell
		//   note: in cells that are preset by the puzzle no numbers are valid (fill myPossibleNumbers)
		for(i <- puzzle.indices){
			val curValue = puzzle(i)
			if(curValue > 0) {
				saveValueForCell(curValue, i)
			}
		}

		//2. store count possible numbers in myIndices (get possible numbers by PossibleCounts(i))
		//   zip possible numbers by index, and sort tuple array by counts (asc)
		//   then forget counts
		//   sort can be implemented very fast by only 2 scans:
		//   a. count possible numbers in an int array (since all counts must be between 0-9)
		//   b. have another int array for the current counter index
		//   c. go once again thorough all numbers and put each index to position numberOffset + countNumberInCounterpostion
		val numberOffsets = Array.ofDim[Int](PuzzleSize + 2) //counts 0 - 9 + 1 offset = puzzleSize + 2 (9 + 2)
		for(i <- 0 until CellCount){
			val countOfIndex = getPossibleCounts(i)
			numberOffsets(countOfIndex + 1)+=1
		}
		for(i <- 1 until numberOffsets.length){ //correct offsets
			numberOffsets(i)+=numberOffsets(i - 1)
		}
		for(i <- 0 until CellCount){
			val countOfIndex = getPossibleCounts(i)
			val offset = numberOffsets(countOfIndex)
			indices(offset) = i
			numberOffsets(countOfIndex)+=1
		}
		sortPuzzle() //avoid jumping in the puzzle array

		//3. solve the puzzle by backtracking (without recursion!)
		var lastInvaldTry: Byte = 0
		var i = 0
		while(i < CellCount){
			val curValue = puzzleSorted(i)
			if(curValue == 0){ //Is not given?

				//Is there a current guess possible?
				val puzzleIndex = indices(i)
				val rowIndex = calculateRowIndex(puzzleIndex)
				val colIndex = calculateColIndex(puzzleIndex)
				val squareIndex = calculateSquareIndex(rowIndex, colIndex)
				val possibleNumberIndex = rowNums(rowIndex) | colNums(colIndex) | squareNums(squareIndex)
				//TODO ring-cache! 3 elements
				//nextNumbers, lastindexOf
				val nextNumbers = SudokuConstants.BitsetPossibleNumbers(possibleNumberIndex)

				val nextNumberIndex = if(lastInvaldTry == 0) {
					0
				} else {
					fastIndexOf(nextNumbers, lastInvaldTry) + 1
				}

				if(nextNumberIndex < nextNumbers.length){
					//next possible number to try found:
					val nextNumber = nextNumbers(nextNumberIndex)
					puzzleSorted(i) = nextNumber
					saveValueForCell(nextNumber, rowIndex, colIndex, squareIndex)
					lastInvaldTry = 0 //0 since success
					i+=1
				} else {
					//backtrack:
					i-=1 //not given values are in the head of myIndices, there we can simply go one step back!
					lastInvaldTry = puzzleSorted(i)
					puzzleSorted(i) = 0
					val lastPuzzleIndex = indices(i)
					revertValueForCell(lastInvaldTry, lastPuzzleIndex)
				}

			} else {
				i+=1 //value was given!
			}
		}
		fillPositions() //put values back to original puzzle positions
		myIsSolved = true
	}

	private def sortPuzzle() : Unit = {
		for(i <- puzzle.indices){
			puzzleSorted(i) = puzzle(indices(i))
		}
	}

	private def fillPositions(): Unit = {
		for(i <- puzzle.indices) {
			puzzle(indices(i)) = puzzleSorted(i)
		}
	}

	private def fastIndexOf(array: Array[Byte], b: Byte): Int = {
		val lastIndex = array.length - 1
		if(array(lastIndex) == b){
			lastIndex
		} else {
			var run = true
			var index = 0
			while (run) {
				if (array(index) != b) {
					index += 1
				} else {
					run = false
				}
			}
			index
		}
	}

	private def saveValueForCell(value: Int, index: Int) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		rowNums(rowIndex) |= checkBit
		colNums(colIndex) |= checkBit
		squareNums(squareIndex) |= checkBit
	}

	private def saveValueForCell(value: Int, rowIndex: Int, colIndex:Int, squareIndex: Int) : Unit = {
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		rowNums(rowIndex) |= checkBit
		colNums(colIndex) |= checkBit
		squareNums(squareIndex) |= checkBit
	}

	private def revertValueForCell(value: Int, index: Int) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		rowNums(rowIndex) ^= checkBit
		colNums(colIndex) ^= checkBit
		squareNums(squareIndex) ^= checkBit
	}

	private def calculateRowIndex(index: Int) = {
		index / PuzzleSize
	}

	private def calculateColIndex(index: Int) = {
		index % PuzzleSize
	}

	private def calculateSquareIndex(row: Int, col: Int): Int = {
		col / SudokuConstants.SquareSize + row / SudokuConstants.SquareSize * SudokuConstants.SquareSize
	}

	private def getPossibleNumbers(index: Int) : Array[Byte]= {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val possibleNumberIndex = rowNums(rowIndex) | colNums(colIndex) | squareNums(squareIndex)
		SudokuConstants.BitsetPossibleNumbers(possibleNumberIndex)
	}

	private def getPossibleCounts(index: Int) : Int = {
		if(puzzle(index) == 0){
			getPossibleNumbers(index).length //calculate possible numbers!
		} else {
			0 //number preset (no more possible!)
		}
	}

	override def toString: String = {
		val buffer = new ListBuffer[String]
		for (row <- 0 until PuzzleSize) {
			val from = row * PuzzleSize
			val until = from + PuzzleSize
			buffer.append(puzzle.slice(from, until).mkString)
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
			val currentRow = puzzle.slice(from, until)
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
