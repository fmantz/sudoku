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
	private val CellCount = 81
	private val myPuzzle = Array.ofDim[Byte](CellCount)
	private val myIndices = Array.ofDim[Int](CellCount)

	private val possibleNumbersInitial = new SudokuNumbers()
	private val possibleNumbersCurrent = new SudokuNumbers()

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
				saveValueForCell(curValue, i, possibleNumbersInitial)
			}
		}

		//Test:
		for(i <- 0 until CellCount) {
			println(s"$i: " + getPossibleNumbers(i, possibleNumbersInitial).toVector)
		}

		//2. store count possible numbers in myIndices (get possible numbers by PossibleCounts(i))
		//   zip possible numbers by index, and sort tuple array by counts (asc)
		//   then forget counts
		//   sort can be implemented very fast by only 2 scans:
		//   a. count possible numbers in an int array (since all counts must be between 0-9)
		//   b. have another int array for the current counter index
		//   c. go once again thorough all numbers and put each index to postion numberOffset + countNumberInCounterpostion
		val numberOffsets = Array.ofDim[Int](PuzzleSize + 1)
		for(i <- 0 until CellCount){
			val countOfIndex = getPossibleCounts(i, possibleNumbersInitial)
			numberOffsets(countOfIndex + 1)+=1
		}
		for(i <- 1 until PuzzleSize){ //correct offsets
			numberOffsets(i)+=numberOffsets(i - 1)
		}
		println(numberOffsets.toVector)
		for(i <- 0 until CellCount){
			val countOfIndex = getPossibleCounts(i, possibleNumbersInitial)
			val offset = numberOffsets(countOfIndex)
			require(myIndices(offset) == 0, s"tried to overwrite $i / $offset ")
			myIndices(offset) = i
			numberOffsets(countOfIndex)+=1
		}
		println(myIndices.toVector)

		//Init runtime arrays:
		System.arraycopy(possibleNumbersInitial.rowNums,0, possibleNumbersCurrent.rowNums, 0, SudokuConstants.PuzzleSize)
		System.arraycopy(possibleNumbersInitial.colNums,0, possibleNumbersCurrent.colNums, 0, SudokuConstants.PuzzleSize)
		System.arraycopy(possibleNumbersInitial.squareNums,0, possibleNumbersCurrent.squareNums, 0, SudokuConstants.PuzzleSize)

		//3. solve the puzzle by backtracking (without recursion!)
		var i = 0
		while(i < CellCount){
			val puzzleIndex = myIndices(i)
			val curValue = myPuzzle(puzzleIndex)
			if(curValue == 0){ //Is not given?
				//Is there a current guess possible?
				val nextNumbers = getPossibleNumbers(puzzleIndex, possibleNumbersCurrent)
				if(nextNumbers.length > 0){
					val nextNumber = nextNumbers.head
					myPuzzle(puzzleIndex) = nextNumber
					saveValueForCell(nextNumber, puzzleIndex, possibleNumbersCurrent)
					i+=1
				} else {
					//backtrack: //TODO anders:
					i-=1
					val lastPuzzleIndex = myIndices(i)
					val lastValue = myPuzzle(lastPuzzleIndex)
					revertValueForCell(lastValue, lastPuzzleIndex, possibleNumbersCurrent)
					val lastNumbers = getPossibleNumbers(lastPuzzleIndex, possibleNumbersCurrent)
					val lastNextNumberIndex = lastNumbers.indexOf(lastValue) + 1 //next

					if(lastNextNumberIndex < lastNumbers.length){
						//there is another try possible:
						val lastNextNumber = lastNumbers(lastNextNumberIndex)
						myPuzzle(lastPuzzleIndex) = lastNextNumber
						saveValueForCell(lastNextNumber, lastPuzzleIndex, possibleNumbersCurrent)
						i+=1 //we were lucky!
					} else {

					}
				}
			} else {
				i+=1 //value was given!
			}
		}
	}

	private def saveValueForCell(value: Int, index: Int, nums: SudokuNumbers) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		nums.rowNums(rowIndex) |= checkBit
		nums.colNums(colIndex) |= checkBit
		nums.squareNums(squareIndex) |= checkBit
	}

	private def revertValueForCell(value: Int, index: Int, nums: SudokuNumbers) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		nums.rowNums(rowIndex) ^= checkBit
		nums.colNums(colIndex) ^= checkBit
		nums.squareNums(squareIndex) ^= checkBit
		//		println(s"saved($value, $index) in ($rowIndex, $colIndex, $squareIndex)")
	}

	private def revertCell(index: Int, curNums: SudokuNumbers, initialNums: SudokuNumbers) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		curNums.rowNums(rowIndex) = initialNums.rowNums(rowIndex)
		curNums.colNums(colIndex) = initialNums.colNums(colIndex)
		curNums.squareNums(squareIndex) = initialNums.squareNums(squareIndex)
		//		println(s"saved($value, $index) in ($rowIndex, $colIndex, $squareIndex)")
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

	private def getPossibleNumbers(index: Int, nums: SudokuNumbers) : Array[Byte]= {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val possibleNumberIndex = nums.rowNums(rowIndex) | nums.colNums(colIndex) | nums.squareNums(squareIndex)
		SudokuConstants.BitsetPossibleNumbers(possibleNumberIndex)
	}

	private def getPossibleCounts(index: Int, nums: SudokuNumbers) : Int = {
		if(myPuzzle(index) == 0){
			getPossibleNumbers(index, nums).length //calculate possible numbers!
		} else {
			0 //number preset (no more possible!)
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
