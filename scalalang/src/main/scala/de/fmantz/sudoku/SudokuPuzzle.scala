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
	private val myLastPossibleNumberIndex = Array.ofDim[Int](CellCount) //next index to use in each array of possible numbers
	private val rowNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val colNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val squareNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
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
				saveValueForCell(curValue, i)
			}
		}

		//Test:
		for(i <- 0 until CellCount) {
			println(s"$i: " + getPossibleNumbers(i).toVector)
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
			val countOfIndex = getPossibleCounts(i)
			numberOffsets(countOfIndex + 1)+=1
		}
		for(i <- 1 until PuzzleSize){ //correct offsets
			numberOffsets(i)+=numberOffsets(i - 1)
		}
		println(numberOffsets.toVector)
		for(i <- 0 until CellCount){
			val countOfIndex = getPossibleCounts(i)
			val offset = numberOffsets(countOfIndex)
			require(myIndices(offset) == 0, s"tried to overwrite $i / $offset ")
			myIndices(offset) = i
			numberOffsets(countOfIndex)+=1
		}
		println(myIndices.toVector)

		//3. solve the puzzle by backtracking
		var i = 0
		while(i < CellCount){
			val puzzleIndex = myIndices(i)
			val curValue = myPuzzle(puzzleIndex)
			if(curValue == 0){ //Is already solved?
				//myPuzzle()
			} else {
				i+=1
			}
		}
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

	//methode zeimal + einmal revert! mit neunen num arrays
	private def saveValueForCell(value: Int, index: Int) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		rowNums(rowIndex) |= checkBit
		colNums(colIndex) |= checkBit
		squareNums(squareIndex) |= checkBit
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

	private def getPossibleNumbers(index: Int) : Array[Byte]= {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val possibleNumberIndex = rowNums(rowIndex) | colNums(colIndex) | squareNums(squareIndex)
		SudokuConstants.BitsetPossibleNumbers(possibleNumberIndex)
	}

	private def getPossibleCounts(index: Int) : Int = {
		if(myPuzzle(index) == 0){
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
