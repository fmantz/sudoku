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

	def get(row: Int, col: Int): Byte

	def set(row: Int, col: Int, value: Byte): Unit

	def init(): Unit

	def isSolvable: Boolean

	def isSolved: Boolean

	def solve(): Unit

	def toPrettyString: String

}

class SudokuPuzzleImpl extends SudokuPuzzle {

	import SudokuConstants._

	//state:
	private var myIsSolvable: Boolean = true
	private var myIsSolved: Boolean = false

	private val puzzle = Array.ofDim[Byte](CellCount)
	private val puzzleSorted = Array.ofDim[Byte](CellCount)
	private val indices = Array.ofDim[Int](CellCount)

	private val rowNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val colNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val squareNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)

	override def get(row: Int, col: Int): Byte = {
		puzzle(getSingleArrayIndex(row, col))
	}

	override def set(row: Int, col: Int, value: Byte): Unit = {
		puzzle(getSingleArrayIndex(row, col)) = value
	}

	private def getSingleArrayIndex(row: Int, col: Int) = {
		row * PuzzleSize + col
	}

	override def isSolved: Boolean = {
		myIsSolved
	}

	override def isSolvable: Boolean = {
		myIsSolvable
	}

	override def init(): Unit = {
		findAllPossibleValuesForEachEmptyCell()
		preparePuzzleForSolving()
	}

	/**
	 * solves the sudoku by a simple non-recursive backtracking algorithm (brute force)
	 * (own simple solution, its an algorithm which may be ported to CUDA or OpenCL)
	 * to get a faster result use e.g. https://github.com/Emerentius/sudoku
	 */
	override def solve(): Unit = {
		if(isSolvable && !isSolved) {
			findSolutionNonRecursively()
		}
	}

	private def findAllPossibleValuesForEachEmptyCell(): Unit = {
		for (i <- puzzle.indices) {
			val curValue = puzzle(i)
			if (curValue > 0) {
				saveValueForCell(curValue, i)
			}
		}
	}

	private def preparePuzzleForSolving(): Unit = {
		val numberOffsets = Array.ofDim[Int](PuzzleSize + 2) //counts 0 - 9 + 1 offset = puzzleSize + 2 (9 + 2)
		for(i <- 0 until CellCount){
			val countOfIndex = getPossibleCounts(i)
			numberOffsets(countOfIndex + 1)+=1
		}
		myIsSolved = numberOffsets(1) == CellCount //all cells have already a solution!
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
	}

	def findSolutionNonRecursively(): Unit = {
		var lastInvaldTry: Byte = 0
		var i = 0
		while (i < CellCount) {
			val curValue = puzzleSorted(i)
			if (curValue == 0) { //Is not given?

				//Is there a current guess possible?
				val puzzleIndex = indices(i)
				val rowIndex = calculateRowIndex(puzzleIndex)
				val colIndex = calculateColIndex(puzzleIndex)
				val squareIndex = calculateSquareIndex(rowIndex, colIndex)
				val possibleNumberIndex = rowNums(rowIndex) | colNums(colIndex) | squareNums(squareIndex)
				val nextNumbers = SudokuConstants.BitsetPossibleNumbers(possibleNumberIndex)
				val nextNumberIndex = if (lastInvaldTry == 0) {
					0
				} else {
					fastIndexOf(nextNumbers, lastInvaldTry) + 1
				}

				if (nextNumberIndex < nextNumbers.length) {
					//next possible number to try found:
					val nextNumber = nextNumbers(nextNumberIndex)
					puzzleSorted(i) = nextNumber
					saveValueForCell(nextNumber, rowIndex, colIndex, squareIndex)
					lastInvaldTry = 0 //0 since success
					i += 1 //go to next cell
				} else {
					i -= 1 //backtrack, note not given values are in the head of myIndices, we can simply go one step back!
					lastInvaldTry = puzzleSorted(i)
					puzzleSorted(i) = 0
					val lastPuzzleIndex = indices(i)
					revertValueForCell(lastInvaldTry, lastPuzzleIndex)
				}

			} else {
				i += 1 //value was given!
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
		var run = true
		var index = 0
		while(run){
			if(array(index) != b){
				index+=1
			} else {
				run = false
			}
		}
		index
	}

	private def saveValueForCell(value: Int, index: Int) : Unit = {
		val rowIndex = calculateRowIndex(index)
		val colIndex = calculateColIndex(index)
		val squareIndex = calculateSquareIndex(rowIndex, colIndex)
		val checkBit: Int = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
		setAndCheckBit(checkBit, rowNums, rowIndex)
		setAndCheckBit(checkBit, colNums, colIndex)
		setAndCheckBit(checkBit, squareNums, squareIndex)
	}

	private def setAndCheckBit(checkBit: Int, array:Array[Int], index: Int) : Unit = {
		val oldValue = array(index)
		array(index) |= checkBit
		myIsSolvable &= oldValue != array(index)
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
