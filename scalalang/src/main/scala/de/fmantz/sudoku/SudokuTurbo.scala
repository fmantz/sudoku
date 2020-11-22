package de.fmantz.sudoku

class SudokuTurbo private () {

	import SudokuTurbo._

	private val colCounts: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val rowCounts: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)

	private val colNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val rowNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)
	private val squareNums: Array[Int] = Array.ofDim[Int](SudokuConstants.PuzzleSize)

	var rowIndices: Array[Int] = Array.empty
	var colIndices: Array[Int] = Array.empty

	def createSolutionSpace(row: Int, col: Int): SudokuBitSet = {
		val squareIndex = SudokuTurbo.calculateSquareIndex(row, col)
		val bits: Int = colNums(col) | rowNums(row) | squareNums(squareIndex)
		new SudokuBitSet(bits)
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

}

object SudokuTurbo {

	def create(puzzleData: Array[Array[Int]]): SudokuTurbo = {
		val rs = new SudokuTurbo()
		var row, col = 0
		while (row < SudokuConstants.PuzzleSize) {
			val rowData = puzzleData(row)
			while (col < SudokuConstants.PuzzleSize) {
				rs.saveValue(row, col, rowData(col))
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

	private def createSortedIndices(num:Array[Int]) : Array[Int] = {
		num.zipWithIndex.sortBy(_._1).reverse.map(_._2) ++ Array(-1) //sort according to number heuristic
	}

}
