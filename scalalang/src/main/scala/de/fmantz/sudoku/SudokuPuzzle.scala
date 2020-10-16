package de.fmantz.sudoku

import scala.collection.mutable.ListBuffer

trait SudokuPuzzle {
  def set(row: Int, col: Int, value: Int): Unit
  def isSolvable: Boolean
  def isSolved: Boolean
  def solve(): Unit
  def toPrettyString: String
}

object SudokuPuzzle extends SudokuPuzzle {

  final val Size = 9
  final val SquareSize = 3

  //state:
  private val puzzle: Array[Array[Int]] = Array.ofDim[Int](Size, Size)
  private var open: Boolean = true
  private var isEmpty: Boolean = true

  def nonEmpty: Boolean = !this.isEmpty

  override def set(row: Int, col: Int, value: Int): Unit = {
    if(open){
      puzzle(row)(col) = value
      isEmpty = false
    }
  }

  private def isEmpty(row: Int, col: Int): Boolean = {
    puzzle(row)(col) == 0
  }

  override def isSolved: Boolean = checkConditions(relaxed = false)

  override def isSolvable: Boolean = checkConditions(relaxed = true)

  /**
   * solves the suduko by a simple backtracking algorithm (brute force)
   * inspired by https://www.youtube.com/watch?v=G_UYXzGuqvM
   */
  override def solve(): Unit = {
    def go(): Unit = {
      var row = 0
      var run = true
      while (run && row < Size) {
        var col = 0
        while (run && col < Size) {
          if (isEmpty(row, col)) {
            val solutionSpace = createSolutionSpace(row, col)
            for (n <- 1 to Size) {
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
        open = false
      }
    }

    go()
    open = true
  }

  override def toString: String = {
    val buffer = new ListBuffer[String]
    for (row <- 0 until Size) {
      buffer.append(puzzle(row).mkString)
    }
    buffer.mkString("\n")
  }

  override def toPrettyString: String = {
    val dottedLine = "-" * (Size * 3 + SquareSize - 1)
    val empty = "*"
    val buffer = new ListBuffer[String]
    buffer.append(dottedLine)
    for (row <- 0 until Size) {
      val formattedRow = puzzle(row).zipWithIndex.map({ case (colValue, col) =>
        val rs = if (colValue == 0) s" $empty " else s" $colValue "
        if (col + 1 < Size && col % SquareSize == 2) {
          rs + "|"
        } else {
          rs
        }
      }).mkString("")
      buffer.append(formattedRow)
      if ((row + 1) % SquareSize == 0) {
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
    bits: SudokuBitSet = SudokuBitSet.FirstInstance.init()
  ): SudokuBitSet = {
    val selectedRow = puzzle(row)
    var col = 0
    while (col < Size) {
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
    bits: SudokuBitSet = SudokuBitSet.FirstInstance.init()
  ): SudokuBitSet = {
    var row = 0
    while (row < Size) {
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
    bits: SudokuBitSet = SudokuBitSet.FirstInstance.init()
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
    (0 until Size).forall(row => isRowOK(row, relaxed)) &&
      (0 until Size).forall(col => isColOK(col, relaxed)) &&
        (0 until Size).forall(i => isSquareOK(i / SquareSize, i % SquareSize, relaxed))
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
