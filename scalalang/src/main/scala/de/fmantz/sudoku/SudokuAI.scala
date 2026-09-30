package de.fmantz.sudoku

/**
 * This solver is generated with Gemnini in google KI mode
 * I ask for generating a efficient Sudoku solver and afterwards to fix two compiler warnings.
 * I manually changed only that the board is a SudokuPuzzle and not an array[array[Int]]
 */
import scala.util.boundary

object BitmaskSudokuSolver:

  inline def blockIndex(r: Int, c: Int): Int = (r / 3) * 3 + (c / 3)

  def solve(board: SudokuPuzzle): Option[SudokuPuzzle] =
    val rows   = Array.fill(9)(0)
    val cols   = Array.fill(9)(0)
    val blocks = Array.fill(9)(0)

    for
      r <- 0 until 9
      c <- 0 until 9
      num = board.get(r, c)
      if num > 0
    do
      val mask = 1 << num
      rows(r) |= mask
      cols(c) |= mask
      blocks(blockIndex(r, c)) |= mask

    // Wir umschließen das Backtracking mit einem Boundary-Block
    def backtrack(): Boolean = boundary:
      var minChoices = 10
      var bestR = -1
      var bestC = -1
      var bestChoicesMask = 0

      for
        r <- 0 until 9
        c <- 0 until 9
        if board.get(r, c) == 0
      do
        val used = rows(r) | cols(c) | blocks(blockIndex(r, c))
        val allowed = (~used) & 0x3FE
        val choiceCount = java.lang.Integer.bitCount(allowed)

        if choiceCount == 0 then boundary.break(false)

        if choiceCount < minChoices then
          minChoices = choiceCount
          bestR = r
          bestC = c
          bestChoicesMask = allowed

      if bestR == -1 then boundary.break(true)

      var choices = bestChoicesMask
      while choices > 0 do
        val nextBit = choices & -choices
        val num = java.lang.Integer.numberOfTrailingZeros(nextBit)
        val bIdx = blockIndex(bestR, bestC)

        board.set(bestR, bestC, num.toByte)
        rows(bestR) |= nextBit
        cols(bestC) |= nextBit
        blocks(bIdx) |= nextBit

        if backtrack() then boundary.break(true)

        board.set(bestR, bestC, 0)
        rows(bestR) &= ~nextBit
        cols(bestC) &= ~nextBit
        blocks(bIdx) &= ~nextBit

        choices &= ~nextBit

      false
    end backtrack

    if backtrack() then Some(board) else None

