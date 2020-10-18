package de.fmantz.sudoku

object SudokuBitSet {
  private final val CheckBits = ~0 >>> (32 - SudokuPuzzle.Size) //binary: Size times "1"
}

class SudokuBitSet {

  import SudokuBitSet._

  //stores numbers as bits:
  private var bits: Int = 0
  private var notFoundBefore: Boolean = true

  /**
   * Save a value
   */
  def saveValue(value: Int): Unit = {
    if (value > 0) {
      val checkBit = 1 << (value - 1) //set for each number a bit by index from left, number 1 has index zero
      notFoundBefore &&= firstMatch(bits, checkBit)
      bits |= checkBit
    }
  }

  @inline private def firstMatch(bits: Int, checkBit: Int): Boolean = {
    (bits & checkBit) == 0
  }

  def isAllNumbersFound: Boolean = {
    bits == CheckBits
  }

  def hasSolution: Boolean = {
    !isAllNumbersFound
  }

  def isFoundNumbersUnique: Boolean = {
    notFoundBefore
  }

  def isSolution(sol: Int): Boolean = {
    if(sol > 0){
      val checkBit = 1 << sol - 1
      (bits & checkBit) == 0
    } else {
      false
    }
  }

  override def toString: String = {
     s"BITS=%0${SudokuPuzzle.Size}d".format(this.bits.toBinaryString.toInt)
  }
}
