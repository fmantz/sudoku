//scalastyle:off
/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.txt file in the top level folder.
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

object SudokuBitSet {
  private final val CheckBits = ~0 >>> (32 - SudokuConstants.PuzzleSize) //binary: Size times "1"
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

//  def hasSolution: Boolean = {
//    !isAllNumbersFound
//  }

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

//  override def toString: String = {
//     s"BITS=0b%0${SudokuConstants.PuzzleSize}d".format(this.bits.toBinaryString.toInt)
//  }

}
