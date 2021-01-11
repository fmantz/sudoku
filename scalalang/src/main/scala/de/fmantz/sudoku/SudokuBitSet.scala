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

class SudokuBitSet( private var bits: Int) {

  def this(){
    this(bits = 0)
  }

  //stores numbers as bits:
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
    bits == SudokuConstants.CheckBits
  }

//  def hasSolution: Boolean = {
//    !isAllNumbersFound
//  }

  def isFoundNumbersUnique: Boolean = {
    notFoundBefore
  }

//  def isSolution(sol: Int): Boolean = {
//    if(sol > 0){
//      val checkBit = 1 << sol - 1
//      (bits & checkBit) == 0
//    } else {
//      false
//    }
//  }

//  override def toString: String = {
//     s"BITS=0b%0${SudokuConstants.PuzzleSize}d".format(this.bits.toBinaryString.toInt)
//  }

  def possibleNumbers: Array[Byte] = {
    SudokuConstants.BitsetPossibleNumbers(this.bits)
  }

}

object SudokuBitSet {

  def main(args: Array[String]): Unit = {

    import scala.util.Random

    val xs = (1 to 9).toVector
    var powerset: Seq[Vector[Int]] = Vector.empty
    var counter: Array[Array[Int]] = Array.empty
    while(counter.isEmpty || counter.exists({ a =>
      val avg = a.sum / 9d
      a.exists(v => Math.abs(v - avg) > 6)
    })) {
      counter = Array.ofDim[Int](9, 9)
      powerset = ((0 to xs.size) flatMap xs.combinations).map(s => Random.shuffle(s)).toVector
      powerset.foreach( a =>
        for(i <- a.indices){
          val v = a(i)
          counter(i)(v-1)+=1 //column 1 all ones
        }
      )
    }

    //Counter:
    println()
    counter
      .zipWithIndex
      .foreach({ case(a, i) => println(s"pos $i: ${a.mkString(", ")}")})
    println()

    val mapping: Seq[(Int, Vector[Int])] = powerset.map(s => {
      val bitset = new SudokuBitSet(0)
      val oppositeNumbers = (1 to 9).toVector.filterNot(s.contains)
      oppositeNumbers.foreach(bitset.saveValue)
      val bitsetValue = bitset.bits
      (bitsetValue, s)
    }).toVector.sortBy(_._1)

    println("length=" + mapping.length)

    val codegen1 = mapping
      .map({ case (i, a) =>
        s"private final val BitsetNumbers_%03d: Array[Byte] = $a".format(i).replaceAll("Vector", "Array")
      })
      .mkString("\n")

    println(codegen1)

    val codegen2 = mapping
      .map({ case (i, a) =>
        s"const BITSET_NUMBERS_%03d: &[u8] = $a".format(i).replaceAll("Vector\\(", "&[").replaceAll("\\)", "];")
      })
      .mkString("\n")

    println(codegen2)

  }
}