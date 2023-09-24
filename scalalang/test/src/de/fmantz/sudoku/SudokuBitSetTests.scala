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

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SudokuBitSetTests extends AnyFlatSpec with Matchers {

//  "toString" should "work correctly" in {
//    val testObject = new SudokuBitSet
//    testObject.saveValue(value = 5)
//    testObject.toString shouldBe "BITS=0b000010000"
//  }

  "isFoundNumbersUnique" should "work correctly" in {

    val testObject = new SudokuBitSet
    testObject.isFoundNumbersUnique shouldBe true
    testObject.saveValue(value = 5)
    testObject.isFoundNumbersUnique shouldBe true
    testObject.saveValue(value = 5)
    testObject.isFoundNumbersUnique shouldBe false

  }

  "isAllNumbersFound" should "work correctly" in {

    val testObject = new SudokuBitSet

    testObject.isAllNumbersFound shouldBe false

    (0 to 8).foreach(n => {
      testObject.saveValue(n)
      testObject.isAllNumbersFound shouldBe false
    })

    testObject.saveValue(value = 9)
    testObject.isAllNumbersFound shouldBe true

  }

}
