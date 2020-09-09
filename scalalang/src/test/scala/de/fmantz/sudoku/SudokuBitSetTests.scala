package de.fmantz.sudoku

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SudokuBitSetTests extends AnyFlatSpec with Matchers {

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
