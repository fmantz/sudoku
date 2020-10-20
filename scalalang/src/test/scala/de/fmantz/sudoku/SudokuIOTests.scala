package de.fmantz.sudoku

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

class SudokuIOTests extends AnyFlatSpec with Matchers {

	"read" should "correctly parse sudokus" in {
		val fileName = this.getClass.getResource("/").getPath + "/p096_sudoku.txt"
		val expectedRs = readFile(fileName)
		var counter = 0
		val (source, puzzles) = SudokuIO.read(fileName)
		try {
			puzzles.zipWithIndex.foreach({ case (read, index) =>
				read.toString shouldBe expectedRs(index)
				counter += 1
			})
			counter shouldBe 51
		} finally {
			source.close()
		}
	}

	it should "correct number of documents" in {
		val fileName = this.getClass.getResource("/").getPath + "/sudoku.txt"
		val expectedLength = readFile(fileName).length
		val (source, puzzles) = SudokuIO.read(fileName)
		try {
			val readLength = puzzles.length
			readLength shouldBe expectedLength
		} finally {
			source.close()
		}
	}

	private def readFile(fileName: String) : Array[String] = {
		val rs = ArrayBuffer.empty[String]
		val buffer = ArrayBuffer.empty[String]
		for (line <- Source.fromFile(fileName).getLines()) {
			if (line.isEmpty || line.startsWith(SudokuConstants.NewSudokuSeparator)) {
				if (buffer.nonEmpty) {
					rs.append(buffer.mkString("\n"))
				}
				buffer.clear()
			} else {
				buffer.append(line.trim)
			}
		}
		if (buffer.nonEmpty) {
			rs.append(buffer.mkString("\n"))
		}
		rs.toArray
	}

}
