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

	it should "read correct number of sudokus" in {
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

	private def readFile(path: String) : Array[String] = {
		val rs = ArrayBuffer.empty[String]
		val buffer = ArrayBuffer.empty[String]
		val source = Source.fromFile(path)
		for (line <- source.getLines()) {
			if (line.isEmpty || line.startsWith(SudokuConstants.NewSudokuSeparator)) {
				if (buffer.nonEmpty) {
					rs.append(buffer.mkString("\n"))
				}
				buffer.clear()
			} else {
				buffer.append(line.trim)
			}
		}
		source.close()
		if (buffer.nonEmpty) {
			rs.append(buffer.mkString("\n"))
		}
		rs.toArray
	}

}
