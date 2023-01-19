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

import java.io.{FileWriter, PrintWriter}

import scala.io.{BufferedSource, Source}

object SudokuIO {

	/**
	 * Read usual 9x9 Suduko from text file
	 */
	def read(path: String): (BufferedSource, Iterator[SudokuPuzzle]) = {
		val source: BufferedSource = Source.fromFile(path)
		val iter = new SudokuIterator(source.getLines())
		(source, iter)
	}

	def write(path: String, puzzles: Iterator[SudokuPuzzle]): Unit = {
		val writer = new PrintWriter(new FileWriter(path, true))
		try {
			puzzles.foreach({ sudoku =>
				writer.println(sudoku)
				writer.println()
				writer.flush()
			})
		} finally {
			writer.close()
		}
	}
}
