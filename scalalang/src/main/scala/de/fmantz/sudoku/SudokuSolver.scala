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

import java.io.File

import de.fmantz.sudoku.SudokuIO.{read, writeQQWing}

object SudokuSolver {

	def main(args: Array[String]): Unit = {
		if (args.isEmpty) {
			println(">SudokuSolver inputFile [outputFile]")
			println("-First argument must be path to sudoku puzzles!")
			println("-Second argument can be output path for sudoku puzzles solution!")
		} else {
			val startTotal = System.currentTimeMillis()
			val inputFileName = args.head
			val inputFile = new File(inputFileName)
			val defaultOutputFileName = s"${inputFile.getParentFile.getPath}${File.separator}SOLUTION_${inputFile.getName}"
			val outputFileName = args.tail.headOption.getOrElse(defaultOutputFileName)
			println("input:" + inputFile.getAbsolutePath)
			val (source, puzzles) = read(inputFileName)
			try {
				var index = 0
				puzzles
					.grouped(SudokuConstants.ParallizationCount)
					.foreach({ g =>
						val puzzlesSolved = g.zipWithIndex.par.map({ case (sudoku, index) =>
							solveCurrentSudoku(index, sudoku) //solve in parallel!
						}).toIterator
						writeQQWing(outputFileName, puzzlesSolved)
					})
				println("output:" + new File(outputFileName).getAbsolutePath)
				println(s"All sudoku puzzles solved by simple backtracking algorithm in ${System.currentTimeMillis() - startTotal} ms")
			} finally {
				source.close()
			}
		}
	}

	private def solveCurrentSudoku(index: Int, sudoku: SudokuPuzzle): SudokuPuzzle = {
		sudoku.initTurbo()
		if (sudoku.isSolved) {
			println(s"Sudoku $index is already solved!")
		} else if (sudoku.isSolvable) {
			sudoku.solve()
		} else {
			println(s"Sudoku index is unsolvable:\n" + sudoku.toPrettyString)
		}
		sudoku
	}

}
