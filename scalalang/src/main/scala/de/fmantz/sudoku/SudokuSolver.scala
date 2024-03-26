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
import de.fmantz.sudoku.SudokuIO.{read, write}

import scala.concurrent.duration.Duration

//import scala.collection.parallel.CollectionConverters.ImmutableIterableIsParallelizable
import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.ExecutionContext.Implicits.global

object SudokuSolver {

	def main(args: Array[String]): Unit = {
		if (args.isEmpty) {
			println(">SudokuSolver inputFile [outputFile]")
			println("-First argument must be path to sudoku puzzles!")
			println("-Second argument can be output path for sudoku puzzles solution!")
		} else {
			val startTotal = System.currentTimeMillis()
			val inputPath = args.head
			val inputFile = new File(inputPath)
			val defaultOutputPath = s"${inputFile.getParentFile.getPath}${File.separator}SOLUTION_${inputFile.getName}"
			val outputPath = args.tail.headOption.getOrElse(defaultOutputPath)
			println("input: " + inputFile.getAbsolutePath)
			val (source, puzzles) = read(inputPath)
			try {
				puzzles
					.grouped(SudokuConstants.ParallelizationCount)
					.foreach({ g =>
						val puzzlesSolvedF: Iterator[Future[SudokuPuzzle]] = g.map({ sudoku => Future {
							solveCurrentSudoku(sudoku); sudoku //solve in parallel!
						}}).iterator
						val puzzlesSolved = puzzlesSolvedF.map(Await.result(_, Duration.Inf))
						write(outputPath, puzzlesSolved)
					})
				println("output: " + new File(outputPath).getAbsolutePath)
				println(s"All sudoku puzzles solved by simple backtracking algorithm in ${System.currentTimeMillis() - startTotal} ms")
			} finally {
				source.close()
			}
		}
	}

	private def solveCurrentSudoku(sudoku: SudokuPuzzle): Unit = {
		val solved = sudoku.solve()
		if (!solved) {
			println(s"Sudoku index is unsolvable:\n" + sudoku.toPrettyString)
		}
	}

}
