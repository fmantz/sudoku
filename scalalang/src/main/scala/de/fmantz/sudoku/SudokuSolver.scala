package de.fmantz.sudoku

import java.io.File

import de.fmantz.sudoku.SudokuIO.{read, writeQQWing}

object SudokuSolver {

	def main(args: Array[String]): Unit = {
		if(args.isEmpty){
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
			val puzzles = read(inputFileName)
			puzzles.zipWithIndex.foreach({ case (sudoku, index) =>
				if(sudoku.isSolved){
					println(s"Sudoku ${index + 1} is already solved!")
				} else if(sudoku.isSolvable){
					sudoku.solve()
					if(!sudoku.isSolved){
						println(s"ERROR: Sudoku ${index + 1} is not correctly solved!")
					}
				} else {
					println(s"Sudoku ${index + 1} is unsolvable!")
				}
			})
			writeQQWing(outputFileName, puzzles)
			println("output:" + new File(outputFileName).getAbsolutePath)
			println(s"All sudoku puzzles solved by simple backtracking algorithm in ${System.currentTimeMillis() - startTotal} ms")
		}
	}

}
