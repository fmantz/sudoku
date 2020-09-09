package de.fmantz.sudoku

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object SudokuReader {

  private final val NewSudokuSeperator: String = "Grid"

  /**
   * Read usual 9x9 Suduko from text file in resources folder
   */
  def read(filename: String) : IndexedSeq[SudokuPuzzle] = {
    val rs = ArrayBuffer.empty[SudokuPuzzle]
    var currentSuko:SudokuPuzzle = null
    var currentRow = 0
    for (line <- Source.fromResource(filename).getLines) {
      if(line.startsWith(NewSudokuSeperator)){
        currentSuko= new SudokuPuzzle
        currentRow = 0
        rs.append(currentSuko)
      } else {
        val rowValues = line.trim.toCharArray.map(_ - '0')
        for(col <- 0 until SudokuPuzzle.Size){
          currentSuko.set(currentRow, col, rowValues(col))
        }
        currentRow += 1
      }
    }
    rs.toIndexedSeq
  }

  def main(args: Array[String]): Unit = {
    read("p096_sudoku.txt").zipWithIndex.foreach({ case (sudoku, index) =>
      println(s"Soduko ${index + 1}")
      println(sudoku)
      println()
    })
  }

}
