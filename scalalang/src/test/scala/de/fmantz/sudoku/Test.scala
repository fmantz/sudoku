package de.fmantz.sudoku

object Test {

	def main(args: Array[String]): Unit = {
//		SudokuConstants
//			 .BitsetArray
//			 .zipWithIndex
//			 .foreach({ case (a, i) =>
//					println(s"__constant__ char BITSET_NUMBERS_%03d[] = {${a.mkString(", ")}};".format(i))
//		  })

		val a = SudokuConstants
					 .BitsetArray
					 .map(_.length)

		println(s"const BITSET_LENGTH: &[u8] = &[${a.mkString(", ")}];")

	}

}
