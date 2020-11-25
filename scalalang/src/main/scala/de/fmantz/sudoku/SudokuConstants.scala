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

object SudokuConstants {

	final val NewSudokuSeparator: String = "Grid"
	final val EmptyChar: Char = '0'
	final val QQWingEmptyChar: Char = '.' //https://qqwing.com
	final val PuzzleSize: Int = 9
	final val SquareSize: Int = 3
	final val CheckBits = ~0 >>> (32 - SudokuConstants.PuzzleSize) //binary: Size times "1"

}
