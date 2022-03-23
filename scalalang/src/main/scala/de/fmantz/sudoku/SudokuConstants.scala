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
	final val CellCount: Int = 81
	final val PuzzleSize: Int = 9
	final val SquareSize: Int = 3
	final val CheckBits = ~0 >>> (32 - SudokuConstants.PuzzleSize) //binary: Size times "1"
	final val ParallelizationCount: Int = 65536

	private final val BitsetNumbers_000: Array[Byte] = Array(4, 7, 8, 3, 5, 1, 9, 6, 2)
	private final val BitsetNumbers_001: Array[Byte] = Array(8, 6, 7, 5, 9, 4, 2, 3)
	private final val BitsetNumbers_002: Array[Byte] = Array(3, 4, 5, 1, 9, 6, 7, 8)
	private final val BitsetNumbers_003: Array[Byte] = Array(7, 8, 9, 5, 4, 6, 3)
	private final val BitsetNumbers_004: Array[Byte] = Array(9, 1, 8, 6, 7, 5, 4, 2)
	private final val BitsetNumbers_005: Array[Byte] = Array(8, 6, 5, 7, 4, 2, 9)
	private final val BitsetNumbers_006: Array[Byte] = Array(6, 9, 4, 5, 8, 7, 1)
	private final val BitsetNumbers_007: Array[Byte] = Array(9, 7, 4, 6, 8, 5)
	private final val BitsetNumbers_008: Array[Byte] = Array(1, 8, 9, 3, 2, 5, 7, 6)
	private final val BitsetNumbers_009: Array[Byte] = Array(3, 7, 2, 5, 9, 8, 6)
	private final val BitsetNumbers_010: Array[Byte] = Array(7, 6, 9, 3, 1, 8, 5)
	private final val BitsetNumbers_011: Array[Byte] = Array(7, 8, 3, 6, 5, 9)
	private final val BitsetNumbers_012: Array[Byte] = Array(8, 2, 6, 7, 9, 1, 5)
	private final val BitsetNumbers_013: Array[Byte] = Array(7, 5, 9, 8, 6, 2)
	private final val BitsetNumbers_014: Array[Byte] = Array(9, 8, 7, 1, 5, 6)
	private final val BitsetNumbers_015: Array[Byte] = Array(6, 8, 7, 9, 5)
	private final val BitsetNumbers_016: Array[Byte] = Array(8, 2, 9, 7, 1, 3, 6, 4)
	private final val BitsetNumbers_017: Array[Byte] = Array(3, 4, 7, 8, 2, 6, 9)
	private final val BitsetNumbers_018: Array[Byte] = Array(6, 1, 4, 3, 9, 7, 8)
	private final val BitsetNumbers_019: Array[Byte] = Array(9, 3, 7, 4, 6, 8)
	private final val BitsetNumbers_020: Array[Byte] = Array(7, 2, 6, 9, 4, 1, 8)
	private final val BitsetNumbers_021: Array[Byte] = Array(4, 9, 6, 2, 7, 8)
	private final val BitsetNumbers_022: Array[Byte] = Array(7, 8, 4, 1, 6, 9)
	private final val BitsetNumbers_023: Array[Byte] = Array(9, 4, 8, 7, 6)
	private final val BitsetNumbers_024: Array[Byte] = Array(7, 9, 8, 2, 3, 6, 1)
	private final val BitsetNumbers_025: Array[Byte] = Array(3, 6, 9, 8, 7, 2)
	private final val BitsetNumbers_026: Array[Byte] = Array(3, 6, 1, 8, 9, 7)
	private final val BitsetNumbers_027: Array[Byte] = Array(8, 3, 7, 6, 9)
	private final val BitsetNumbers_028: Array[Byte] = Array(8, 1, 2, 7, 9, 6)
	private final val BitsetNumbers_029: Array[Byte] = Array(6, 8, 9, 7, 2)
	private final val BitsetNumbers_030: Array[Byte] = Array(1, 8, 9, 7, 6)
	private final val BitsetNumbers_031: Array[Byte] = Array(8, 9, 7, 6)
	private final val BitsetNumbers_032: Array[Byte] = Array(8, 7, 3, 1, 2, 5, 9, 4)
	private final val BitsetNumbers_033: Array[Byte] = Array(9, 3, 5, 4, 2, 7, 8)
	private final val BitsetNumbers_034: Array[Byte] = Array(7, 3, 8, 4, 9, 5, 1)
	private final val BitsetNumbers_035: Array[Byte] = Array(9, 4, 3, 7, 5, 8)
	private final val BitsetNumbers_036: Array[Byte] = Array(2, 7, 8, 4, 9, 1, 5)
	private final val BitsetNumbers_037: Array[Byte] = Array(4, 7, 5, 8, 2, 9)
	private final val BitsetNumbers_038: Array[Byte] = Array(5, 7, 9, 8, 1, 4)
	private final val BitsetNumbers_039: Array[Byte] = Array(5, 8, 9, 7, 4)
	private final val BitsetNumbers_040: Array[Byte] = Array(7, 5, 8, 9, 3, 1, 2)
	private final val BitsetNumbers_041: Array[Byte] = Array(9, 8, 3, 7, 5, 2)
	private final val BitsetNumbers_042: Array[Byte] = Array(5, 3, 9, 7, 8, 1)
	private final val BitsetNumbers_043: Array[Byte] = Array(5, 3, 9, 8, 7)
	private final val BitsetNumbers_044: Array[Byte] = Array(5, 2, 9, 7, 1, 8)
	private final val BitsetNumbers_045: Array[Byte] = Array(5, 8, 9, 2, 7)
	private final val BitsetNumbers_046: Array[Byte] = Array(1, 5, 7, 9, 8)
	private final val BitsetNumbers_047: Array[Byte] = Array(7, 8, 9, 5)
	private final val BitsetNumbers_048: Array[Byte] = Array(7, 9, 8, 2, 1, 4, 3)
	private final val BitsetNumbers_049: Array[Byte] = Array(8, 9, 3, 7, 4, 2)
	private final val BitsetNumbers_050: Array[Byte] = Array(1, 3, 4, 7, 8, 9)
	private final val BitsetNumbers_051: Array[Byte] = Array(9, 4, 7, 3, 8)
	private final val BitsetNumbers_052: Array[Byte] = Array(7, 4, 8, 2, 1, 9)
	private final val BitsetNumbers_053: Array[Byte] = Array(4, 2, 7, 8, 9)
	private final val BitsetNumbers_054: Array[Byte] = Array(1, 9, 7, 8, 4)
	private final val BitsetNumbers_055: Array[Byte] = Array(4, 9, 8, 7)
	private final val BitsetNumbers_056: Array[Byte] = Array(2, 9, 1, 3, 8, 7)
	private final val BitsetNumbers_057: Array[Byte] = Array(2, 8, 9, 7, 3)
	private final val BitsetNumbers_058: Array[Byte] = Array(8, 9, 1, 7, 3)
	private final val BitsetNumbers_059: Array[Byte] = Array(3, 7, 9, 8)
	private final val BitsetNumbers_060: Array[Byte] = Array(8, 7, 1, 9, 2)
	private final val BitsetNumbers_061: Array[Byte] = Array(2, 7, 8, 9)
	private final val BitsetNumbers_062: Array[Byte] = Array(1, 8, 7, 9)
	private final val BitsetNumbers_063: Array[Byte] = Array(9, 7, 8)
	private final val BitsetNumbers_064: Array[Byte] = Array(6, 5, 3, 8, 2, 4, 1, 9)
	private final val BitsetNumbers_065: Array[Byte] = Array(9, 2, 5, 3, 6, 8, 4)
	private final val BitsetNumbers_066: Array[Byte] = Array(5, 4, 6, 8, 1, 9, 3)
	private final val BitsetNumbers_067: Array[Byte] = Array(6, 8, 4, 5, 9, 3)
	private final val BitsetNumbers_068: Array[Byte] = Array(1, 8, 5, 2, 6, 9, 4)
	private final val BitsetNumbers_069: Array[Byte] = Array(6, 9, 5, 2, 8, 4)
	private final val BitsetNumbers_070: Array[Byte] = Array(8, 9, 5, 6, 1, 4)
	private final val BitsetNumbers_071: Array[Byte] = Array(9, 8, 6, 5, 4)
	private final val BitsetNumbers_072: Array[Byte] = Array(2, 5, 9, 3, 1, 6, 8)
	private final val BitsetNumbers_073: Array[Byte] = Array(5, 6, 8, 3, 2, 9)
	private final val BitsetNumbers_074: Array[Byte] = Array(6, 5, 8, 9, 3, 1)
	private final val BitsetNumbers_075: Array[Byte] = Array(3, 9, 5, 6, 8)
	private final val BitsetNumbers_076: Array[Byte] = Array(1, 2, 6, 5, 9, 8)
	private final val BitsetNumbers_077: Array[Byte] = Array(5, 8, 9, 6, 2)
	private final val BitsetNumbers_078: Array[Byte] = Array(5, 9, 1, 8, 6)
	private final val BitsetNumbers_079: Array[Byte] = Array(5, 8, 9, 6)
	private final val BitsetNumbers_080: Array[Byte] = Array(8, 1, 2, 6, 9, 3, 4)
	private final val BitsetNumbers_081: Array[Byte] = Array(3, 4, 8, 6, 2, 9)
	private final val BitsetNumbers_082: Array[Byte] = Array(3, 8, 4, 1, 6, 9)
	private final val BitsetNumbers_083: Array[Byte] = Array(9, 6, 3, 8, 4)
	private final val BitsetNumbers_084: Array[Byte] = Array(9, 1, 4, 6, 2, 8)
	private final val BitsetNumbers_085: Array[Byte] = Array(2, 8, 6, 9, 4)
	private final val BitsetNumbers_086: Array[Byte] = Array(6, 9, 8, 1, 4)
	private final val BitsetNumbers_087: Array[Byte] = Array(9, 4, 8, 6)
	private final val BitsetNumbers_088: Array[Byte] = Array(1, 2, 6, 9, 3, 8)
	private final val BitsetNumbers_089: Array[Byte] = Array(3, 8, 9, 6, 2)
	private final val BitsetNumbers_090: Array[Byte] = Array(9, 3, 6, 1, 8)
	private final val BitsetNumbers_091: Array[Byte] = Array(8, 3, 9, 6)
	private final val BitsetNumbers_092: Array[Byte] = Array(1, 6, 2, 9, 8)
	private final val BitsetNumbers_093: Array[Byte] = Array(8, 6, 2, 9)
	private final val BitsetNumbers_094: Array[Byte] = Array(9, 1, 8, 6)
	private final val BitsetNumbers_095: Array[Byte] = Array(8, 9, 6)
	private final val BitsetNumbers_096: Array[Byte] = Array(1, 2, 5, 8, 9, 3, 4)
	private final val BitsetNumbers_097: Array[Byte] = Array(9, 5, 2, 8, 3, 4)
	private final val BitsetNumbers_098: Array[Byte] = Array(3, 1, 5, 4, 8, 9)
	private final val BitsetNumbers_099: Array[Byte] = Array(8, 5, 9, 3, 4)
	private final val BitsetNumbers_100: Array[Byte] = Array(9, 1, 5, 8, 4, 2)
	private final val BitsetNumbers_101: Array[Byte] = Array(8, 2, 5, 9, 4)
	private final val BitsetNumbers_102: Array[Byte] = Array(8, 4, 1, 9, 5)
	private final val BitsetNumbers_103: Array[Byte] = Array(5, 4, 9, 8)
	private final val BitsetNumbers_104: Array[Byte] = Array(2, 8, 5, 1, 9, 3)
	private final val BitsetNumbers_105: Array[Byte] = Array(2, 8, 3, 9, 5)
	private final val BitsetNumbers_106: Array[Byte] = Array(1, 9, 5, 8, 3)
	private final val BitsetNumbers_107: Array[Byte] = Array(9, 5, 3, 8)
	private final val BitsetNumbers_108: Array[Byte] = Array(5, 2, 8, 9, 1)
	private final val BitsetNumbers_109: Array[Byte] = Array(8, 5, 9, 2)
	private final val BitsetNumbers_110: Array[Byte] = Array(9, 1, 5, 8)
	private final val BitsetNumbers_111: Array[Byte] = Array(9, 5, 8)
	private final val BitsetNumbers_112: Array[Byte] = Array(2, 9, 4, 1, 8, 3)
	private final val BitsetNumbers_113: Array[Byte] = Array(3, 2, 4, 9, 8)
	private final val BitsetNumbers_114: Array[Byte] = Array(8, 3, 4, 9, 1)
	private final val BitsetNumbers_115: Array[Byte] = Array(9, 8, 4, 3)
	private final val BitsetNumbers_116: Array[Byte] = Array(9, 8, 1, 2, 4)
	private final val BitsetNumbers_117: Array[Byte] = Array(4, 2, 9, 8)
	private final val BitsetNumbers_118: Array[Byte] = Array(9, 1, 8, 4)
	private final val BitsetNumbers_119: Array[Byte] = Array(9, 8, 4)
	private final val BitsetNumbers_120: Array[Byte] = Array(3, 9, 8, 1, 2)
	private final val BitsetNumbers_121: Array[Byte] = Array(8, 2, 3, 9)
	private final val BitsetNumbers_122: Array[Byte] = Array(8, 3, 1, 9)
	private final val BitsetNumbers_123: Array[Byte] = Array(3, 8, 9)
	private final val BitsetNumbers_124: Array[Byte] = Array(1, 9, 2, 8)
	private final val BitsetNumbers_125: Array[Byte] = Array(2, 9, 8)
	private final val BitsetNumbers_126: Array[Byte] = Array(8, 1, 9)
	private final val BitsetNumbers_127: Array[Byte] = Array(8, 9)
	private final val BitsetNumbers_128: Array[Byte] = Array(7, 5, 1, 2, 3, 4, 9, 6)
	private final val BitsetNumbers_129: Array[Byte] = Array(7, 2, 5, 4, 9, 6, 3)
	private final val BitsetNumbers_130: Array[Byte] = Array(4, 7, 1, 3, 9, 6, 5)
	private final val BitsetNumbers_131: Array[Byte] = Array(3, 6, 7, 9, 4, 5)
	private final val BitsetNumbers_132: Array[Byte] = Array(9, 6, 7, 2, 4, 5, 1)
	private final val BitsetNumbers_133: Array[Byte] = Array(5, 7, 4, 6, 2, 9)
	private final val BitsetNumbers_134: Array[Byte] = Array(6, 4, 5, 9, 1, 7)
	private final val BitsetNumbers_135: Array[Byte] = Array(4, 5, 6, 9, 7)
	private final val BitsetNumbers_136: Array[Byte] = Array(1, 2, 7, 6, 3, 9, 5)
	private final val BitsetNumbers_137: Array[Byte] = Array(7, 5, 3, 9, 2, 6)
	private final val BitsetNumbers_138: Array[Byte] = Array(6, 7, 9, 3, 1, 5)
	private final val BitsetNumbers_139: Array[Byte] = Array(5, 7, 3, 9, 6)
	private final val BitsetNumbers_140: Array[Byte] = Array(5, 2, 1, 6, 9, 7)
	private final val BitsetNumbers_141: Array[Byte] = Array(9, 7, 5, 6, 2)
	private final val BitsetNumbers_142: Array[Byte] = Array(6, 1, 7, 5, 9)
	private final val BitsetNumbers_143: Array[Byte] = Array(5, 7, 9, 6)
	private final val BitsetNumbers_144: Array[Byte] = Array(4, 9, 2, 6, 7, 3, 1)
	private final val BitsetNumbers_145: Array[Byte] = Array(3, 4, 7, 2, 6, 9)
	private final val BitsetNumbers_146: Array[Byte] = Array(3, 1, 6, 7, 9, 4)
	private final val BitsetNumbers_147: Array[Byte] = Array(4, 3, 7, 6, 9)
	private final val BitsetNumbers_148: Array[Byte] = Array(4, 7, 9, 2, 6, 1)
	private final val BitsetNumbers_149: Array[Byte] = Array(6, 9, 4, 2, 7)
	private final val BitsetNumbers_150: Array[Byte] = Array(7, 4, 9, 6, 1)
	private final val BitsetNumbers_151: Array[Byte] = Array(6, 7, 4, 9)
	private final val BitsetNumbers_152: Array[Byte] = Array(7, 2, 1, 6, 3, 9)
	private final val BitsetNumbers_153: Array[Byte] = Array(3, 9, 6, 2, 7)
	private final val BitsetNumbers_154: Array[Byte] = Array(3, 6, 9, 7, 1)
	private final val BitsetNumbers_155: Array[Byte] = Array(9, 6, 3, 7)
	private final val BitsetNumbers_156: Array[Byte] = Array(9, 2, 1, 6, 7)
	private final val BitsetNumbers_157: Array[Byte] = Array(9, 7, 2, 6)
	private final val BitsetNumbers_158: Array[Byte] = Array(9, 7, 1, 6)
	private final val BitsetNumbers_159: Array[Byte] = Array(9, 7, 6)
	private final val BitsetNumbers_160: Array[Byte] = Array(2, 9, 4, 1, 7, 3, 5)
	private final val BitsetNumbers_161: Array[Byte] = Array(2, 5, 3, 7, 9, 4)
	private final val BitsetNumbers_162: Array[Byte] = Array(9, 7, 3, 5, 1, 4)
	private final val BitsetNumbers_163: Array[Byte] = Array(9, 4, 7, 5, 3)
	private final val BitsetNumbers_164: Array[Byte] = Array(9, 4, 1, 2, 7, 5)
	private final val BitsetNumbers_165: Array[Byte] = Array(4, 9, 7, 2, 5)
	private final val BitsetNumbers_166: Array[Byte] = Array(4, 7, 9, 1, 5)
	private final val BitsetNumbers_167: Array[Byte] = Array(9, 5, 7, 4)
	private final val BitsetNumbers_168: Array[Byte] = Array(5, 1, 9, 2, 7, 3)
	private final val BitsetNumbers_169: Array[Byte] = Array(9, 3, 2, 5, 7)
	private final val BitsetNumbers_170: Array[Byte] = Array(7, 5, 9, 3, 1)
	private final val BitsetNumbers_171: Array[Byte] = Array(3, 9, 5, 7)
	private final val BitsetNumbers_172: Array[Byte] = Array(5, 7, 2, 1, 9)
	private final val BitsetNumbers_173: Array[Byte] = Array(7, 9, 5, 2)
	private final val BitsetNumbers_174: Array[Byte] = Array(9, 7, 5, 1)
	private final val BitsetNumbers_175: Array[Byte] = Array(5, 9, 7)
	private final val BitsetNumbers_176: Array[Byte] = Array(1, 3, 2, 4, 9, 7)
	private final val BitsetNumbers_177: Array[Byte] = Array(2, 4, 7, 9, 3)
	private final val BitsetNumbers_178: Array[Byte] = Array(7, 4, 1, 3, 9)
	private final val BitsetNumbers_179: Array[Byte] = Array(9, 7, 4, 3)
	private final val BitsetNumbers_180: Array[Byte] = Array(2, 1, 4, 9, 7)
	private final val BitsetNumbers_181: Array[Byte] = Array(4, 9, 7, 2)
	private final val BitsetNumbers_182: Array[Byte] = Array(9, 4, 1, 7)
	private final val BitsetNumbers_183: Array[Byte] = Array(7, 9, 4)
	private final val BitsetNumbers_184: Array[Byte] = Array(7, 3, 9, 1, 2)
	private final val BitsetNumbers_185: Array[Byte] = Array(2, 7, 9, 3)
	private final val BitsetNumbers_186: Array[Byte] = Array(1, 3, 9, 7)
	private final val BitsetNumbers_187: Array[Byte] = Array(3, 9, 7)
	private final val BitsetNumbers_188: Array[Byte] = Array(2, 1, 9, 7)
	private final val BitsetNumbers_189: Array[Byte] = Array(2, 7, 9)
	private final val BitsetNumbers_190: Array[Byte] = Array(1, 9, 7)
	private final val BitsetNumbers_191: Array[Byte] = Array(9, 7)
	private final val BitsetNumbers_192: Array[Byte] = Array(6, 9, 3, 5, 4, 1, 2)
	private final val BitsetNumbers_193: Array[Byte] = Array(9, 3, 6, 4, 2, 5)
	private final val BitsetNumbers_194: Array[Byte] = Array(1, 5, 3, 4, 9, 6)
	private final val BitsetNumbers_195: Array[Byte] = Array(5, 3, 9, 4, 6)
	private final val BitsetNumbers_196: Array[Byte] = Array(4, 6, 9, 1, 5, 2)
	private final val BitsetNumbers_197: Array[Byte] = Array(9, 4, 6, 5, 2)
	private final val BitsetNumbers_198: Array[Byte] = Array(9, 1, 5, 4, 6)
	private final val BitsetNumbers_199: Array[Byte] = Array(6, 9, 4, 5)
	private final val BitsetNumbers_200: Array[Byte] = Array(6, 3, 5, 9, 1, 2)
	private final val BitsetNumbers_201: Array[Byte] = Array(3, 5, 2, 9, 6)
	private final val BitsetNumbers_202: Array[Byte] = Array(5, 3, 6, 1, 9)
	private final val BitsetNumbers_203: Array[Byte] = Array(6, 5, 3, 9)
	private final val BitsetNumbers_204: Array[Byte] = Array(9, 2, 1, 6, 5)
	private final val BitsetNumbers_205: Array[Byte] = Array(9, 5, 2, 6)
	private final val BitsetNumbers_206: Array[Byte] = Array(6, 1, 9, 5)
	private final val BitsetNumbers_207: Array[Byte] = Array(5, 6, 9)
	private final val BitsetNumbers_208: Array[Byte] = Array(4, 9, 2, 1, 3, 6)
	private final val BitsetNumbers_209: Array[Byte] = Array(2, 9, 4, 3, 6)
	private final val BitsetNumbers_210: Array[Byte] = Array(4, 6, 3, 1, 9)
	private final val BitsetNumbers_211: Array[Byte] = Array(3, 4, 6, 9)
	private final val BitsetNumbers_212: Array[Byte] = Array(4, 2, 6, 1, 9)
	private final val BitsetNumbers_213: Array[Byte] = Array(2, 6, 4, 9)
	private final val BitsetNumbers_214: Array[Byte] = Array(4, 6, 9, 1)
	private final val BitsetNumbers_215: Array[Byte] = Array(4, 9, 6)
	private final val BitsetNumbers_216: Array[Byte] = Array(6, 1, 9, 2, 3)
	private final val BitsetNumbers_217: Array[Byte] = Array(9, 6, 3, 2)
	private final val BitsetNumbers_218: Array[Byte] = Array(9, 3, 6, 1)
	private final val BitsetNumbers_219: Array[Byte] = Array(9, 6, 3)
	private final val BitsetNumbers_220: Array[Byte] = Array(6, 2, 9, 1)
	private final val BitsetNumbers_221: Array[Byte] = Array(2, 9, 6)
	private final val BitsetNumbers_222: Array[Byte] = Array(1, 6, 9)
	private final val BitsetNumbers_223: Array[Byte] = Array(6, 9)
	private final val BitsetNumbers_224: Array[Byte] = Array(4, 5, 2, 9, 1, 3)
	private final val BitsetNumbers_225: Array[Byte] = Array(4, 9, 3, 5, 2)
	private final val BitsetNumbers_226: Array[Byte] = Array(9, 3, 1, 4, 5)
	private final val BitsetNumbers_227: Array[Byte] = Array(4, 5, 9, 3)
	private final val BitsetNumbers_228: Array[Byte] = Array(2, 1, 4, 5, 9)
	private final val BitsetNumbers_229: Array[Byte] = Array(5, 9, 2, 4)
	private final val BitsetNumbers_230: Array[Byte] = Array(1, 9, 5, 4)
	private final val BitsetNumbers_231: Array[Byte] = Array(5, 9, 4)
	private final val BitsetNumbers_232: Array[Byte] = Array(2, 9, 1, 5, 3)
	private final val BitsetNumbers_233: Array[Byte] = Array(5, 2, 9, 3)
	private final val BitsetNumbers_234: Array[Byte] = Array(1, 3, 9, 5)
	private final val BitsetNumbers_235: Array[Byte] = Array(9, 5, 3)
	private final val BitsetNumbers_236: Array[Byte] = Array(9, 2, 5, 1)
	private final val BitsetNumbers_237: Array[Byte] = Array(9, 2, 5)
	private final val BitsetNumbers_238: Array[Byte] = Array(9, 5, 1)
	private final val BitsetNumbers_239: Array[Byte] = Array(5, 9)
	private final val BitsetNumbers_240: Array[Byte] = Array(3, 9, 4, 1, 2)
	private final val BitsetNumbers_241: Array[Byte] = Array(2, 9, 3, 4)
	private final val BitsetNumbers_242: Array[Byte] = Array(3, 4, 1, 9)
	private final val BitsetNumbers_243: Array[Byte] = Array(3, 9, 4)
	private final val BitsetNumbers_244: Array[Byte] = Array(4, 2, 1, 9)
	private final val BitsetNumbers_245: Array[Byte] = Array(9, 4, 2)
	private final val BitsetNumbers_246: Array[Byte] = Array(1, 9, 4)
	private final val BitsetNumbers_247: Array[Byte] = Array(9, 4)
	private final val BitsetNumbers_248: Array[Byte] = Array(3, 1, 2, 9)
	private final val BitsetNumbers_249: Array[Byte] = Array(2, 9, 3)
	private final val BitsetNumbers_250: Array[Byte] = Array(1, 9, 3)
	private final val BitsetNumbers_251: Array[Byte] = Array(9, 3)
	private final val BitsetNumbers_252: Array[Byte] = Array(2, 9, 1)
	private final val BitsetNumbers_253: Array[Byte] = Array(2, 9)
	private final val BitsetNumbers_254: Array[Byte] = Array(9, 1)
	private final val BitsetNumbers_255: Array[Byte] = Array(9)
	private final val BitsetNumbers_256: Array[Byte] = Array(1, 3, 6, 2, 5, 4, 8, 7)
	private final val BitsetNumbers_257: Array[Byte] = Array(5, 4, 8, 3, 6, 2, 7)
	private final val BitsetNumbers_258: Array[Byte] = Array(3, 4, 1, 5, 7, 6, 8)
	private final val BitsetNumbers_259: Array[Byte] = Array(4, 6, 8, 7, 5, 3)
	private final val BitsetNumbers_260: Array[Byte] = Array(7, 8, 2, 4, 1, 6, 5)
	private final val BitsetNumbers_261: Array[Byte] = Array(2, 7, 8, 5, 6, 4)
	private final val BitsetNumbers_262: Array[Byte] = Array(8, 5, 4, 6, 7, 1)
	private final val BitsetNumbers_263: Array[Byte] = Array(6, 5, 7, 4, 8)
	private final val BitsetNumbers_264: Array[Byte] = Array(7, 3, 5, 8, 6, 1, 2)
	private final val BitsetNumbers_265: Array[Byte] = Array(6, 7, 3, 8, 2, 5)
	private final val BitsetNumbers_266: Array[Byte] = Array(6, 1, 5, 3, 7, 8)
	private final val BitsetNumbers_267: Array[Byte] = Array(8, 7, 6, 3, 5)
	private final val BitsetNumbers_268: Array[Byte] = Array(5, 2, 8, 1, 6, 7)
	private final val BitsetNumbers_269: Array[Byte] = Array(5, 8, 7, 6, 2)
	private final val BitsetNumbers_270: Array[Byte] = Array(1, 7, 8, 6, 5)
	private final val BitsetNumbers_271: Array[Byte] = Array(7, 6, 5, 8)
	private final val BitsetNumbers_272: Array[Byte] = Array(7, 1, 3, 4, 6, 2, 8)
	private final val BitsetNumbers_273: Array[Byte] = Array(7, 3, 6, 8, 4, 2)
	private final val BitsetNumbers_274: Array[Byte] = Array(3, 6, 1, 7, 8, 4)
	private final val BitsetNumbers_275: Array[Byte] = Array(3, 8, 4, 7, 6)
	private final val BitsetNumbers_276: Array[Byte] = Array(7, 8, 6, 2, 1, 4)
	private final val BitsetNumbers_277: Array[Byte] = Array(6, 8, 4, 2, 7)
	private final val BitsetNumbers_278: Array[Byte] = Array(6, 8, 1, 4, 7)
	private final val BitsetNumbers_279: Array[Byte] = Array(7, 6, 4, 8)
	private final val BitsetNumbers_280: Array[Byte] = Array(1, 2, 7, 6, 3, 8)
	private final val BitsetNumbers_281: Array[Byte] = Array(7, 3, 6, 8, 2)
	private final val BitsetNumbers_282: Array[Byte] = Array(6, 1, 8, 7, 3)
	private final val BitsetNumbers_283: Array[Byte] = Array(8, 6, 7, 3)
	private final val BitsetNumbers_284: Array[Byte] = Array(7, 2, 1, 6, 8)
	private final val BitsetNumbers_285: Array[Byte] = Array(2, 8, 6, 7)
	private final val BitsetNumbers_286: Array[Byte] = Array(8, 1, 7, 6)
	private final val BitsetNumbers_287: Array[Byte] = Array(6, 7, 8)
	private final val BitsetNumbers_288: Array[Byte] = Array(8, 7, 2, 4, 3, 1, 5)
	private final val BitsetNumbers_289: Array[Byte] = Array(4, 5, 7, 2, 3, 8)
	private final val BitsetNumbers_290: Array[Byte] = Array(3, 8, 5, 1, 7, 4)
	private final val BitsetNumbers_291: Array[Byte] = Array(4, 3, 8, 5, 7)
	private final val BitsetNumbers_292: Array[Byte] = Array(7, 2, 1, 8, 4, 5)
	private final val BitsetNumbers_293: Array[Byte] = Array(4, 7, 5, 8, 2)
	private final val BitsetNumbers_294: Array[Byte] = Array(5, 1, 7, 4, 8)
	private final val BitsetNumbers_295: Array[Byte] = Array(5, 7, 4, 8)
	private final val BitsetNumbers_296: Array[Byte] = Array(7, 3, 5, 1, 2, 8)
	private final val BitsetNumbers_297: Array[Byte] = Array(7, 8, 2, 3, 5)
	private final val BitsetNumbers_298: Array[Byte] = Array(7, 1, 3, 8, 5)
	private final val BitsetNumbers_299: Array[Byte] = Array(3, 7, 8, 5)
	private final val BitsetNumbers_300: Array[Byte] = Array(2, 1, 8, 7, 5)
	private final val BitsetNumbers_301: Array[Byte] = Array(2, 8, 5, 7)
	private final val BitsetNumbers_302: Array[Byte] = Array(1, 5, 7, 8)
	private final val BitsetNumbers_303: Array[Byte] = Array(5, 8, 7)
	private final val BitsetNumbers_304: Array[Byte] = Array(7, 8, 2, 4, 3, 1)
	private final val BitsetNumbers_305: Array[Byte] = Array(7, 2, 3, 8, 4)
	private final val BitsetNumbers_306: Array[Byte] = Array(8, 7, 4, 1, 3)
	private final val BitsetNumbers_307: Array[Byte] = Array(7, 8, 4, 3)
	private final val BitsetNumbers_308: Array[Byte] = Array(7, 4, 8, 2, 1)
	private final val BitsetNumbers_309: Array[Byte] = Array(4, 7, 2, 8)
	private final val BitsetNumbers_310: Array[Byte] = Array(4, 7, 1, 8)
	private final val BitsetNumbers_311: Array[Byte] = Array(8, 4, 7)
	private final val BitsetNumbers_312: Array[Byte] = Array(7, 1, 2, 3, 8)
	private final val BitsetNumbers_313: Array[Byte] = Array(2, 3, 7, 8)
	private final val BitsetNumbers_314: Array[Byte] = Array(1, 7, 8, 3)
	private final val BitsetNumbers_315: Array[Byte] = Array(8, 3, 7)
	private final val BitsetNumbers_316: Array[Byte] = Array(1, 2, 8, 7)
	private final val BitsetNumbers_317: Array[Byte] = Array(2, 7, 8)
	private final val BitsetNumbers_318: Array[Byte] = Array(7, 1, 8)
	private final val BitsetNumbers_319: Array[Byte] = Array(8, 7)
	private final val BitsetNumbers_320: Array[Byte] = Array(3, 2, 5, 8, 1, 6, 4)
	private final val BitsetNumbers_321: Array[Byte] = Array(8, 3, 2, 6, 4, 5)
	private final val BitsetNumbers_322: Array[Byte] = Array(6, 4, 3, 8, 1, 5)
	private final val BitsetNumbers_323: Array[Byte] = Array(3, 6, 8, 4, 5)
	private final val BitsetNumbers_324: Array[Byte] = Array(6, 5, 4, 2, 8, 1)
	private final val BitsetNumbers_325: Array[Byte] = Array(6, 5, 2, 4, 8)
	private final val BitsetNumbers_326: Array[Byte] = Array(6, 1, 8, 4, 5)
	private final val BitsetNumbers_327: Array[Byte] = Array(5, 8, 4, 6)
	private final val BitsetNumbers_328: Array[Byte] = Array(8, 1, 5, 2, 3, 6)
	private final val BitsetNumbers_329: Array[Byte] = Array(3, 6, 5, 2, 8)
	private final val BitsetNumbers_330: Array[Byte] = Array(6, 5, 8, 1, 3)
	private final val BitsetNumbers_331: Array[Byte] = Array(8, 5, 3, 6)
	private final val BitsetNumbers_332: Array[Byte] = Array(5, 1, 6, 2, 8)
	private final val BitsetNumbers_333: Array[Byte] = Array(6, 5, 8, 2)
	private final val BitsetNumbers_334: Array[Byte] = Array(6, 5, 8, 1)
	private final val BitsetNumbers_335: Array[Byte] = Array(8, 5, 6)
	private final val BitsetNumbers_336: Array[Byte] = Array(1, 6, 3, 8, 4, 2)
	private final val BitsetNumbers_337: Array[Byte] = Array(4, 3, 2, 8, 6)
	private final val BitsetNumbers_338: Array[Byte] = Array(8, 4, 3, 6, 1)
	private final val BitsetNumbers_339: Array[Byte] = Array(4, 8, 3, 6)
	private final val BitsetNumbers_340: Array[Byte] = Array(6, 1, 4, 2, 8)
	private final val BitsetNumbers_341: Array[Byte] = Array(6, 8, 2, 4)
	private final val BitsetNumbers_342: Array[Byte] = Array(4, 6, 8, 1)
	private final val BitsetNumbers_343: Array[Byte] = Array(4, 8, 6)
	private final val BitsetNumbers_344: Array[Byte] = Array(8, 6, 3, 1, 2)
	private final val BitsetNumbers_345: Array[Byte] = Array(3, 8, 2, 6)
	private final val BitsetNumbers_346: Array[Byte] = Array(8, 3, 6, 1)
	private final val BitsetNumbers_347: Array[Byte] = Array(8, 3, 6)
	private final val BitsetNumbers_348: Array[Byte] = Array(8, 6, 2, 1)
	private final val BitsetNumbers_349: Array[Byte] = Array(8, 6, 2)
	private final val BitsetNumbers_350: Array[Byte] = Array(1, 8, 6)
	private final val BitsetNumbers_351: Array[Byte] = Array(6, 8)
	private final val BitsetNumbers_352: Array[Byte] = Array(1, 3, 4, 2, 8, 5)
	private final val BitsetNumbers_353: Array[Byte] = Array(4, 3, 8, 2, 5)
	private final val BitsetNumbers_354: Array[Byte] = Array(4, 5, 1, 3, 8)
	private final val BitsetNumbers_355: Array[Byte] = Array(5, 8, 3, 4)
	private final val BitsetNumbers_356: Array[Byte] = Array(2, 1, 4, 5, 8)
	private final val BitsetNumbers_357: Array[Byte] = Array(4, 5, 2, 8)
	private final val BitsetNumbers_358: Array[Byte] = Array(4, 1, 8, 5)
	private final val BitsetNumbers_359: Array[Byte] = Array(5, 4, 8)
	private final val BitsetNumbers_360: Array[Byte] = Array(3, 1, 5, 8, 2)
	private final val BitsetNumbers_361: Array[Byte] = Array(8, 2, 3, 5)
	private final val BitsetNumbers_362: Array[Byte] = Array(3, 8, 1, 5)
	private final val BitsetNumbers_363: Array[Byte] = Array(3, 5, 8)
	private final val BitsetNumbers_364: Array[Byte] = Array(2, 8, 1, 5)
	private final val BitsetNumbers_365: Array[Byte] = Array(8, 2, 5)
	private final val BitsetNumbers_366: Array[Byte] = Array(5, 1, 8)
	private final val BitsetNumbers_367: Array[Byte] = Array(5, 8)
	private final val BitsetNumbers_368: Array[Byte] = Array(4, 1, 3, 8, 2)
	private final val BitsetNumbers_369: Array[Byte] = Array(2, 4, 3, 8)
	private final val BitsetNumbers_370: Array[Byte] = Array(1, 4, 3, 8)
	private final val BitsetNumbers_371: Array[Byte] = Array(8, 3, 4)
	private final val BitsetNumbers_372: Array[Byte] = Array(2, 4, 8, 1)
	private final val BitsetNumbers_373: Array[Byte] = Array(2, 8, 4)
	private final val BitsetNumbers_374: Array[Byte] = Array(8, 1, 4)
	private final val BitsetNumbers_375: Array[Byte] = Array(8, 4)
	private final val BitsetNumbers_376: Array[Byte] = Array(3, 8, 2, 1)
	private final val BitsetNumbers_377: Array[Byte] = Array(8, 3, 2)
	private final val BitsetNumbers_378: Array[Byte] = Array(1, 8, 3)
	private final val BitsetNumbers_379: Array[Byte] = Array(8, 3)
	private final val BitsetNumbers_380: Array[Byte] = Array(1, 2, 8)
	private final val BitsetNumbers_381: Array[Byte] = Array(2, 8)
	private final val BitsetNumbers_382: Array[Byte] = Array(1, 8)
	private final val BitsetNumbers_383: Array[Byte] = Array(8)
	private final val BitsetNumbers_384: Array[Byte] = Array(1, 4, 6, 2, 5, 7, 3)
	private final val BitsetNumbers_385: Array[Byte] = Array(7, 3, 6, 4, 5, 2)
	private final val BitsetNumbers_386: Array[Byte] = Array(3, 4, 1, 6, 5, 7)
	private final val BitsetNumbers_387: Array[Byte] = Array(6, 5, 3, 7, 4)
	private final val BitsetNumbers_388: Array[Byte] = Array(7, 2, 1, 4, 6, 5)
	private final val BitsetNumbers_389: Array[Byte] = Array(7, 4, 6, 2, 5)
	private final val BitsetNumbers_390: Array[Byte] = Array(5, 6, 1, 4, 7)
	private final val BitsetNumbers_391: Array[Byte] = Array(7, 4, 6, 5)
	private final val BitsetNumbers_392: Array[Byte] = Array(2, 6, 1, 5, 7, 3)
	private final val BitsetNumbers_393: Array[Byte] = Array(3, 2, 5, 7, 6)
	private final val BitsetNumbers_394: Array[Byte] = Array(3, 5, 6, 1, 7)
	private final val BitsetNumbers_395: Array[Byte] = Array(3, 7, 6, 5)
	private final val BitsetNumbers_396: Array[Byte] = Array(5, 1, 2, 7, 6)
	private final val BitsetNumbers_397: Array[Byte] = Array(2, 6, 7, 5)
	private final val BitsetNumbers_398: Array[Byte] = Array(5, 6, 7, 1)
	private final val BitsetNumbers_399: Array[Byte] = Array(6, 7, 5)
	private final val BitsetNumbers_400: Array[Byte] = Array(6, 4, 7, 2, 3, 1)
	private final val BitsetNumbers_401: Array[Byte] = Array(7, 3, 4, 2, 6)
	private final val BitsetNumbers_402: Array[Byte] = Array(6, 1, 3, 7, 4)
	private final val BitsetNumbers_403: Array[Byte] = Array(3, 4, 6, 7)
	private final val BitsetNumbers_404: Array[Byte] = Array(1, 2, 6, 7, 4)
	private final val BitsetNumbers_405: Array[Byte] = Array(4, 2, 7, 6)
	private final val BitsetNumbers_406: Array[Byte] = Array(4, 7, 6, 1)
	private final val BitsetNumbers_407: Array[Byte] = Array(4, 6, 7)
	private final val BitsetNumbers_408: Array[Byte] = Array(6, 1, 2, 3, 7)
	private final val BitsetNumbers_409: Array[Byte] = Array(7, 6, 3, 2)
	private final val BitsetNumbers_410: Array[Byte] = Array(3, 1, 6, 7)
	private final val BitsetNumbers_411: Array[Byte] = Array(7, 6, 3)
	private final val BitsetNumbers_412: Array[Byte] = Array(1, 6, 7, 2)
	private final val BitsetNumbers_413: Array[Byte] = Array(6, 2, 7)
	private final val BitsetNumbers_414: Array[Byte] = Array(7, 6, 1)
	private final val BitsetNumbers_415: Array[Byte] = Array(7, 6)
	private final val BitsetNumbers_416: Array[Byte] = Array(7, 3, 4, 1, 2, 5)
	private final val BitsetNumbers_417: Array[Byte] = Array(3, 2, 5, 4, 7)
	private final val BitsetNumbers_418: Array[Byte] = Array(1, 5, 4, 3, 7)
	private final val BitsetNumbers_419: Array[Byte] = Array(4, 7, 5, 3)
	private final val BitsetNumbers_420: Array[Byte] = Array(7, 5, 2, 4, 1)
	private final val BitsetNumbers_421: Array[Byte] = Array(7, 2, 5, 4)
	private final val BitsetNumbers_422: Array[Byte] = Array(5, 1, 4, 7)
	private final val BitsetNumbers_423: Array[Byte] = Array(7, 4, 5)
	private final val BitsetNumbers_424: Array[Byte] = Array(3, 2, 5, 1, 7)
	private final val BitsetNumbers_425: Array[Byte] = Array(2, 5, 3, 7)
	private final val BitsetNumbers_426: Array[Byte] = Array(3, 5, 1, 7)
	private final val BitsetNumbers_427: Array[Byte] = Array(7, 5, 3)
	private final val BitsetNumbers_428: Array[Byte] = Array(2, 1, 7, 5)
	private final val BitsetNumbers_429: Array[Byte] = Array(5, 2, 7)
	private final val BitsetNumbers_430: Array[Byte] = Array(1, 7, 5)
	private final val BitsetNumbers_431: Array[Byte] = Array(5, 7)
	private final val BitsetNumbers_432: Array[Byte] = Array(2, 4, 3, 7, 1)
	private final val BitsetNumbers_433: Array[Byte] = Array(7, 2, 4, 3)
	private final val BitsetNumbers_434: Array[Byte] = Array(1, 4, 3, 7)
	private final val BitsetNumbers_435: Array[Byte] = Array(4, 3, 7)
	private final val BitsetNumbers_436: Array[Byte] = Array(7, 1, 4, 2)
	private final val BitsetNumbers_437: Array[Byte] = Array(2, 7, 4)
	private final val BitsetNumbers_438: Array[Byte] = Array(1, 4, 7)
	private final val BitsetNumbers_439: Array[Byte] = Array(4, 7)
	private final val BitsetNumbers_440: Array[Byte] = Array(2, 3, 1, 7)
	private final val BitsetNumbers_441: Array[Byte] = Array(7, 3, 2)
	private final val BitsetNumbers_442: Array[Byte] = Array(1, 7, 3)
	private final val BitsetNumbers_443: Array[Byte] = Array(3, 7)
	private final val BitsetNumbers_444: Array[Byte] = Array(7, 1, 2)
	private final val BitsetNumbers_445: Array[Byte] = Array(7, 2)
	private final val BitsetNumbers_446: Array[Byte] = Array(7, 1)
	private final val BitsetNumbers_447: Array[Byte] = Array(7)
	private final val BitsetNumbers_448: Array[Byte] = Array(1, 4, 6, 5, 3, 2)
	private final val BitsetNumbers_449: Array[Byte] = Array(2, 6, 5, 4, 3)
	private final val BitsetNumbers_450: Array[Byte] = Array(6, 4, 1, 3, 5)
	private final val BitsetNumbers_451: Array[Byte] = Array(6, 5, 4, 3)
	private final val BitsetNumbers_452: Array[Byte] = Array(6, 1, 4, 2, 5)
	private final val BitsetNumbers_453: Array[Byte] = Array(5, 4, 2, 6)
	private final val BitsetNumbers_454: Array[Byte] = Array(1, 4, 5, 6)
	private final val BitsetNumbers_455: Array[Byte] = Array(6, 4, 5)
	private final val BitsetNumbers_456: Array[Byte] = Array(6, 3, 1, 2, 5)
	private final val BitsetNumbers_457: Array[Byte] = Array(6, 5, 2, 3)
	private final val BitsetNumbers_458: Array[Byte] = Array(1, 6, 3, 5)
	private final val BitsetNumbers_459: Array[Byte] = Array(5, 6, 3)
	private final val BitsetNumbers_460: Array[Byte] = Array(6, 1, 2, 5)
	private final val BitsetNumbers_461: Array[Byte] = Array(5, 6, 2)
	private final val BitsetNumbers_462: Array[Byte] = Array(6, 1, 5)
	private final val BitsetNumbers_463: Array[Byte] = Array(5, 6)
	private final val BitsetNumbers_464: Array[Byte] = Array(2, 4, 6, 3, 1)
	private final val BitsetNumbers_465: Array[Byte] = Array(3, 6, 4, 2)
	private final val BitsetNumbers_466: Array[Byte] = Array(6, 3, 1, 4)
	private final val BitsetNumbers_467: Array[Byte] = Array(3, 4, 6)
	private final val BitsetNumbers_468: Array[Byte] = Array(4, 2, 1, 6)
	private final val BitsetNumbers_469: Array[Byte] = Array(4, 2, 6)
	private final val BitsetNumbers_470: Array[Byte] = Array(1, 6, 4)
	private final val BitsetNumbers_471: Array[Byte] = Array(6, 4)
	private final val BitsetNumbers_472: Array[Byte] = Array(1, 6, 2, 3)
	private final val BitsetNumbers_473: Array[Byte] = Array(2, 3, 6)
	private final val BitsetNumbers_474: Array[Byte] = Array(1, 3, 6)
	private final val BitsetNumbers_475: Array[Byte] = Array(3, 6)
	private final val BitsetNumbers_476: Array[Byte] = Array(6, 1, 2)
	private final val BitsetNumbers_477: Array[Byte] = Array(6, 2)
	private final val BitsetNumbers_478: Array[Byte] = Array(1, 6)
	private final val BitsetNumbers_479: Array[Byte] = Array(6)
	private final val BitsetNumbers_480: Array[Byte] = Array(1, 3, 2, 4, 5)
	private final val BitsetNumbers_481: Array[Byte] = Array(4, 2, 3, 5)
	private final val BitsetNumbers_482: Array[Byte] = Array(1, 3, 4, 5)
	private final val BitsetNumbers_483: Array[Byte] = Array(4, 3, 5)
	private final val BitsetNumbers_484: Array[Byte] = Array(5, 1, 4, 2)
	private final val BitsetNumbers_485: Array[Byte] = Array(4, 2, 5)
	private final val BitsetNumbers_486: Array[Byte] = Array(5, 1, 4)
	private final val BitsetNumbers_487: Array[Byte] = Array(4, 5)
	private final val BitsetNumbers_488: Array[Byte] = Array(5, 3, 1, 2)
	private final val BitsetNumbers_489: Array[Byte] = Array(2, 5, 3)
	private final val BitsetNumbers_490: Array[Byte] = Array(3, 5, 1)
	private final val BitsetNumbers_491: Array[Byte] = Array(3, 5)
	private final val BitsetNumbers_492: Array[Byte] = Array(5, 2, 1)
	private final val BitsetNumbers_493: Array[Byte] = Array(2, 5)
	private final val BitsetNumbers_494: Array[Byte] = Array(1, 5)
	private final val BitsetNumbers_495: Array[Byte] = Array(5)
	private final val BitsetNumbers_496: Array[Byte] = Array(4, 2, 3, 1)
	private final val BitsetNumbers_497: Array[Byte] = Array(4, 3, 2)
	private final val BitsetNumbers_498: Array[Byte] = Array(1, 4, 3)
	private final val BitsetNumbers_499: Array[Byte] = Array(4, 3)
	private final val BitsetNumbers_500: Array[Byte] = Array(1, 4, 2)
	private final val BitsetNumbers_501: Array[Byte] = Array(2, 4)
	private final val BitsetNumbers_502: Array[Byte] = Array(1, 4)
	private final val BitsetNumbers_503: Array[Byte] = Array(4)
	private final val BitsetNumbers_504: Array[Byte] = Array(3, 1, 2)
	private final val BitsetNumbers_505: Array[Byte] = Array(3, 2)
	private final val BitsetNumbers_506: Array[Byte] = Array(1, 3)
	private final val BitsetNumbers_507: Array[Byte] = Array(3)
	private final val BitsetNumbers_508: Array[Byte] = Array(2, 1)
	private final val BitsetNumbers_509: Array[Byte] = Array(2)
	private final val BitsetNumbers_510: Array[Byte] = Array(1)
	private final val BitsetNumbers_511: Array[Byte] = Array()

	/**
	 * For each bitset combination there is an array pointing
	 * to the numbers set in the bitset
	 */
	final val BitsetArray: Array[Array[Byte]] = Array(
		BitsetNumbers_000,
		BitsetNumbers_001,
		BitsetNumbers_002,
		BitsetNumbers_003,
		BitsetNumbers_004,
		BitsetNumbers_005,
		BitsetNumbers_006,
		BitsetNumbers_007,
		BitsetNumbers_008,
		BitsetNumbers_009,
		BitsetNumbers_010,
		BitsetNumbers_011,
		BitsetNumbers_012,
		BitsetNumbers_013,
		BitsetNumbers_014,
		BitsetNumbers_015,
		BitsetNumbers_016,
		BitsetNumbers_017,
		BitsetNumbers_018,
		BitsetNumbers_019,
		BitsetNumbers_020,
		BitsetNumbers_021,
		BitsetNumbers_022,
		BitsetNumbers_023,
		BitsetNumbers_024,
		BitsetNumbers_025,
		BitsetNumbers_026,
		BitsetNumbers_027,
		BitsetNumbers_028,
		BitsetNumbers_029,
		BitsetNumbers_030,
		BitsetNumbers_031,
		BitsetNumbers_032,
		BitsetNumbers_033,
		BitsetNumbers_034,
		BitsetNumbers_035,
		BitsetNumbers_036,
		BitsetNumbers_037,
		BitsetNumbers_038,
		BitsetNumbers_039,
		BitsetNumbers_040,
		BitsetNumbers_041,
		BitsetNumbers_042,
		BitsetNumbers_043,
		BitsetNumbers_044,
		BitsetNumbers_045,
		BitsetNumbers_046,
		BitsetNumbers_047,
		BitsetNumbers_048,
		BitsetNumbers_049,
		BitsetNumbers_050,
		BitsetNumbers_051,
		BitsetNumbers_052,
		BitsetNumbers_053,
		BitsetNumbers_054,
		BitsetNumbers_055,
		BitsetNumbers_056,
		BitsetNumbers_057,
		BitsetNumbers_058,
		BitsetNumbers_059,
		BitsetNumbers_060,
		BitsetNumbers_061,
		BitsetNumbers_062,
		BitsetNumbers_063,
		BitsetNumbers_064,
		BitsetNumbers_065,
		BitsetNumbers_066,
		BitsetNumbers_067,
		BitsetNumbers_068,
		BitsetNumbers_069,
		BitsetNumbers_070,
		BitsetNumbers_071,
		BitsetNumbers_072,
		BitsetNumbers_073,
		BitsetNumbers_074,
		BitsetNumbers_075,
		BitsetNumbers_076,
		BitsetNumbers_077,
		BitsetNumbers_078,
		BitsetNumbers_079,
		BitsetNumbers_080,
		BitsetNumbers_081,
		BitsetNumbers_082,
		BitsetNumbers_083,
		BitsetNumbers_084,
		BitsetNumbers_085,
		BitsetNumbers_086,
		BitsetNumbers_087,
		BitsetNumbers_088,
		BitsetNumbers_089,
		BitsetNumbers_090,
		BitsetNumbers_091,
		BitsetNumbers_092,
		BitsetNumbers_093,
		BitsetNumbers_094,
		BitsetNumbers_095,
		BitsetNumbers_096,
		BitsetNumbers_097,
		BitsetNumbers_098,
		BitsetNumbers_099,
		BitsetNumbers_100,
		BitsetNumbers_101,
		BitsetNumbers_102,
		BitsetNumbers_103,
		BitsetNumbers_104,
		BitsetNumbers_105,
		BitsetNumbers_106,
		BitsetNumbers_107,
		BitsetNumbers_108,
		BitsetNumbers_109,
		BitsetNumbers_110,
		BitsetNumbers_111,
		BitsetNumbers_112,
		BitsetNumbers_113,
		BitsetNumbers_114,
		BitsetNumbers_115,
		BitsetNumbers_116,
		BitsetNumbers_117,
		BitsetNumbers_118,
		BitsetNumbers_119,
		BitsetNumbers_120,
		BitsetNumbers_121,
		BitsetNumbers_122,
		BitsetNumbers_123,
		BitsetNumbers_124,
		BitsetNumbers_125,
		BitsetNumbers_126,
		BitsetNumbers_127,
		BitsetNumbers_128,
		BitsetNumbers_129,
		BitsetNumbers_130,
		BitsetNumbers_131,
		BitsetNumbers_132,
		BitsetNumbers_133,
		BitsetNumbers_134,
		BitsetNumbers_135,
		BitsetNumbers_136,
		BitsetNumbers_137,
		BitsetNumbers_138,
		BitsetNumbers_139,
		BitsetNumbers_140,
		BitsetNumbers_141,
		BitsetNumbers_142,
		BitsetNumbers_143,
		BitsetNumbers_144,
		BitsetNumbers_145,
		BitsetNumbers_146,
		BitsetNumbers_147,
		BitsetNumbers_148,
		BitsetNumbers_149,
		BitsetNumbers_150,
		BitsetNumbers_151,
		BitsetNumbers_152,
		BitsetNumbers_153,
		BitsetNumbers_154,
		BitsetNumbers_155,
		BitsetNumbers_156,
		BitsetNumbers_157,
		BitsetNumbers_158,
		BitsetNumbers_159,
		BitsetNumbers_160,
		BitsetNumbers_161,
		BitsetNumbers_162,
		BitsetNumbers_163,
		BitsetNumbers_164,
		BitsetNumbers_165,
		BitsetNumbers_166,
		BitsetNumbers_167,
		BitsetNumbers_168,
		BitsetNumbers_169,
		BitsetNumbers_170,
		BitsetNumbers_171,
		BitsetNumbers_172,
		BitsetNumbers_173,
		BitsetNumbers_174,
		BitsetNumbers_175,
		BitsetNumbers_176,
		BitsetNumbers_177,
		BitsetNumbers_178,
		BitsetNumbers_179,
		BitsetNumbers_180,
		BitsetNumbers_181,
		BitsetNumbers_182,
		BitsetNumbers_183,
		BitsetNumbers_184,
		BitsetNumbers_185,
		BitsetNumbers_186,
		BitsetNumbers_187,
		BitsetNumbers_188,
		BitsetNumbers_189,
		BitsetNumbers_190,
		BitsetNumbers_191,
		BitsetNumbers_192,
		BitsetNumbers_193,
		BitsetNumbers_194,
		BitsetNumbers_195,
		BitsetNumbers_196,
		BitsetNumbers_197,
		BitsetNumbers_198,
		BitsetNumbers_199,
		BitsetNumbers_200,
		BitsetNumbers_201,
		BitsetNumbers_202,
		BitsetNumbers_203,
		BitsetNumbers_204,
		BitsetNumbers_205,
		BitsetNumbers_206,
		BitsetNumbers_207,
		BitsetNumbers_208,
		BitsetNumbers_209,
		BitsetNumbers_210,
		BitsetNumbers_211,
		BitsetNumbers_212,
		BitsetNumbers_213,
		BitsetNumbers_214,
		BitsetNumbers_215,
		BitsetNumbers_216,
		BitsetNumbers_217,
		BitsetNumbers_218,
		BitsetNumbers_219,
		BitsetNumbers_220,
		BitsetNumbers_221,
		BitsetNumbers_222,
		BitsetNumbers_223,
		BitsetNumbers_224,
		BitsetNumbers_225,
		BitsetNumbers_226,
		BitsetNumbers_227,
		BitsetNumbers_228,
		BitsetNumbers_229,
		BitsetNumbers_230,
		BitsetNumbers_231,
		BitsetNumbers_232,
		BitsetNumbers_233,
		BitsetNumbers_234,
		BitsetNumbers_235,
		BitsetNumbers_236,
		BitsetNumbers_237,
		BitsetNumbers_238,
		BitsetNumbers_239,
		BitsetNumbers_240,
		BitsetNumbers_241,
		BitsetNumbers_242,
		BitsetNumbers_243,
		BitsetNumbers_244,
		BitsetNumbers_245,
		BitsetNumbers_246,
		BitsetNumbers_247,
		BitsetNumbers_248,
		BitsetNumbers_249,
		BitsetNumbers_250,
		BitsetNumbers_251,
		BitsetNumbers_252,
		BitsetNumbers_253,
		BitsetNumbers_254,
		BitsetNumbers_255,
		BitsetNumbers_256,
		BitsetNumbers_257,
		BitsetNumbers_258,
		BitsetNumbers_259,
		BitsetNumbers_260,
		BitsetNumbers_261,
		BitsetNumbers_262,
		BitsetNumbers_263,
		BitsetNumbers_264,
		BitsetNumbers_265,
		BitsetNumbers_266,
		BitsetNumbers_267,
		BitsetNumbers_268,
		BitsetNumbers_269,
		BitsetNumbers_270,
		BitsetNumbers_271,
		BitsetNumbers_272,
		BitsetNumbers_273,
		BitsetNumbers_274,
		BitsetNumbers_275,
		BitsetNumbers_276,
		BitsetNumbers_277,
		BitsetNumbers_278,
		BitsetNumbers_279,
		BitsetNumbers_280,
		BitsetNumbers_281,
		BitsetNumbers_282,
		BitsetNumbers_283,
		BitsetNumbers_284,
		BitsetNumbers_285,
		BitsetNumbers_286,
		BitsetNumbers_287,
		BitsetNumbers_288,
		BitsetNumbers_289,
		BitsetNumbers_290,
		BitsetNumbers_291,
		BitsetNumbers_292,
		BitsetNumbers_293,
		BitsetNumbers_294,
		BitsetNumbers_295,
		BitsetNumbers_296,
		BitsetNumbers_297,
		BitsetNumbers_298,
		BitsetNumbers_299,
		BitsetNumbers_300,
		BitsetNumbers_301,
		BitsetNumbers_302,
		BitsetNumbers_303,
		BitsetNumbers_304,
		BitsetNumbers_305,
		BitsetNumbers_306,
		BitsetNumbers_307,
		BitsetNumbers_308,
		BitsetNumbers_309,
		BitsetNumbers_310,
		BitsetNumbers_311,
		BitsetNumbers_312,
		BitsetNumbers_313,
		BitsetNumbers_314,
		BitsetNumbers_315,
		BitsetNumbers_316,
		BitsetNumbers_317,
		BitsetNumbers_318,
		BitsetNumbers_319,
		BitsetNumbers_320,
		BitsetNumbers_321,
		BitsetNumbers_322,
		BitsetNumbers_323,
		BitsetNumbers_324,
		BitsetNumbers_325,
		BitsetNumbers_326,
		BitsetNumbers_327,
		BitsetNumbers_328,
		BitsetNumbers_329,
		BitsetNumbers_330,
		BitsetNumbers_331,
		BitsetNumbers_332,
		BitsetNumbers_333,
		BitsetNumbers_334,
		BitsetNumbers_335,
		BitsetNumbers_336,
		BitsetNumbers_337,
		BitsetNumbers_338,
		BitsetNumbers_339,
		BitsetNumbers_340,
		BitsetNumbers_341,
		BitsetNumbers_342,
		BitsetNumbers_343,
		BitsetNumbers_344,
		BitsetNumbers_345,
		BitsetNumbers_346,
		BitsetNumbers_347,
		BitsetNumbers_348,
		BitsetNumbers_349,
		BitsetNumbers_350,
		BitsetNumbers_351,
		BitsetNumbers_352,
		BitsetNumbers_353,
		BitsetNumbers_354,
		BitsetNumbers_355,
		BitsetNumbers_356,
		BitsetNumbers_357,
		BitsetNumbers_358,
		BitsetNumbers_359,
		BitsetNumbers_360,
		BitsetNumbers_361,
		BitsetNumbers_362,
		BitsetNumbers_363,
		BitsetNumbers_364,
		BitsetNumbers_365,
		BitsetNumbers_366,
		BitsetNumbers_367,
		BitsetNumbers_368,
		BitsetNumbers_369,
		BitsetNumbers_370,
		BitsetNumbers_371,
		BitsetNumbers_372,
		BitsetNumbers_373,
		BitsetNumbers_374,
		BitsetNumbers_375,
		BitsetNumbers_376,
		BitsetNumbers_377,
		BitsetNumbers_378,
		BitsetNumbers_379,
		BitsetNumbers_380,
		BitsetNumbers_381,
		BitsetNumbers_382,
		BitsetNumbers_383,
		BitsetNumbers_384,
		BitsetNumbers_385,
		BitsetNumbers_386,
		BitsetNumbers_387,
		BitsetNumbers_388,
		BitsetNumbers_389,
		BitsetNumbers_390,
		BitsetNumbers_391,
		BitsetNumbers_392,
		BitsetNumbers_393,
		BitsetNumbers_394,
		BitsetNumbers_395,
		BitsetNumbers_396,
		BitsetNumbers_397,
		BitsetNumbers_398,
		BitsetNumbers_399,
		BitsetNumbers_400,
		BitsetNumbers_401,
		BitsetNumbers_402,
		BitsetNumbers_403,
		BitsetNumbers_404,
		BitsetNumbers_405,
		BitsetNumbers_406,
		BitsetNumbers_407,
		BitsetNumbers_408,
		BitsetNumbers_409,
		BitsetNumbers_410,
		BitsetNumbers_411,
		BitsetNumbers_412,
		BitsetNumbers_413,
		BitsetNumbers_414,
		BitsetNumbers_415,
		BitsetNumbers_416,
		BitsetNumbers_417,
		BitsetNumbers_418,
		BitsetNumbers_419,
		BitsetNumbers_420,
		BitsetNumbers_421,
		BitsetNumbers_422,
		BitsetNumbers_423,
		BitsetNumbers_424,
		BitsetNumbers_425,
		BitsetNumbers_426,
		BitsetNumbers_427,
		BitsetNumbers_428,
		BitsetNumbers_429,
		BitsetNumbers_430,
		BitsetNumbers_431,
		BitsetNumbers_432,
		BitsetNumbers_433,
		BitsetNumbers_434,
		BitsetNumbers_435,
		BitsetNumbers_436,
		BitsetNumbers_437,
		BitsetNumbers_438,
		BitsetNumbers_439,
		BitsetNumbers_440,
		BitsetNumbers_441,
		BitsetNumbers_442,
		BitsetNumbers_443,
		BitsetNumbers_444,
		BitsetNumbers_445,
		BitsetNumbers_446,
		BitsetNumbers_447,
		BitsetNumbers_448,
		BitsetNumbers_449,
		BitsetNumbers_450,
		BitsetNumbers_451,
		BitsetNumbers_452,
		BitsetNumbers_453,
		BitsetNumbers_454,
		BitsetNumbers_455,
		BitsetNumbers_456,
		BitsetNumbers_457,
		BitsetNumbers_458,
		BitsetNumbers_459,
		BitsetNumbers_460,
		BitsetNumbers_461,
		BitsetNumbers_462,
		BitsetNumbers_463,
		BitsetNumbers_464,
		BitsetNumbers_465,
		BitsetNumbers_466,
		BitsetNumbers_467,
		BitsetNumbers_468,
		BitsetNumbers_469,
		BitsetNumbers_470,
		BitsetNumbers_471,
		BitsetNumbers_472,
		BitsetNumbers_473,
		BitsetNumbers_474,
		BitsetNumbers_475,
		BitsetNumbers_476,
		BitsetNumbers_477,
		BitsetNumbers_478,
		BitsetNumbers_479,
		BitsetNumbers_480,
		BitsetNumbers_481,
		BitsetNumbers_482,
		BitsetNumbers_483,
		BitsetNumbers_484,
		BitsetNumbers_485,
		BitsetNumbers_486,
		BitsetNumbers_487,
		BitsetNumbers_488,
		BitsetNumbers_489,
		BitsetNumbers_490,
		BitsetNumbers_491,
		BitsetNumbers_492,
		BitsetNumbers_493,
		BitsetNumbers_494,
		BitsetNumbers_495,
		BitsetNumbers_496,
		BitsetNumbers_497,
		BitsetNumbers_498,
		BitsetNumbers_499,
		BitsetNumbers_500,
		BitsetNumbers_501,
		BitsetNumbers_502,
		BitsetNumbers_503,
		BitsetNumbers_504,
		BitsetNumbers_505,
		BitsetNumbers_506,
		BitsetNumbers_507,
		BitsetNumbers_508,
		BitsetNumbers_509,
		BitsetNumbers_510,
		BitsetNumbers_511
	)

}
