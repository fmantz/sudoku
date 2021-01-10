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
	final val ParallelizationCount: Int = 1024

	private final val BitsetNumbers_000: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_001: Array[Int] = Array(2, 3, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_002: Array[Int] = Array(1, 3, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_003: Array[Int] = Array(3, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_004: Array[Int] = Array(1, 2, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_005: Array[Int] = Array(2, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_006: Array[Int] = Array(1, 4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_007: Array[Int] = Array(4, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_008: Array[Int] = Array(1, 2, 3, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_009: Array[Int] = Array(2, 3, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_010: Array[Int] = Array(1, 3, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_011: Array[Int] = Array(3, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_012: Array[Int] = Array(1, 2, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_013: Array[Int] = Array(2, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_014: Array[Int] = Array(1, 5, 6, 7, 8, 9)
	private final val BitsetNumbers_015: Array[Int] = Array(5, 6, 7, 8, 9)
	private final val BitsetNumbers_016: Array[Int] = Array(1, 2, 3, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_017: Array[Int] = Array(2, 3, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_018: Array[Int] = Array(1, 3, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_019: Array[Int] = Array(3, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_020: Array[Int] = Array(1, 2, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_021: Array[Int] = Array(2, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_022: Array[Int] = Array(1, 4, 6, 7, 8, 9)
	private final val BitsetNumbers_023: Array[Int] = Array(4, 6, 7, 8, 9)
	private final val BitsetNumbers_024: Array[Int] = Array(1, 2, 3, 6, 7, 8, 9)
	private final val BitsetNumbers_025: Array[Int] = Array(2, 3, 6, 7, 8, 9)
	private final val BitsetNumbers_026: Array[Int] = Array(1, 3, 6, 7, 8, 9)
	private final val BitsetNumbers_027: Array[Int] = Array(3, 6, 7, 8, 9)
	private final val BitsetNumbers_028: Array[Int] = Array(1, 2, 6, 7, 8, 9)
	private final val BitsetNumbers_029: Array[Int] = Array(2, 6, 7, 8, 9)
	private final val BitsetNumbers_030: Array[Int] = Array(1, 6, 7, 8, 9)
	private final val BitsetNumbers_031: Array[Int] = Array(6, 7, 8, 9)
	private final val BitsetNumbers_032: Array[Int] = Array(1, 2, 3, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_033: Array[Int] = Array(2, 3, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_034: Array[Int] = Array(1, 3, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_035: Array[Int] = Array(3, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_036: Array[Int] = Array(1, 2, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_037: Array[Int] = Array(2, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_038: Array[Int] = Array(1, 4, 5, 7, 8, 9)
	private final val BitsetNumbers_039: Array[Int] = Array(4, 5, 7, 8, 9)
	private final val BitsetNumbers_040: Array[Int] = Array(1, 2, 3, 5, 7, 8, 9)
	private final val BitsetNumbers_041: Array[Int] = Array(2, 3, 5, 7, 8, 9)
	private final val BitsetNumbers_042: Array[Int] = Array(1, 3, 5, 7, 8, 9)
	private final val BitsetNumbers_043: Array[Int] = Array(3, 5, 7, 8, 9)
	private final val BitsetNumbers_044: Array[Int] = Array(1, 2, 5, 7, 8, 9)
	private final val BitsetNumbers_045: Array[Int] = Array(2, 5, 7, 8, 9)
	private final val BitsetNumbers_046: Array[Int] = Array(1, 5, 7, 8, 9)
	private final val BitsetNumbers_047: Array[Int] = Array(5, 7, 8, 9)
	private final val BitsetNumbers_048: Array[Int] = Array(1, 2, 3, 4, 7, 8, 9)
	private final val BitsetNumbers_049: Array[Int] = Array(2, 3, 4, 7, 8, 9)
	private final val BitsetNumbers_050: Array[Int] = Array(1, 3, 4, 7, 8, 9)
	private final val BitsetNumbers_051: Array[Int] = Array(3, 4, 7, 8, 9)
	private final val BitsetNumbers_052: Array[Int] = Array(1, 2, 4, 7, 8, 9)
	private final val BitsetNumbers_053: Array[Int] = Array(2, 4, 7, 8, 9)
	private final val BitsetNumbers_054: Array[Int] = Array(1, 4, 7, 8, 9)
	private final val BitsetNumbers_055: Array[Int] = Array(4, 7, 8, 9)
	private final val BitsetNumbers_056: Array[Int] = Array(1, 2, 3, 7, 8, 9)
	private final val BitsetNumbers_057: Array[Int] = Array(2, 3, 7, 8, 9)
	private final val BitsetNumbers_058: Array[Int] = Array(1, 3, 7, 8, 9)
	private final val BitsetNumbers_059: Array[Int] = Array(3, 7, 8, 9)
	private final val BitsetNumbers_060: Array[Int] = Array(1, 2, 7, 8, 9)
	private final val BitsetNumbers_061: Array[Int] = Array(2, 7, 8, 9)
	private final val BitsetNumbers_062: Array[Int] = Array(1, 7, 8, 9)
	private final val BitsetNumbers_063: Array[Int] = Array(7, 8, 9)
	private final val BitsetNumbers_064: Array[Int] = Array(1, 2, 3, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_065: Array[Int] = Array(2, 3, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_066: Array[Int] = Array(1, 3, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_067: Array[Int] = Array(3, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_068: Array[Int] = Array(1, 2, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_069: Array[Int] = Array(2, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_070: Array[Int] = Array(1, 4, 5, 6, 8, 9)
	private final val BitsetNumbers_071: Array[Int] = Array(4, 5, 6, 8, 9)
	private final val BitsetNumbers_072: Array[Int] = Array(1, 2, 3, 5, 6, 8, 9)
	private final val BitsetNumbers_073: Array[Int] = Array(2, 3, 5, 6, 8, 9)
	private final val BitsetNumbers_074: Array[Int] = Array(1, 3, 5, 6, 8, 9)
	private final val BitsetNumbers_075: Array[Int] = Array(3, 5, 6, 8, 9)
	private final val BitsetNumbers_076: Array[Int] = Array(1, 2, 5, 6, 8, 9)
	private final val BitsetNumbers_077: Array[Int] = Array(2, 5, 6, 8, 9)
	private final val BitsetNumbers_078: Array[Int] = Array(1, 5, 6, 8, 9)
	private final val BitsetNumbers_079: Array[Int] = Array(5, 6, 8, 9)
	private final val BitsetNumbers_080: Array[Int] = Array(1, 2, 3, 4, 6, 8, 9)
	private final val BitsetNumbers_081: Array[Int] = Array(2, 3, 4, 6, 8, 9)
	private final val BitsetNumbers_082: Array[Int] = Array(1, 3, 4, 6, 8, 9)
	private final val BitsetNumbers_083: Array[Int] = Array(3, 4, 6, 8, 9)
	private final val BitsetNumbers_084: Array[Int] = Array(1, 2, 4, 6, 8, 9)
	private final val BitsetNumbers_085: Array[Int] = Array(2, 4, 6, 8, 9)
	private final val BitsetNumbers_086: Array[Int] = Array(1, 4, 6, 8, 9)
	private final val BitsetNumbers_087: Array[Int] = Array(4, 6, 8, 9)
	private final val BitsetNumbers_088: Array[Int] = Array(1, 2, 3, 6, 8, 9)
	private final val BitsetNumbers_089: Array[Int] = Array(2, 3, 6, 8, 9)
	private final val BitsetNumbers_090: Array[Int] = Array(1, 3, 6, 8, 9)
	private final val BitsetNumbers_091: Array[Int] = Array(3, 6, 8, 9)
	private final val BitsetNumbers_092: Array[Int] = Array(1, 2, 6, 8, 9)
	private final val BitsetNumbers_093: Array[Int] = Array(2, 6, 8, 9)
	private final val BitsetNumbers_094: Array[Int] = Array(1, 6, 8, 9)
	private final val BitsetNumbers_095: Array[Int] = Array(6, 8, 9)
	private final val BitsetNumbers_096: Array[Int] = Array(1, 2, 3, 4, 5, 8, 9)
	private final val BitsetNumbers_097: Array[Int] = Array(2, 3, 4, 5, 8, 9)
	private final val BitsetNumbers_098: Array[Int] = Array(1, 3, 4, 5, 8, 9)
	private final val BitsetNumbers_099: Array[Int] = Array(3, 4, 5, 8, 9)
	private final val BitsetNumbers_100: Array[Int] = Array(1, 2, 4, 5, 8, 9)
	private final val BitsetNumbers_101: Array[Int] = Array(2, 4, 5, 8, 9)
	private final val BitsetNumbers_102: Array[Int] = Array(1, 4, 5, 8, 9)
	private final val BitsetNumbers_103: Array[Int] = Array(4, 5, 8, 9)
	private final val BitsetNumbers_104: Array[Int] = Array(1, 2, 3, 5, 8, 9)
	private final val BitsetNumbers_105: Array[Int] = Array(2, 3, 5, 8, 9)
	private final val BitsetNumbers_106: Array[Int] = Array(1, 3, 5, 8, 9)
	private final val BitsetNumbers_107: Array[Int] = Array(3, 5, 8, 9)
	private final val BitsetNumbers_108: Array[Int] = Array(1, 2, 5, 8, 9)
	private final val BitsetNumbers_109: Array[Int] = Array(2, 5, 8, 9)
	private final val BitsetNumbers_110: Array[Int] = Array(1, 5, 8, 9)
	private final val BitsetNumbers_111: Array[Int] = Array(5, 8, 9)
	private final val BitsetNumbers_112: Array[Int] = Array(1, 2, 3, 4, 8, 9)
	private final val BitsetNumbers_113: Array[Int] = Array(2, 3, 4, 8, 9)
	private final val BitsetNumbers_114: Array[Int] = Array(1, 3, 4, 8, 9)
	private final val BitsetNumbers_115: Array[Int] = Array(3, 4, 8, 9)
	private final val BitsetNumbers_116: Array[Int] = Array(1, 2, 4, 8, 9)
	private final val BitsetNumbers_117: Array[Int] = Array(2, 4, 8, 9)
	private final val BitsetNumbers_118: Array[Int] = Array(1, 4, 8, 9)
	private final val BitsetNumbers_119: Array[Int] = Array(4, 8, 9)
	private final val BitsetNumbers_120: Array[Int] = Array(1, 2, 3, 8, 9)
	private final val BitsetNumbers_121: Array[Int] = Array(2, 3, 8, 9)
	private final val BitsetNumbers_122: Array[Int] = Array(1, 3, 8, 9)
	private final val BitsetNumbers_123: Array[Int] = Array(3, 8, 9)
	private final val BitsetNumbers_124: Array[Int] = Array(1, 2, 8, 9)
	private final val BitsetNumbers_125: Array[Int] = Array(2, 8, 9)
	private final val BitsetNumbers_126: Array[Int] = Array(1, 8, 9)
	private final val BitsetNumbers_127: Array[Int] = Array(8, 9)
	private final val BitsetNumbers_128: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_129: Array[Int] = Array(2, 3, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_130: Array[Int] = Array(1, 3, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_131: Array[Int] = Array(3, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_132: Array[Int] = Array(1, 2, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_133: Array[Int] = Array(2, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_134: Array[Int] = Array(1, 4, 5, 6, 7, 9)
	private final val BitsetNumbers_135: Array[Int] = Array(4, 5, 6, 7, 9)
	private final val BitsetNumbers_136: Array[Int] = Array(1, 2, 3, 5, 6, 7, 9)
	private final val BitsetNumbers_137: Array[Int] = Array(2, 3, 5, 6, 7, 9)
	private final val BitsetNumbers_138: Array[Int] = Array(1, 3, 5, 6, 7, 9)
	private final val BitsetNumbers_139: Array[Int] = Array(3, 5, 6, 7, 9)
	private final val BitsetNumbers_140: Array[Int] = Array(1, 2, 5, 6, 7, 9)
	private final val BitsetNumbers_141: Array[Int] = Array(2, 5, 6, 7, 9)
	private final val BitsetNumbers_142: Array[Int] = Array(1, 5, 6, 7, 9)
	private final val BitsetNumbers_143: Array[Int] = Array(5, 6, 7, 9)
	private final val BitsetNumbers_144: Array[Int] = Array(1, 2, 3, 4, 6, 7, 9)
	private final val BitsetNumbers_145: Array[Int] = Array(2, 3, 4, 6, 7, 9)
	private final val BitsetNumbers_146: Array[Int] = Array(1, 3, 4, 6, 7, 9)
	private final val BitsetNumbers_147: Array[Int] = Array(3, 4, 6, 7, 9)
	private final val BitsetNumbers_148: Array[Int] = Array(1, 2, 4, 6, 7, 9)
	private final val BitsetNumbers_149: Array[Int] = Array(2, 4, 6, 7, 9)
	private final val BitsetNumbers_150: Array[Int] = Array(1, 4, 6, 7, 9)
	private final val BitsetNumbers_151: Array[Int] = Array(4, 6, 7, 9)
	private final val BitsetNumbers_152: Array[Int] = Array(1, 2, 3, 6, 7, 9)
	private final val BitsetNumbers_153: Array[Int] = Array(2, 3, 6, 7, 9)
	private final val BitsetNumbers_154: Array[Int] = Array(1, 3, 6, 7, 9)
	private final val BitsetNumbers_155: Array[Int] = Array(3, 6, 7, 9)
	private final val BitsetNumbers_156: Array[Int] = Array(1, 2, 6, 7, 9)
	private final val BitsetNumbers_157: Array[Int] = Array(2, 6, 7, 9)
	private final val BitsetNumbers_158: Array[Int] = Array(1, 6, 7, 9)
	private final val BitsetNumbers_159: Array[Int] = Array(6, 7, 9)
	private final val BitsetNumbers_160: Array[Int] = Array(1, 2, 3, 4, 5, 7, 9)
	private final val BitsetNumbers_161: Array[Int] = Array(2, 3, 4, 5, 7, 9)
	private final val BitsetNumbers_162: Array[Int] = Array(1, 3, 4, 5, 7, 9)
	private final val BitsetNumbers_163: Array[Int] = Array(3, 4, 5, 7, 9)
	private final val BitsetNumbers_164: Array[Int] = Array(1, 2, 4, 5, 7, 9)
	private final val BitsetNumbers_165: Array[Int] = Array(2, 4, 5, 7, 9)
	private final val BitsetNumbers_166: Array[Int] = Array(1, 4, 5, 7, 9)
	private final val BitsetNumbers_167: Array[Int] = Array(4, 5, 7, 9)
	private final val BitsetNumbers_168: Array[Int] = Array(1, 2, 3, 5, 7, 9)
	private final val BitsetNumbers_169: Array[Int] = Array(2, 3, 5, 7, 9)
	private final val BitsetNumbers_170: Array[Int] = Array(1, 3, 5, 7, 9)
	private final val BitsetNumbers_171: Array[Int] = Array(3, 5, 7, 9)
	private final val BitsetNumbers_172: Array[Int] = Array(1, 2, 5, 7, 9)
	private final val BitsetNumbers_173: Array[Int] = Array(2, 5, 7, 9)
	private final val BitsetNumbers_174: Array[Int] = Array(1, 5, 7, 9)
	private final val BitsetNumbers_175: Array[Int] = Array(5, 7, 9)
	private final val BitsetNumbers_176: Array[Int] = Array(1, 2, 3, 4, 7, 9)
	private final val BitsetNumbers_177: Array[Int] = Array(2, 3, 4, 7, 9)
	private final val BitsetNumbers_178: Array[Int] = Array(1, 3, 4, 7, 9)
	private final val BitsetNumbers_179: Array[Int] = Array(3, 4, 7, 9)
	private final val BitsetNumbers_180: Array[Int] = Array(1, 2, 4, 7, 9)
	private final val BitsetNumbers_181: Array[Int] = Array(2, 4, 7, 9)
	private final val BitsetNumbers_182: Array[Int] = Array(1, 4, 7, 9)
	private final val BitsetNumbers_183: Array[Int] = Array(4, 7, 9)
	private final val BitsetNumbers_184: Array[Int] = Array(1, 2, 3, 7, 9)
	private final val BitsetNumbers_185: Array[Int] = Array(2, 3, 7, 9)
	private final val BitsetNumbers_186: Array[Int] = Array(1, 3, 7, 9)
	private final val BitsetNumbers_187: Array[Int] = Array(3, 7, 9)
	private final val BitsetNumbers_188: Array[Int] = Array(1, 2, 7, 9)
	private final val BitsetNumbers_189: Array[Int] = Array(2, 7, 9)
	private final val BitsetNumbers_190: Array[Int] = Array(1, 7, 9)
	private final val BitsetNumbers_191: Array[Int] = Array(7, 9)
	private final val BitsetNumbers_192: Array[Int] = Array(1, 2, 3, 4, 5, 6, 9)
	private final val BitsetNumbers_193: Array[Int] = Array(2, 3, 4, 5, 6, 9)
	private final val BitsetNumbers_194: Array[Int] = Array(1, 3, 4, 5, 6, 9)
	private final val BitsetNumbers_195: Array[Int] = Array(3, 4, 5, 6, 9)
	private final val BitsetNumbers_196: Array[Int] = Array(1, 2, 4, 5, 6, 9)
	private final val BitsetNumbers_197: Array[Int] = Array(2, 4, 5, 6, 9)
	private final val BitsetNumbers_198: Array[Int] = Array(1, 4, 5, 6, 9)
	private final val BitsetNumbers_199: Array[Int] = Array(4, 5, 6, 9)
	private final val BitsetNumbers_200: Array[Int] = Array(1, 2, 3, 5, 6, 9)
	private final val BitsetNumbers_201: Array[Int] = Array(2, 3, 5, 6, 9)
	private final val BitsetNumbers_202: Array[Int] = Array(1, 3, 5, 6, 9)
	private final val BitsetNumbers_203: Array[Int] = Array(3, 5, 6, 9)
	private final val BitsetNumbers_204: Array[Int] = Array(1, 2, 5, 6, 9)
	private final val BitsetNumbers_205: Array[Int] = Array(2, 5, 6, 9)
	private final val BitsetNumbers_206: Array[Int] = Array(1, 5, 6, 9)
	private final val BitsetNumbers_207: Array[Int] = Array(5, 6, 9)
	private final val BitsetNumbers_208: Array[Int] = Array(1, 2, 3, 4, 6, 9)
	private final val BitsetNumbers_209: Array[Int] = Array(2, 3, 4, 6, 9)
	private final val BitsetNumbers_210: Array[Int] = Array(1, 3, 4, 6, 9)
	private final val BitsetNumbers_211: Array[Int] = Array(3, 4, 6, 9)
	private final val BitsetNumbers_212: Array[Int] = Array(1, 2, 4, 6, 9)
	private final val BitsetNumbers_213: Array[Int] = Array(2, 4, 6, 9)
	private final val BitsetNumbers_214: Array[Int] = Array(1, 4, 6, 9)
	private final val BitsetNumbers_215: Array[Int] = Array(4, 6, 9)
	private final val BitsetNumbers_216: Array[Int] = Array(1, 2, 3, 6, 9)
	private final val BitsetNumbers_217: Array[Int] = Array(2, 3, 6, 9)
	private final val BitsetNumbers_218: Array[Int] = Array(1, 3, 6, 9)
	private final val BitsetNumbers_219: Array[Int] = Array(3, 6, 9)
	private final val BitsetNumbers_220: Array[Int] = Array(1, 2, 6, 9)
	private final val BitsetNumbers_221: Array[Int] = Array(2, 6, 9)
	private final val BitsetNumbers_222: Array[Int] = Array(1, 6, 9)
	private final val BitsetNumbers_223: Array[Int] = Array(6, 9)
	private final val BitsetNumbers_224: Array[Int] = Array(1, 2, 3, 4, 5, 9)
	private final val BitsetNumbers_225: Array[Int] = Array(2, 3, 4, 5, 9)
	private final val BitsetNumbers_226: Array[Int] = Array(1, 3, 4, 5, 9)
	private final val BitsetNumbers_227: Array[Int] = Array(3, 4, 5, 9)
	private final val BitsetNumbers_228: Array[Int] = Array(1, 2, 4, 5, 9)
	private final val BitsetNumbers_229: Array[Int] = Array(2, 4, 5, 9)
	private final val BitsetNumbers_230: Array[Int] = Array(1, 4, 5, 9)
	private final val BitsetNumbers_231: Array[Int] = Array(4, 5, 9)
	private final val BitsetNumbers_232: Array[Int] = Array(1, 2, 3, 5, 9)
	private final val BitsetNumbers_233: Array[Int] = Array(2, 3, 5, 9)
	private final val BitsetNumbers_234: Array[Int] = Array(1, 3, 5, 9)
	private final val BitsetNumbers_235: Array[Int] = Array(3, 5, 9)
	private final val BitsetNumbers_236: Array[Int] = Array(1, 2, 5, 9)
	private final val BitsetNumbers_237: Array[Int] = Array(2, 5, 9)
	private final val BitsetNumbers_238: Array[Int] = Array(1, 5, 9)
	private final val BitsetNumbers_239: Array[Int] = Array(5, 9)
	private final val BitsetNumbers_240: Array[Int] = Array(1, 2, 3, 4, 9)
	private final val BitsetNumbers_241: Array[Int] = Array(2, 3, 4, 9)
	private final val BitsetNumbers_242: Array[Int] = Array(1, 3, 4, 9)
	private final val BitsetNumbers_243: Array[Int] = Array(3, 4, 9)
	private final val BitsetNumbers_244: Array[Int] = Array(1, 2, 4, 9)
	private final val BitsetNumbers_245: Array[Int] = Array(2, 4, 9)
	private final val BitsetNumbers_246: Array[Int] = Array(1, 4, 9)
	private final val BitsetNumbers_247: Array[Int] = Array(4, 9)
	private final val BitsetNumbers_248: Array[Int] = Array(1, 2, 3, 9)
	private final val BitsetNumbers_249: Array[Int] = Array(2, 3, 9)
	private final val BitsetNumbers_250: Array[Int] = Array(1, 3, 9)
	private final val BitsetNumbers_251: Array[Int] = Array(3, 9)
	private final val BitsetNumbers_252: Array[Int] = Array(1, 2, 9)
	private final val BitsetNumbers_253: Array[Int] = Array(2, 9)
	private final val BitsetNumbers_254: Array[Int] = Array(1, 9)
	private final val BitsetNumbers_255: Array[Int] = Array(9)
	private final val BitsetNumbers_256: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_257: Array[Int] = Array(2, 3, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_258: Array[Int] = Array(1, 3, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_259: Array[Int] = Array(3, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_260: Array[Int] = Array(1, 2, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_261: Array[Int] = Array(2, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_262: Array[Int] = Array(1, 4, 5, 6, 7, 8)
	private final val BitsetNumbers_263: Array[Int] = Array(4, 5, 6, 7, 8)
	private final val BitsetNumbers_264: Array[Int] = Array(1, 2, 3, 5, 6, 7, 8)
	private final val BitsetNumbers_265: Array[Int] = Array(2, 3, 5, 6, 7, 8)
	private final val BitsetNumbers_266: Array[Int] = Array(1, 3, 5, 6, 7, 8)
	private final val BitsetNumbers_267: Array[Int] = Array(3, 5, 6, 7, 8)
	private final val BitsetNumbers_268: Array[Int] = Array(1, 2, 5, 6, 7, 8)
	private final val BitsetNumbers_269: Array[Int] = Array(2, 5, 6, 7, 8)
	private final val BitsetNumbers_270: Array[Int] = Array(1, 5, 6, 7, 8)
	private final val BitsetNumbers_271: Array[Int] = Array(5, 6, 7, 8)
	private final val BitsetNumbers_272: Array[Int] = Array(1, 2, 3, 4, 6, 7, 8)
	private final val BitsetNumbers_273: Array[Int] = Array(2, 3, 4, 6, 7, 8)
	private final val BitsetNumbers_274: Array[Int] = Array(1, 3, 4, 6, 7, 8)
	private final val BitsetNumbers_275: Array[Int] = Array(3, 4, 6, 7, 8)
	private final val BitsetNumbers_276: Array[Int] = Array(1, 2, 4, 6, 7, 8)
	private final val BitsetNumbers_277: Array[Int] = Array(2, 4, 6, 7, 8)
	private final val BitsetNumbers_278: Array[Int] = Array(1, 4, 6, 7, 8)
	private final val BitsetNumbers_279: Array[Int] = Array(4, 6, 7, 8)
	private final val BitsetNumbers_280: Array[Int] = Array(1, 2, 3, 6, 7, 8)
	private final val BitsetNumbers_281: Array[Int] = Array(2, 3, 6, 7, 8)
	private final val BitsetNumbers_282: Array[Int] = Array(1, 3, 6, 7, 8)
	private final val BitsetNumbers_283: Array[Int] = Array(3, 6, 7, 8)
	private final val BitsetNumbers_284: Array[Int] = Array(1, 2, 6, 7, 8)
	private final val BitsetNumbers_285: Array[Int] = Array(2, 6, 7, 8)
	private final val BitsetNumbers_286: Array[Int] = Array(1, 6, 7, 8)
	private final val BitsetNumbers_287: Array[Int] = Array(6, 7, 8)
	private final val BitsetNumbers_288: Array[Int] = Array(1, 2, 3, 4, 5, 7, 8)
	private final val BitsetNumbers_289: Array[Int] = Array(2, 3, 4, 5, 7, 8)
	private final val BitsetNumbers_290: Array[Int] = Array(1, 3, 4, 5, 7, 8)
	private final val BitsetNumbers_291: Array[Int] = Array(3, 4, 5, 7, 8)
	private final val BitsetNumbers_292: Array[Int] = Array(1, 2, 4, 5, 7, 8)
	private final val BitsetNumbers_293: Array[Int] = Array(2, 4, 5, 7, 8)
	private final val BitsetNumbers_294: Array[Int] = Array(1, 4, 5, 7, 8)
	private final val BitsetNumbers_295: Array[Int] = Array(4, 5, 7, 8)
	private final val BitsetNumbers_296: Array[Int] = Array(1, 2, 3, 5, 7, 8)
	private final val BitsetNumbers_297: Array[Int] = Array(2, 3, 5, 7, 8)
	private final val BitsetNumbers_298: Array[Int] = Array(1, 3, 5, 7, 8)
	private final val BitsetNumbers_299: Array[Int] = Array(3, 5, 7, 8)
	private final val BitsetNumbers_300: Array[Int] = Array(1, 2, 5, 7, 8)
	private final val BitsetNumbers_301: Array[Int] = Array(2, 5, 7, 8)
	private final val BitsetNumbers_302: Array[Int] = Array(1, 5, 7, 8)
	private final val BitsetNumbers_303: Array[Int] = Array(5, 7, 8)
	private final val BitsetNumbers_304: Array[Int] = Array(1, 2, 3, 4, 7, 8)
	private final val BitsetNumbers_305: Array[Int] = Array(2, 3, 4, 7, 8)
	private final val BitsetNumbers_306: Array[Int] = Array(1, 3, 4, 7, 8)
	private final val BitsetNumbers_307: Array[Int] = Array(3, 4, 7, 8)
	private final val BitsetNumbers_308: Array[Int] = Array(1, 2, 4, 7, 8)
	private final val BitsetNumbers_309: Array[Int] = Array(2, 4, 7, 8)
	private final val BitsetNumbers_310: Array[Int] = Array(1, 4, 7, 8)
	private final val BitsetNumbers_311: Array[Int] = Array(4, 7, 8)
	private final val BitsetNumbers_312: Array[Int] = Array(1, 2, 3, 7, 8)
	private final val BitsetNumbers_313: Array[Int] = Array(2, 3, 7, 8)
	private final val BitsetNumbers_314: Array[Int] = Array(1, 3, 7, 8)
	private final val BitsetNumbers_315: Array[Int] = Array(3, 7, 8)
	private final val BitsetNumbers_316: Array[Int] = Array(1, 2, 7, 8)
	private final val BitsetNumbers_317: Array[Int] = Array(2, 7, 8)
	private final val BitsetNumbers_318: Array[Int] = Array(1, 7, 8)
	private final val BitsetNumbers_319: Array[Int] = Array(7, 8)
	private final val BitsetNumbers_320: Array[Int] = Array(1, 2, 3, 4, 5, 6, 8)
	private final val BitsetNumbers_321: Array[Int] = Array(2, 3, 4, 5, 6, 8)
	private final val BitsetNumbers_322: Array[Int] = Array(1, 3, 4, 5, 6, 8)
	private final val BitsetNumbers_323: Array[Int] = Array(3, 4, 5, 6, 8)
	private final val BitsetNumbers_324: Array[Int] = Array(1, 2, 4, 5, 6, 8)
	private final val BitsetNumbers_325: Array[Int] = Array(2, 4, 5, 6, 8)
	private final val BitsetNumbers_326: Array[Int] = Array(1, 4, 5, 6, 8)
	private final val BitsetNumbers_327: Array[Int] = Array(4, 5, 6, 8)
	private final val BitsetNumbers_328: Array[Int] = Array(1, 2, 3, 5, 6, 8)
	private final val BitsetNumbers_329: Array[Int] = Array(2, 3, 5, 6, 8)
	private final val BitsetNumbers_330: Array[Int] = Array(1, 3, 5, 6, 8)
	private final val BitsetNumbers_331: Array[Int] = Array(3, 5, 6, 8)
	private final val BitsetNumbers_332: Array[Int] = Array(1, 2, 5, 6, 8)
	private final val BitsetNumbers_333: Array[Int] = Array(2, 5, 6, 8)
	private final val BitsetNumbers_334: Array[Int] = Array(1, 5, 6, 8)
	private final val BitsetNumbers_335: Array[Int] = Array(5, 6, 8)
	private final val BitsetNumbers_336: Array[Int] = Array(1, 2, 3, 4, 6, 8)
	private final val BitsetNumbers_337: Array[Int] = Array(2, 3, 4, 6, 8)
	private final val BitsetNumbers_338: Array[Int] = Array(1, 3, 4, 6, 8)
	private final val BitsetNumbers_339: Array[Int] = Array(3, 4, 6, 8)
	private final val BitsetNumbers_340: Array[Int] = Array(1, 2, 4, 6, 8)
	private final val BitsetNumbers_341: Array[Int] = Array(2, 4, 6, 8)
	private final val BitsetNumbers_342: Array[Int] = Array(1, 4, 6, 8)
	private final val BitsetNumbers_343: Array[Int] = Array(4, 6, 8)
	private final val BitsetNumbers_344: Array[Int] = Array(1, 2, 3, 6, 8)
	private final val BitsetNumbers_345: Array[Int] = Array(2, 3, 6, 8)
	private final val BitsetNumbers_346: Array[Int] = Array(1, 3, 6, 8)
	private final val BitsetNumbers_347: Array[Int] = Array(3, 6, 8)
	private final val BitsetNumbers_348: Array[Int] = Array(1, 2, 6, 8)
	private final val BitsetNumbers_349: Array[Int] = Array(2, 6, 8)
	private final val BitsetNumbers_350: Array[Int] = Array(1, 6, 8)
	private final val BitsetNumbers_351: Array[Int] = Array(6, 8)
	private final val BitsetNumbers_352: Array[Int] = Array(1, 2, 3, 4, 5, 8)
	private final val BitsetNumbers_353: Array[Int] = Array(2, 3, 4, 5, 8)
	private final val BitsetNumbers_354: Array[Int] = Array(1, 3, 4, 5, 8)
	private final val BitsetNumbers_355: Array[Int] = Array(3, 4, 5, 8)
	private final val BitsetNumbers_356: Array[Int] = Array(1, 2, 4, 5, 8)
	private final val BitsetNumbers_357: Array[Int] = Array(2, 4, 5, 8)
	private final val BitsetNumbers_358: Array[Int] = Array(1, 4, 5, 8)
	private final val BitsetNumbers_359: Array[Int] = Array(4, 5, 8)
	private final val BitsetNumbers_360: Array[Int] = Array(1, 2, 3, 5, 8)
	private final val BitsetNumbers_361: Array[Int] = Array(2, 3, 5, 8)
	private final val BitsetNumbers_362: Array[Int] = Array(1, 3, 5, 8)
	private final val BitsetNumbers_363: Array[Int] = Array(3, 5, 8)
	private final val BitsetNumbers_364: Array[Int] = Array(1, 2, 5, 8)
	private final val BitsetNumbers_365: Array[Int] = Array(2, 5, 8)
	private final val BitsetNumbers_366: Array[Int] = Array(1, 5, 8)
	private final val BitsetNumbers_367: Array[Int] = Array(5, 8)
	private final val BitsetNumbers_368: Array[Int] = Array(1, 2, 3, 4, 8)
	private final val BitsetNumbers_369: Array[Int] = Array(2, 3, 4, 8)
	private final val BitsetNumbers_370: Array[Int] = Array(1, 3, 4, 8)
	private final val BitsetNumbers_371: Array[Int] = Array(3, 4, 8)
	private final val BitsetNumbers_372: Array[Int] = Array(1, 2, 4, 8)
	private final val BitsetNumbers_373: Array[Int] = Array(2, 4, 8)
	private final val BitsetNumbers_374: Array[Int] = Array(1, 4, 8)
	private final val BitsetNumbers_375: Array[Int] = Array(4, 8)
	private final val BitsetNumbers_376: Array[Int] = Array(1, 2, 3, 8)
	private final val BitsetNumbers_377: Array[Int] = Array(2, 3, 8)
	private final val BitsetNumbers_378: Array[Int] = Array(1, 3, 8)
	private final val BitsetNumbers_379: Array[Int] = Array(3, 8)
	private final val BitsetNumbers_380: Array[Int] = Array(1, 2, 8)
	private final val BitsetNumbers_381: Array[Int] = Array(2, 8)
	private final val BitsetNumbers_382: Array[Int] = Array(1, 8)
	private final val BitsetNumbers_383: Array[Int] = Array(8)
	private final val BitsetNumbers_384: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7)
	private final val BitsetNumbers_385: Array[Int] = Array(2, 3, 4, 5, 6, 7)
	private final val BitsetNumbers_386: Array[Int] = Array(1, 3, 4, 5, 6, 7)
	private final val BitsetNumbers_387: Array[Int] = Array(3, 4, 5, 6, 7)
	private final val BitsetNumbers_388: Array[Int] = Array(1, 2, 4, 5, 6, 7)
	private final val BitsetNumbers_389: Array[Int] = Array(2, 4, 5, 6, 7)
	private final val BitsetNumbers_390: Array[Int] = Array(1, 4, 5, 6, 7)
	private final val BitsetNumbers_391: Array[Int] = Array(4, 5, 6, 7)
	private final val BitsetNumbers_392: Array[Int] = Array(1, 2, 3, 5, 6, 7)
	private final val BitsetNumbers_393: Array[Int] = Array(2, 3, 5, 6, 7)
	private final val BitsetNumbers_394: Array[Int] = Array(1, 3, 5, 6, 7)
	private final val BitsetNumbers_395: Array[Int] = Array(3, 5, 6, 7)
	private final val BitsetNumbers_396: Array[Int] = Array(1, 2, 5, 6, 7)
	private final val BitsetNumbers_397: Array[Int] = Array(2, 5, 6, 7)
	private final val BitsetNumbers_398: Array[Int] = Array(1, 5, 6, 7)
	private final val BitsetNumbers_399: Array[Int] = Array(5, 6, 7)
	private final val BitsetNumbers_400: Array[Int] = Array(1, 2, 3, 4, 6, 7)
	private final val BitsetNumbers_401: Array[Int] = Array(2, 3, 4, 6, 7)
	private final val BitsetNumbers_402: Array[Int] = Array(1, 3, 4, 6, 7)
	private final val BitsetNumbers_403: Array[Int] = Array(3, 4, 6, 7)
	private final val BitsetNumbers_404: Array[Int] = Array(1, 2, 4, 6, 7)
	private final val BitsetNumbers_405: Array[Int] = Array(2, 4, 6, 7)
	private final val BitsetNumbers_406: Array[Int] = Array(1, 4, 6, 7)
	private final val BitsetNumbers_407: Array[Int] = Array(4, 6, 7)
	private final val BitsetNumbers_408: Array[Int] = Array(1, 2, 3, 6, 7)
	private final val BitsetNumbers_409: Array[Int] = Array(2, 3, 6, 7)
	private final val BitsetNumbers_410: Array[Int] = Array(1, 3, 6, 7)
	private final val BitsetNumbers_411: Array[Int] = Array(3, 6, 7)
	private final val BitsetNumbers_412: Array[Int] = Array(1, 2, 6, 7)
	private final val BitsetNumbers_413: Array[Int] = Array(2, 6, 7)
	private final val BitsetNumbers_414: Array[Int] = Array(1, 6, 7)
	private final val BitsetNumbers_415: Array[Int] = Array(6, 7)
	private final val BitsetNumbers_416: Array[Int] = Array(1, 2, 3, 4, 5, 7)
	private final val BitsetNumbers_417: Array[Int] = Array(2, 3, 4, 5, 7)
	private final val BitsetNumbers_418: Array[Int] = Array(1, 3, 4, 5, 7)
	private final val BitsetNumbers_419: Array[Int] = Array(3, 4, 5, 7)
	private final val BitsetNumbers_420: Array[Int] = Array(1, 2, 4, 5, 7)
	private final val BitsetNumbers_421: Array[Int] = Array(2, 4, 5, 7)
	private final val BitsetNumbers_422: Array[Int] = Array(1, 4, 5, 7)
	private final val BitsetNumbers_423: Array[Int] = Array(4, 5, 7)
	private final val BitsetNumbers_424: Array[Int] = Array(1, 2, 3, 5, 7)
	private final val BitsetNumbers_425: Array[Int] = Array(2, 3, 5, 7)
	private final val BitsetNumbers_426: Array[Int] = Array(1, 3, 5, 7)
	private final val BitsetNumbers_427: Array[Int] = Array(3, 5, 7)
	private final val BitsetNumbers_428: Array[Int] = Array(1, 2, 5, 7)
	private final val BitsetNumbers_429: Array[Int] = Array(2, 5, 7)
	private final val BitsetNumbers_430: Array[Int] = Array(1, 5, 7)
	private final val BitsetNumbers_431: Array[Int] = Array(5, 7)
	private final val BitsetNumbers_432: Array[Int] = Array(1, 2, 3, 4, 7)
	private final val BitsetNumbers_433: Array[Int] = Array(2, 3, 4, 7)
	private final val BitsetNumbers_434: Array[Int] = Array(1, 3, 4, 7)
	private final val BitsetNumbers_435: Array[Int] = Array(3, 4, 7)
	private final val BitsetNumbers_436: Array[Int] = Array(1, 2, 4, 7)
	private final val BitsetNumbers_437: Array[Int] = Array(2, 4, 7)
	private final val BitsetNumbers_438: Array[Int] = Array(1, 4, 7)
	private final val BitsetNumbers_439: Array[Int] = Array(4, 7)
	private final val BitsetNumbers_440: Array[Int] = Array(1, 2, 3, 7)
	private final val BitsetNumbers_441: Array[Int] = Array(2, 3, 7)
	private final val BitsetNumbers_442: Array[Int] = Array(1, 3, 7)
	private final val BitsetNumbers_443: Array[Int] = Array(3, 7)
	private final val BitsetNumbers_444: Array[Int] = Array(1, 2, 7)
	private final val BitsetNumbers_445: Array[Int] = Array(2, 7)
	private final val BitsetNumbers_446: Array[Int] = Array(1, 7)
	private final val BitsetNumbers_447: Array[Int] = Array(7)
	private final val BitsetNumbers_448: Array[Int] = Array(1, 2, 3, 4, 5, 6)
	private final val BitsetNumbers_449: Array[Int] = Array(2, 3, 4, 5, 6)
	private final val BitsetNumbers_450: Array[Int] = Array(1, 3, 4, 5, 6)
	private final val BitsetNumbers_451: Array[Int] = Array(3, 4, 5, 6)
	private final val BitsetNumbers_452: Array[Int] = Array(1, 2, 4, 5, 6)
	private final val BitsetNumbers_453: Array[Int] = Array(2, 4, 5, 6)
	private final val BitsetNumbers_454: Array[Int] = Array(1, 4, 5, 6)
	private final val BitsetNumbers_455: Array[Int] = Array(4, 5, 6)
	private final val BitsetNumbers_456: Array[Int] = Array(1, 2, 3, 5, 6)
	private final val BitsetNumbers_457: Array[Int] = Array(2, 3, 5, 6)
	private final val BitsetNumbers_458: Array[Int] = Array(1, 3, 5, 6)
	private final val BitsetNumbers_459: Array[Int] = Array(3, 5, 6)
	private final val BitsetNumbers_460: Array[Int] = Array(1, 2, 5, 6)
	private final val BitsetNumbers_461: Array[Int] = Array(2, 5, 6)
	private final val BitsetNumbers_462: Array[Int] = Array(1, 5, 6)
	private final val BitsetNumbers_463: Array[Int] = Array(5, 6)
	private final val BitsetNumbers_464: Array[Int] = Array(1, 2, 3, 4, 6)
	private final val BitsetNumbers_465: Array[Int] = Array(2, 3, 4, 6)
	private final val BitsetNumbers_466: Array[Int] = Array(1, 3, 4, 6)
	private final val BitsetNumbers_467: Array[Int] = Array(3, 4, 6)
	private final val BitsetNumbers_468: Array[Int] = Array(1, 2, 4, 6)
	private final val BitsetNumbers_469: Array[Int] = Array(2, 4, 6)
	private final val BitsetNumbers_470: Array[Int] = Array(1, 4, 6)
	private final val BitsetNumbers_471: Array[Int] = Array(4, 6)
	private final val BitsetNumbers_472: Array[Int] = Array(1, 2, 3, 6)
	private final val BitsetNumbers_473: Array[Int] = Array(2, 3, 6)
	private final val BitsetNumbers_474: Array[Int] = Array(1, 3, 6)
	private final val BitsetNumbers_475: Array[Int] = Array(3, 6)
	private final val BitsetNumbers_476: Array[Int] = Array(1, 2, 6)
	private final val BitsetNumbers_477: Array[Int] = Array(2, 6)
	private final val BitsetNumbers_478: Array[Int] = Array(1, 6)
	private final val BitsetNumbers_479: Array[Int] = Array(6)
	private final val BitsetNumbers_480: Array[Int] = Array(1, 2, 3, 4, 5)
	private final val BitsetNumbers_481: Array[Int] = Array(2, 3, 4, 5)
	private final val BitsetNumbers_482: Array[Int] = Array(1, 3, 4, 5)
	private final val BitsetNumbers_483: Array[Int] = Array(3, 4, 5)
	private final val BitsetNumbers_484: Array[Int] = Array(1, 2, 4, 5)
	private final val BitsetNumbers_485: Array[Int] = Array(2, 4, 5)
	private final val BitsetNumbers_486: Array[Int] = Array(1, 4, 5)
	private final val BitsetNumbers_487: Array[Int] = Array(4, 5)
	private final val BitsetNumbers_488: Array[Int] = Array(1, 2, 3, 5)
	private final val BitsetNumbers_489: Array[Int] = Array(2, 3, 5)
	private final val BitsetNumbers_490: Array[Int] = Array(1, 3, 5)
	private final val BitsetNumbers_491: Array[Int] = Array(3, 5)
	private final val BitsetNumbers_492: Array[Int] = Array(1, 2, 5)
	private final val BitsetNumbers_493: Array[Int] = Array(2, 5)
	private final val BitsetNumbers_494: Array[Int] = Array(1, 5)
	private final val BitsetNumbers_495: Array[Int] = Array(5)
	private final val BitsetNumbers_496: Array[Int] = Array(1, 2, 3, 4)
	private final val BitsetNumbers_497: Array[Int] = Array(2, 3, 4)
	private final val BitsetNumbers_498: Array[Int] = Array(1, 3, 4)
	private final val BitsetNumbers_499: Array[Int] = Array(3, 4)
	private final val BitsetNumbers_500: Array[Int] = Array(1, 2, 4)
	private final val BitsetNumbers_501: Array[Int] = Array(2, 4)
	private final val BitsetNumbers_502: Array[Int] = Array(1, 4)
	private final val BitsetNumbers_503: Array[Int] = Array(4)
	private final val BitsetNumbers_504: Array[Int] = Array(1, 2, 3)
	private final val BitsetNumbers_505: Array[Int] = Array(2, 3)
	private final val BitsetNumbers_506: Array[Int] = Array(1, 3)
	private final val BitsetNumbers_507: Array[Int] = Array(3)
	private final val BitsetNumbers_508: Array[Int] = Array(1, 2)
	private final val BitsetNumbers_509: Array[Int] = Array(2)
	private final val BitsetNumbers_510: Array[Int] = Array(1)
	private final val BitsetNumbers_511: Array[Int] = Array()

	/**
	 * For each bitset combination there is an array pointing
	 * to the numbers set in the bitset
	 */
	final val BitsetPossibleNumbers: Array[Array[Int]] = Array(
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
