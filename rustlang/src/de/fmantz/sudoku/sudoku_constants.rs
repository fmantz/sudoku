/*
 * sudoku - Sudoku solver for comparison Scala with Rust
 *        - The motivation is explained in the README.md file in the top level folder.
 * Copyright (C]; 2020 Florian Mantz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option]; any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#[allow(dead_code)]
pub const CHECK_BITS: u16 = !0 >> (16 - PUZZLE_SIZE);

pub const NEW_SUDOKU_SEPARATOR: &str = "Grid";
pub const EMPTY_CHAR: char = '0';
pub const QQWING_EMPTY_CHAR: char = '.';
pub const CELL_COUNT: usize = 81;
pub const PUZZLE_SIZE: usize = 9;
pub const SQUARE_SIZE: usize = 3;

//binary: Size times "1"
pub const PARALLELIZATION_COUNT: u32 = 1024;
pub const PARALLELIZATION_COUNT_CUDA: u32 = 65536;

const BITSET_NUMBERS_000: &[u8] = &[4, 7, 8, 3, 5, 1, 9, 6, 2];
const BITSET_NUMBERS_001: &[u8] = &[8, 6, 7, 5, 9, 4, 2, 3];
const BITSET_NUMBERS_002: &[u8] = &[3, 4, 5, 1, 9, 6, 7, 8];
const BITSET_NUMBERS_003: &[u8] = &[7, 8, 9, 5, 4, 6, 3];
const BITSET_NUMBERS_004: &[u8] = &[9, 1, 8, 6, 7, 5, 4, 2];
const BITSET_NUMBERS_005: &[u8] = &[8, 6, 5, 7, 4, 2, 9];
const BITSET_NUMBERS_006: &[u8] = &[6, 9, 4, 5, 8, 7, 1];
const BITSET_NUMBERS_007: &[u8] = &[9, 7, 4, 6, 8, 5];
const BITSET_NUMBERS_008: &[u8] = &[1, 8, 9, 3, 2, 5, 7, 6];
const BITSET_NUMBERS_009: &[u8] = &[3, 7, 2, 5, 9, 8, 6];
const BITSET_NUMBERS_010: &[u8] = &[7, 6, 9, 3, 1, 8, 5];
const BITSET_NUMBERS_011: &[u8] = &[7, 8, 3, 6, 5, 9];
const BITSET_NUMBERS_012: &[u8] = &[8, 2, 6, 7, 9, 1, 5];
const BITSET_NUMBERS_013: &[u8] = &[7, 5, 9, 8, 6, 2];
const BITSET_NUMBERS_014: &[u8] = &[9, 8, 7, 1, 5, 6];
const BITSET_NUMBERS_015: &[u8] = &[6, 8, 7, 9, 5];
const BITSET_NUMBERS_016: &[u8] = &[8, 2, 9, 7, 1, 3, 6, 4];
const BITSET_NUMBERS_017: &[u8] = &[3, 4, 7, 8, 2, 6, 9];
const BITSET_NUMBERS_018: &[u8] = &[6, 1, 4, 3, 9, 7, 8];
const BITSET_NUMBERS_019: &[u8] = &[9, 3, 7, 4, 6, 8];
const BITSET_NUMBERS_020: &[u8] = &[7, 2, 6, 9, 4, 1, 8];
const BITSET_NUMBERS_021: &[u8] = &[4, 9, 6, 2, 7, 8];
const BITSET_NUMBERS_022: &[u8] = &[7, 8, 4, 1, 6, 9];
const BITSET_NUMBERS_023: &[u8] = &[9, 4, 8, 7, 6];
const BITSET_NUMBERS_024: &[u8] = &[7, 9, 8, 2, 3, 6, 1];
const BITSET_NUMBERS_025: &[u8] = &[3, 6, 9, 8, 7, 2];
const BITSET_NUMBERS_026: &[u8] = &[3, 6, 1, 8, 9, 7];
const BITSET_NUMBERS_027: &[u8] = &[8, 3, 7, 6, 9];
const BITSET_NUMBERS_028: &[u8] = &[8, 1, 2, 7, 9, 6];
const BITSET_NUMBERS_029: &[u8] = &[6, 8, 9, 7, 2];
const BITSET_NUMBERS_030: &[u8] = &[1, 8, 9, 7, 6];
const BITSET_NUMBERS_031: &[u8] = &[8, 9, 7, 6];
const BITSET_NUMBERS_032: &[u8] = &[8, 7, 3, 1, 2, 5, 9, 4];
const BITSET_NUMBERS_033: &[u8] = &[9, 3, 5, 4, 2, 7, 8];
const BITSET_NUMBERS_034: &[u8] = &[7, 3, 8, 4, 9, 5, 1];
const BITSET_NUMBERS_035: &[u8] = &[9, 4, 3, 7, 5, 8];
const BITSET_NUMBERS_036: &[u8] = &[2, 7, 8, 4, 9, 1, 5];
const BITSET_NUMBERS_037: &[u8] = &[4, 7, 5, 8, 2, 9];
const BITSET_NUMBERS_038: &[u8] = &[5, 7, 9, 8, 1, 4];
const BITSET_NUMBERS_039: &[u8] = &[5, 8, 9, 7, 4];
const BITSET_NUMBERS_040: &[u8] = &[7, 5, 8, 9, 3, 1, 2];
const BITSET_NUMBERS_041: &[u8] = &[9, 8, 3, 7, 5, 2];
const BITSET_NUMBERS_042: &[u8] = &[5, 3, 9, 7, 8, 1];
const BITSET_NUMBERS_043: &[u8] = &[5, 3, 9, 8, 7];
const BITSET_NUMBERS_044: &[u8] = &[5, 2, 9, 7, 1, 8];
const BITSET_NUMBERS_045: &[u8] = &[5, 8, 9, 2, 7];
const BITSET_NUMBERS_046: &[u8] = &[1, 5, 7, 9, 8];
const BITSET_NUMBERS_047: &[u8] = &[7, 8, 9, 5];
const BITSET_NUMBERS_048: &[u8] = &[7, 9, 8, 2, 1, 4, 3];
const BITSET_NUMBERS_049: &[u8] = &[8, 9, 3, 7, 4, 2];
const BITSET_NUMBERS_050: &[u8] = &[1, 3, 4, 7, 8, 9];
const BITSET_NUMBERS_051: &[u8] = &[9, 4, 7, 3, 8];
const BITSET_NUMBERS_052: &[u8] = &[7, 4, 8, 2, 1, 9];
const BITSET_NUMBERS_053: &[u8] = &[4, 2, 7, 8, 9];
const BITSET_NUMBERS_054: &[u8] = &[1, 9, 7, 8, 4];
const BITSET_NUMBERS_055: &[u8] = &[4, 9, 8, 7];
const BITSET_NUMBERS_056: &[u8] = &[2, 9, 1, 3, 8, 7];
const BITSET_NUMBERS_057: &[u8] = &[2, 8, 9, 7, 3];
const BITSET_NUMBERS_058: &[u8] = &[8, 9, 1, 7, 3];
const BITSET_NUMBERS_059: &[u8] = &[3, 7, 9, 8];
const BITSET_NUMBERS_060: &[u8] = &[8, 7, 1, 9, 2];
const BITSET_NUMBERS_061: &[u8] = &[2, 7, 8, 9];
const BITSET_NUMBERS_062: &[u8] = &[1, 8, 7, 9];
const BITSET_NUMBERS_063: &[u8] = &[9, 7, 8];
const BITSET_NUMBERS_064: &[u8] = &[6, 5, 3, 8, 2, 4, 1, 9];
const BITSET_NUMBERS_065: &[u8] = &[9, 2, 5, 3, 6, 8, 4];
const BITSET_NUMBERS_066: &[u8] = &[5, 4, 6, 8, 1, 9, 3];
const BITSET_NUMBERS_067: &[u8] = &[6, 8, 4, 5, 9, 3];
const BITSET_NUMBERS_068: &[u8] = &[1, 8, 5, 2, 6, 9, 4];
const BITSET_NUMBERS_069: &[u8] = &[6, 9, 5, 2, 8, 4];
const BITSET_NUMBERS_070: &[u8] = &[8, 9, 5, 6, 1, 4];
const BITSET_NUMBERS_071: &[u8] = &[9, 8, 6, 5, 4];
const BITSET_NUMBERS_072: &[u8] = &[2, 5, 9, 3, 1, 6, 8];
const BITSET_NUMBERS_073: &[u8] = &[5, 6, 8, 3, 2, 9];
const BITSET_NUMBERS_074: &[u8] = &[6, 5, 8, 9, 3, 1];
const BITSET_NUMBERS_075: &[u8] = &[3, 9, 5, 6, 8];
const BITSET_NUMBERS_076: &[u8] = &[1, 2, 6, 5, 9, 8];
const BITSET_NUMBERS_077: &[u8] = &[5, 8, 9, 6, 2];
const BITSET_NUMBERS_078: &[u8] = &[5, 9, 1, 8, 6];
const BITSET_NUMBERS_079: &[u8] = &[5, 8, 9, 6];
const BITSET_NUMBERS_080: &[u8] = &[8, 1, 2, 6, 9, 3, 4];
const BITSET_NUMBERS_081: &[u8] = &[3, 4, 8, 6, 2, 9];
const BITSET_NUMBERS_082: &[u8] = &[3, 8, 4, 1, 6, 9];
const BITSET_NUMBERS_083: &[u8] = &[9, 6, 3, 8, 4];
const BITSET_NUMBERS_084: &[u8] = &[9, 1, 4, 6, 2, 8];
const BITSET_NUMBERS_085: &[u8] = &[2, 8, 6, 9, 4];
const BITSET_NUMBERS_086: &[u8] = &[6, 9, 8, 1, 4];
const BITSET_NUMBERS_087: &[u8] = &[9, 4, 8, 6];
const BITSET_NUMBERS_088: &[u8] = &[1, 2, 6, 9, 3, 8];
const BITSET_NUMBERS_089: &[u8] = &[3, 8, 9, 6, 2];
const BITSET_NUMBERS_090: &[u8] = &[9, 3, 6, 1, 8];
const BITSET_NUMBERS_091: &[u8] = &[8, 3, 9, 6];
const BITSET_NUMBERS_092: &[u8] = &[1, 6, 2, 9, 8];
const BITSET_NUMBERS_093: &[u8] = &[8, 6, 2, 9];
const BITSET_NUMBERS_094: &[u8] = &[9, 1, 8, 6];
const BITSET_NUMBERS_095: &[u8] = &[8, 9, 6];
const BITSET_NUMBERS_096: &[u8] = &[1, 2, 5, 8, 9, 3, 4];
const BITSET_NUMBERS_097: &[u8] = &[9, 5, 2, 8, 3, 4];
const BITSET_NUMBERS_098: &[u8] = &[3, 1, 5, 4, 8, 9];
const BITSET_NUMBERS_099: &[u8] = &[8, 5, 9, 3, 4];
const BITSET_NUMBERS_100: &[u8] = &[9, 1, 5, 8, 4, 2];
const BITSET_NUMBERS_101: &[u8] = &[8, 2, 5, 9, 4];
const BITSET_NUMBERS_102: &[u8] = &[8, 4, 1, 9, 5];
const BITSET_NUMBERS_103: &[u8] = &[5, 4, 9, 8];
const BITSET_NUMBERS_104: &[u8] = &[2, 8, 5, 1, 9, 3];
const BITSET_NUMBERS_105: &[u8] = &[2, 8, 3, 9, 5];
const BITSET_NUMBERS_106: &[u8] = &[1, 9, 5, 8, 3];
const BITSET_NUMBERS_107: &[u8] = &[9, 5, 3, 8];
const BITSET_NUMBERS_108: &[u8] = &[5, 2, 8, 9, 1];
const BITSET_NUMBERS_109: &[u8] = &[8, 5, 9, 2];
const BITSET_NUMBERS_110: &[u8] = &[9, 1, 5, 8];
const BITSET_NUMBERS_111: &[u8] = &[9, 5, 8];
const BITSET_NUMBERS_112: &[u8] = &[2, 9, 4, 1, 8, 3];
const BITSET_NUMBERS_113: &[u8] = &[3, 2, 4, 9, 8];
const BITSET_NUMBERS_114: &[u8] = &[8, 3, 4, 9, 1];
const BITSET_NUMBERS_115: &[u8] = &[9, 8, 4, 3];
const BITSET_NUMBERS_116: &[u8] = &[9, 8, 1, 2, 4];
const BITSET_NUMBERS_117: &[u8] = &[4, 2, 9, 8];
const BITSET_NUMBERS_118: &[u8] = &[9, 1, 8, 4];
const BITSET_NUMBERS_119: &[u8] = &[9, 8, 4];
const BITSET_NUMBERS_120: &[u8] = &[3, 9, 8, 1, 2];
const BITSET_NUMBERS_121: &[u8] = &[8, 2, 3, 9];
const BITSET_NUMBERS_122: &[u8] = &[8, 3, 1, 9];
const BITSET_NUMBERS_123: &[u8] = &[3, 8, 9];
const BITSET_NUMBERS_124: &[u8] = &[1, 9, 2, 8];
const BITSET_NUMBERS_125: &[u8] = &[2, 9, 8];
const BITSET_NUMBERS_126: &[u8] = &[8, 1, 9];
const BITSET_NUMBERS_127: &[u8] = &[8, 9];
const BITSET_NUMBERS_128: &[u8] = &[7, 5, 1, 2, 3, 4, 9, 6];
const BITSET_NUMBERS_129: &[u8] = &[7, 2, 5, 4, 9, 6, 3];
const BITSET_NUMBERS_130: &[u8] = &[4, 7, 1, 3, 9, 6, 5];
const BITSET_NUMBERS_131: &[u8] = &[3, 6, 7, 9, 4, 5];
const BITSET_NUMBERS_132: &[u8] = &[9, 6, 7, 2, 4, 5, 1];
const BITSET_NUMBERS_133: &[u8] = &[5, 7, 4, 6, 2, 9];
const BITSET_NUMBERS_134: &[u8] = &[6, 4, 5, 9, 1, 7];
const BITSET_NUMBERS_135: &[u8] = &[4, 5, 6, 9, 7];
const BITSET_NUMBERS_136: &[u8] = &[1, 2, 7, 6, 3, 9, 5];
const BITSET_NUMBERS_137: &[u8] = &[7, 5, 3, 9, 2, 6];
const BITSET_NUMBERS_138: &[u8] = &[6, 7, 9, 3, 1, 5];
const BITSET_NUMBERS_139: &[u8] = &[5, 7, 3, 9, 6];
const BITSET_NUMBERS_140: &[u8] = &[5, 2, 1, 6, 9, 7];
const BITSET_NUMBERS_141: &[u8] = &[9, 7, 5, 6, 2];
const BITSET_NUMBERS_142: &[u8] = &[6, 1, 7, 5, 9];
const BITSET_NUMBERS_143: &[u8] = &[5, 7, 9, 6];
const BITSET_NUMBERS_144: &[u8] = &[4, 9, 2, 6, 7, 3, 1];
const BITSET_NUMBERS_145: &[u8] = &[3, 4, 7, 2, 6, 9];
const BITSET_NUMBERS_146: &[u8] = &[3, 1, 6, 7, 9, 4];
const BITSET_NUMBERS_147: &[u8] = &[4, 3, 7, 6, 9];
const BITSET_NUMBERS_148: &[u8] = &[4, 7, 9, 2, 6, 1];
const BITSET_NUMBERS_149: &[u8] = &[6, 9, 4, 2, 7];
const BITSET_NUMBERS_150: &[u8] = &[7, 4, 9, 6, 1];
const BITSET_NUMBERS_151: &[u8] = &[6, 7, 4, 9];
const BITSET_NUMBERS_152: &[u8] = &[7, 2, 1, 6, 3, 9];
const BITSET_NUMBERS_153: &[u8] = &[3, 9, 6, 2, 7];
const BITSET_NUMBERS_154: &[u8] = &[3, 6, 9, 7, 1];
const BITSET_NUMBERS_155: &[u8] = &[9, 6, 3, 7];
const BITSET_NUMBERS_156: &[u8] = &[9, 2, 1, 6, 7];
const BITSET_NUMBERS_157: &[u8] = &[9, 7, 2, 6];
const BITSET_NUMBERS_158: &[u8] = &[9, 7, 1, 6];
const BITSET_NUMBERS_159: &[u8] = &[9, 7, 6];
const BITSET_NUMBERS_160: &[u8] = &[2, 9, 4, 1, 7, 3, 5];
const BITSET_NUMBERS_161: &[u8] = &[2, 5, 3, 7, 9, 4];
const BITSET_NUMBERS_162: &[u8] = &[9, 7, 3, 5, 1, 4];
const BITSET_NUMBERS_163: &[u8] = &[9, 4, 7, 5, 3];
const BITSET_NUMBERS_164: &[u8] = &[9, 4, 1, 2, 7, 5];
const BITSET_NUMBERS_165: &[u8] = &[4, 9, 7, 2, 5];
const BITSET_NUMBERS_166: &[u8] = &[4, 7, 9, 1, 5];
const BITSET_NUMBERS_167: &[u8] = &[9, 5, 7, 4];
const BITSET_NUMBERS_168: &[u8] = &[5, 1, 9, 2, 7, 3];
const BITSET_NUMBERS_169: &[u8] = &[9, 3, 2, 5, 7];
const BITSET_NUMBERS_170: &[u8] = &[7, 5, 9, 3, 1];
const BITSET_NUMBERS_171: &[u8] = &[3, 9, 5, 7];
const BITSET_NUMBERS_172: &[u8] = &[5, 7, 2, 1, 9];
const BITSET_NUMBERS_173: &[u8] = &[7, 9, 5, 2];
const BITSET_NUMBERS_174: &[u8] = &[9, 7, 5, 1];
const BITSET_NUMBERS_175: &[u8] = &[5, 9, 7];
const BITSET_NUMBERS_176: &[u8] = &[1, 3, 2, 4, 9, 7];
const BITSET_NUMBERS_177: &[u8] = &[2, 4, 7, 9, 3];
const BITSET_NUMBERS_178: &[u8] = &[7, 4, 1, 3, 9];
const BITSET_NUMBERS_179: &[u8] = &[9, 7, 4, 3];
const BITSET_NUMBERS_180: &[u8] = &[2, 1, 4, 9, 7];
const BITSET_NUMBERS_181: &[u8] = &[4, 9, 7, 2];
const BITSET_NUMBERS_182: &[u8] = &[9, 4, 1, 7];
const BITSET_NUMBERS_183: &[u8] = &[7, 9, 4];
const BITSET_NUMBERS_184: &[u8] = &[7, 3, 9, 1, 2];
const BITSET_NUMBERS_185: &[u8] = &[2, 7, 9, 3];
const BITSET_NUMBERS_186: &[u8] = &[1, 3, 9, 7];
const BITSET_NUMBERS_187: &[u8] = &[3, 9, 7];
const BITSET_NUMBERS_188: &[u8] = &[2, 1, 9, 7];
const BITSET_NUMBERS_189: &[u8] = &[2, 7, 9];
const BITSET_NUMBERS_190: &[u8] = &[1, 9, 7];
const BITSET_NUMBERS_191: &[u8] = &[9, 7];
const BITSET_NUMBERS_192: &[u8] = &[6, 9, 3, 5, 4, 1, 2];
const BITSET_NUMBERS_193: &[u8] = &[9, 3, 6, 4, 2, 5];
const BITSET_NUMBERS_194: &[u8] = &[1, 5, 3, 4, 9, 6];
const BITSET_NUMBERS_195: &[u8] = &[5, 3, 9, 4, 6];
const BITSET_NUMBERS_196: &[u8] = &[4, 6, 9, 1, 5, 2];
const BITSET_NUMBERS_197: &[u8] = &[9, 4, 6, 5, 2];
const BITSET_NUMBERS_198: &[u8] = &[9, 1, 5, 4, 6];
const BITSET_NUMBERS_199: &[u8] = &[6, 9, 4, 5];
const BITSET_NUMBERS_200: &[u8] = &[6, 3, 5, 9, 1, 2];
const BITSET_NUMBERS_201: &[u8] = &[3, 5, 2, 9, 6];
const BITSET_NUMBERS_202: &[u8] = &[5, 3, 6, 1, 9];
const BITSET_NUMBERS_203: &[u8] = &[6, 5, 3, 9];
const BITSET_NUMBERS_204: &[u8] = &[9, 2, 1, 6, 5];
const BITSET_NUMBERS_205: &[u8] = &[9, 5, 2, 6];
const BITSET_NUMBERS_206: &[u8] = &[6, 1, 9, 5];
const BITSET_NUMBERS_207: &[u8] = &[5, 6, 9];
const BITSET_NUMBERS_208: &[u8] = &[4, 9, 2, 1, 3, 6];
const BITSET_NUMBERS_209: &[u8] = &[2, 9, 4, 3, 6];
const BITSET_NUMBERS_210: &[u8] = &[4, 6, 3, 1, 9];
const BITSET_NUMBERS_211: &[u8] = &[3, 4, 6, 9];
const BITSET_NUMBERS_212: &[u8] = &[4, 2, 6, 1, 9];
const BITSET_NUMBERS_213: &[u8] = &[2, 6, 4, 9];
const BITSET_NUMBERS_214: &[u8] = &[4, 6, 9, 1];
const BITSET_NUMBERS_215: &[u8] = &[4, 9, 6];
const BITSET_NUMBERS_216: &[u8] = &[6, 1, 9, 2, 3];
const BITSET_NUMBERS_217: &[u8] = &[9, 6, 3, 2];
const BITSET_NUMBERS_218: &[u8] = &[9, 3, 6, 1];
const BITSET_NUMBERS_219: &[u8] = &[9, 6, 3];
const BITSET_NUMBERS_220: &[u8] = &[6, 2, 9, 1];
const BITSET_NUMBERS_221: &[u8] = &[2, 9, 6];
const BITSET_NUMBERS_222: &[u8] = &[1, 6, 9];
const BITSET_NUMBERS_223: &[u8] = &[6, 9];
const BITSET_NUMBERS_224: &[u8] = &[4, 5, 2, 9, 1, 3];
const BITSET_NUMBERS_225: &[u8] = &[4, 9, 3, 5, 2];
const BITSET_NUMBERS_226: &[u8] = &[9, 3, 1, 4, 5];
const BITSET_NUMBERS_227: &[u8] = &[4, 5, 9, 3];
const BITSET_NUMBERS_228: &[u8] = &[2, 1, 4, 5, 9];
const BITSET_NUMBERS_229: &[u8] = &[5, 9, 2, 4];
const BITSET_NUMBERS_230: &[u8] = &[1, 9, 5, 4];
const BITSET_NUMBERS_231: &[u8] = &[5, 9, 4];
const BITSET_NUMBERS_232: &[u8] = &[2, 9, 1, 5, 3];
const BITSET_NUMBERS_233: &[u8] = &[5, 2, 9, 3];
const BITSET_NUMBERS_234: &[u8] = &[1, 3, 9, 5];
const BITSET_NUMBERS_235: &[u8] = &[9, 5, 3];
const BITSET_NUMBERS_236: &[u8] = &[9, 2, 5, 1];
const BITSET_NUMBERS_237: &[u8] = &[9, 2, 5];
const BITSET_NUMBERS_238: &[u8] = &[9, 5, 1];
const BITSET_NUMBERS_239: &[u8] = &[5, 9];
const BITSET_NUMBERS_240: &[u8] = &[3, 9, 4, 1, 2];
const BITSET_NUMBERS_241: &[u8] = &[2, 9, 3, 4];
const BITSET_NUMBERS_242: &[u8] = &[3, 4, 1, 9];
const BITSET_NUMBERS_243: &[u8] = &[3, 9, 4];
const BITSET_NUMBERS_244: &[u8] = &[4, 2, 1, 9];
const BITSET_NUMBERS_245: &[u8] = &[9, 4, 2];
const BITSET_NUMBERS_246: &[u8] = &[1, 9, 4];
const BITSET_NUMBERS_247: &[u8] = &[9, 4];
const BITSET_NUMBERS_248: &[u8] = &[3, 1, 2, 9];
const BITSET_NUMBERS_249: &[u8] = &[2, 9, 3];
const BITSET_NUMBERS_250: &[u8] = &[1, 9, 3];
const BITSET_NUMBERS_251: &[u8] = &[9, 3];
const BITSET_NUMBERS_252: &[u8] = &[2, 9, 1];
const BITSET_NUMBERS_253: &[u8] = &[2, 9];
const BITSET_NUMBERS_254: &[u8] = &[9, 1];
const BITSET_NUMBERS_255: &[u8] = &[9];
const BITSET_NUMBERS_256: &[u8] = &[1, 3, 6, 2, 5, 4, 8, 7];
const BITSET_NUMBERS_257: &[u8] = &[5, 4, 8, 3, 6, 2, 7];
const BITSET_NUMBERS_258: &[u8] = &[3, 4, 1, 5, 7, 6, 8];
const BITSET_NUMBERS_259: &[u8] = &[4, 6, 8, 7, 5, 3];
const BITSET_NUMBERS_260: &[u8] = &[7, 8, 2, 4, 1, 6, 5];
const BITSET_NUMBERS_261: &[u8] = &[2, 7, 8, 5, 6, 4];
const BITSET_NUMBERS_262: &[u8] = &[8, 5, 4, 6, 7, 1];
const BITSET_NUMBERS_263: &[u8] = &[6, 5, 7, 4, 8];
const BITSET_NUMBERS_264: &[u8] = &[7, 3, 5, 8, 6, 1, 2];
const BITSET_NUMBERS_265: &[u8] = &[6, 7, 3, 8, 2, 5];
const BITSET_NUMBERS_266: &[u8] = &[6, 1, 5, 3, 7, 8];
const BITSET_NUMBERS_267: &[u8] = &[8, 7, 6, 3, 5];
const BITSET_NUMBERS_268: &[u8] = &[5, 2, 8, 1, 6, 7];
const BITSET_NUMBERS_269: &[u8] = &[5, 8, 7, 6, 2];
const BITSET_NUMBERS_270: &[u8] = &[1, 7, 8, 6, 5];
const BITSET_NUMBERS_271: &[u8] = &[7, 6, 5, 8];
const BITSET_NUMBERS_272: &[u8] = &[7, 1, 3, 4, 6, 2, 8];
const BITSET_NUMBERS_273: &[u8] = &[7, 3, 6, 8, 4, 2];
const BITSET_NUMBERS_274: &[u8] = &[3, 6, 1, 7, 8, 4];
const BITSET_NUMBERS_275: &[u8] = &[3, 8, 4, 7, 6];
const BITSET_NUMBERS_276: &[u8] = &[7, 8, 6, 2, 1, 4];
const BITSET_NUMBERS_277: &[u8] = &[6, 8, 4, 2, 7];
const BITSET_NUMBERS_278: &[u8] = &[6, 8, 1, 4, 7];
const BITSET_NUMBERS_279: &[u8] = &[7, 6, 4, 8];
const BITSET_NUMBERS_280: &[u8] = &[1, 2, 7, 6, 3, 8];
const BITSET_NUMBERS_281: &[u8] = &[7, 3, 6, 8, 2];
const BITSET_NUMBERS_282: &[u8] = &[6, 1, 8, 7, 3];
const BITSET_NUMBERS_283: &[u8] = &[8, 6, 7, 3];
const BITSET_NUMBERS_284: &[u8] = &[7, 2, 1, 6, 8];
const BITSET_NUMBERS_285: &[u8] = &[2, 8, 6, 7];
const BITSET_NUMBERS_286: &[u8] = &[8, 1, 7, 6];
const BITSET_NUMBERS_287: &[u8] = &[6, 7, 8];
const BITSET_NUMBERS_288: &[u8] = &[8, 7, 2, 4, 3, 1, 5];
const BITSET_NUMBERS_289: &[u8] = &[4, 5, 7, 2, 3, 8];
const BITSET_NUMBERS_290: &[u8] = &[3, 8, 5, 1, 7, 4];
const BITSET_NUMBERS_291: &[u8] = &[4, 3, 8, 5, 7];
const BITSET_NUMBERS_292: &[u8] = &[7, 2, 1, 8, 4, 5];
const BITSET_NUMBERS_293: &[u8] = &[4, 7, 5, 8, 2];
const BITSET_NUMBERS_294: &[u8] = &[5, 1, 7, 4, 8];
const BITSET_NUMBERS_295: &[u8] = &[5, 7, 4, 8];
const BITSET_NUMBERS_296: &[u8] = &[7, 3, 5, 1, 2, 8];
const BITSET_NUMBERS_297: &[u8] = &[7, 8, 2, 3, 5];
const BITSET_NUMBERS_298: &[u8] = &[7, 1, 3, 8, 5];
const BITSET_NUMBERS_299: &[u8] = &[3, 7, 8, 5];
const BITSET_NUMBERS_300: &[u8] = &[2, 1, 8, 7, 5];
const BITSET_NUMBERS_301: &[u8] = &[2, 8, 5, 7];
const BITSET_NUMBERS_302: &[u8] = &[1, 5, 7, 8];
const BITSET_NUMBERS_303: &[u8] = &[5, 8, 7];
const BITSET_NUMBERS_304: &[u8] = &[7, 8, 2, 4, 3, 1];
const BITSET_NUMBERS_305: &[u8] = &[7, 2, 3, 8, 4];
const BITSET_NUMBERS_306: &[u8] = &[8, 7, 4, 1, 3];
const BITSET_NUMBERS_307: &[u8] = &[7, 8, 4, 3];
const BITSET_NUMBERS_308: &[u8] = &[7, 4, 8, 2, 1];
const BITSET_NUMBERS_309: &[u8] = &[4, 7, 2, 8];
const BITSET_NUMBERS_310: &[u8] = &[4, 7, 1, 8];
const BITSET_NUMBERS_311: &[u8] = &[8, 4, 7];
const BITSET_NUMBERS_312: &[u8] = &[7, 1, 2, 3, 8];
const BITSET_NUMBERS_313: &[u8] = &[2, 3, 7, 8];
const BITSET_NUMBERS_314: &[u8] = &[1, 7, 8, 3];
const BITSET_NUMBERS_315: &[u8] = &[8, 3, 7];
const BITSET_NUMBERS_316: &[u8] = &[1, 2, 8, 7];
const BITSET_NUMBERS_317: &[u8] = &[2, 7, 8];
const BITSET_NUMBERS_318: &[u8] = &[7, 1, 8];
const BITSET_NUMBERS_319: &[u8] = &[8, 7];
const BITSET_NUMBERS_320: &[u8] = &[3, 2, 5, 8, 1, 6, 4];
const BITSET_NUMBERS_321: &[u8] = &[8, 3, 2, 6, 4, 5];
const BITSET_NUMBERS_322: &[u8] = &[6, 4, 3, 8, 1, 5];
const BITSET_NUMBERS_323: &[u8] = &[3, 6, 8, 4, 5];
const BITSET_NUMBERS_324: &[u8] = &[6, 5, 4, 2, 8, 1];
const BITSET_NUMBERS_325: &[u8] = &[6, 5, 2, 4, 8];
const BITSET_NUMBERS_326: &[u8] = &[6, 1, 8, 4, 5];
const BITSET_NUMBERS_327: &[u8] = &[5, 8, 4, 6];
const BITSET_NUMBERS_328: &[u8] = &[8, 1, 5, 2, 3, 6];
const BITSET_NUMBERS_329: &[u8] = &[3, 6, 5, 2, 8];
const BITSET_NUMBERS_330: &[u8] = &[6, 5, 8, 1, 3];
const BITSET_NUMBERS_331: &[u8] = &[8, 5, 3, 6];
const BITSET_NUMBERS_332: &[u8] = &[5, 1, 6, 2, 8];
const BITSET_NUMBERS_333: &[u8] = &[6, 5, 8, 2];
const BITSET_NUMBERS_334: &[u8] = &[6, 5, 8, 1];
const BITSET_NUMBERS_335: &[u8] = &[8, 5, 6];
const BITSET_NUMBERS_336: &[u8] = &[1, 6, 3, 8, 4, 2];
const BITSET_NUMBERS_337: &[u8] = &[4, 3, 2, 8, 6];
const BITSET_NUMBERS_338: &[u8] = &[8, 4, 3, 6, 1];
const BITSET_NUMBERS_339: &[u8] = &[4, 8, 3, 6];
const BITSET_NUMBERS_340: &[u8] = &[6, 1, 4, 2, 8];
const BITSET_NUMBERS_341: &[u8] = &[6, 8, 2, 4];
const BITSET_NUMBERS_342: &[u8] = &[4, 6, 8, 1];
const BITSET_NUMBERS_343: &[u8] = &[4, 8, 6];
const BITSET_NUMBERS_344: &[u8] = &[8, 6, 3, 1, 2];
const BITSET_NUMBERS_345: &[u8] = &[3, 8, 2, 6];
const BITSET_NUMBERS_346: &[u8] = &[8, 3, 6, 1];
const BITSET_NUMBERS_347: &[u8] = &[8, 3, 6];
const BITSET_NUMBERS_348: &[u8] = &[8, 6, 2, 1];
const BITSET_NUMBERS_349: &[u8] = &[8, 6, 2];
const BITSET_NUMBERS_350: &[u8] = &[1, 8, 6];
const BITSET_NUMBERS_351: &[u8] = &[6, 8];
const BITSET_NUMBERS_352: &[u8] = &[1, 3, 4, 2, 8, 5];
const BITSET_NUMBERS_353: &[u8] = &[4, 3, 8, 2, 5];
const BITSET_NUMBERS_354: &[u8] = &[4, 5, 1, 3, 8];
const BITSET_NUMBERS_355: &[u8] = &[5, 8, 3, 4];
const BITSET_NUMBERS_356: &[u8] = &[2, 1, 4, 5, 8];
const BITSET_NUMBERS_357: &[u8] = &[4, 5, 2, 8];
const BITSET_NUMBERS_358: &[u8] = &[4, 1, 8, 5];
const BITSET_NUMBERS_359: &[u8] = &[5, 4, 8];
const BITSET_NUMBERS_360: &[u8] = &[3, 1, 5, 8, 2];
const BITSET_NUMBERS_361: &[u8] = &[8, 2, 3, 5];
const BITSET_NUMBERS_362: &[u8] = &[3, 8, 1, 5];
const BITSET_NUMBERS_363: &[u8] = &[3, 5, 8];
const BITSET_NUMBERS_364: &[u8] = &[2, 8, 1, 5];
const BITSET_NUMBERS_365: &[u8] = &[8, 2, 5];
const BITSET_NUMBERS_366: &[u8] = &[5, 1, 8];
const BITSET_NUMBERS_367: &[u8] = &[5, 8];
const BITSET_NUMBERS_368: &[u8] = &[4, 1, 3, 8, 2];
const BITSET_NUMBERS_369: &[u8] = &[2, 4, 3, 8];
const BITSET_NUMBERS_370: &[u8] = &[1, 4, 3, 8];
const BITSET_NUMBERS_371: &[u8] = &[8, 3, 4];
const BITSET_NUMBERS_372: &[u8] = &[2, 4, 8, 1];
const BITSET_NUMBERS_373: &[u8] = &[2, 8, 4];
const BITSET_NUMBERS_374: &[u8] = &[8, 1, 4];
const BITSET_NUMBERS_375: &[u8] = &[8, 4];
const BITSET_NUMBERS_376: &[u8] = &[3, 8, 2, 1];
const BITSET_NUMBERS_377: &[u8] = &[8, 3, 2];
const BITSET_NUMBERS_378: &[u8] = &[1, 8, 3];
const BITSET_NUMBERS_379: &[u8] = &[8, 3];
const BITSET_NUMBERS_380: &[u8] = &[1, 2, 8];
const BITSET_NUMBERS_381: &[u8] = &[2, 8];
const BITSET_NUMBERS_382: &[u8] = &[1, 8];
const BITSET_NUMBERS_383: &[u8] = &[8];
const BITSET_NUMBERS_384: &[u8] = &[1, 4, 6, 2, 5, 7, 3];
const BITSET_NUMBERS_385: &[u8] = &[7, 3, 6, 4, 5, 2];
const BITSET_NUMBERS_386: &[u8] = &[3, 4, 1, 6, 5, 7];
const BITSET_NUMBERS_387: &[u8] = &[6, 5, 3, 7, 4];
const BITSET_NUMBERS_388: &[u8] = &[7, 2, 1, 4, 6, 5];
const BITSET_NUMBERS_389: &[u8] = &[7, 4, 6, 2, 5];
const BITSET_NUMBERS_390: &[u8] = &[5, 6, 1, 4, 7];
const BITSET_NUMBERS_391: &[u8] = &[7, 4, 6, 5];
const BITSET_NUMBERS_392: &[u8] = &[2, 6, 1, 5, 7, 3];
const BITSET_NUMBERS_393: &[u8] = &[3, 2, 5, 7, 6];
const BITSET_NUMBERS_394: &[u8] = &[3, 5, 6, 1, 7];
const BITSET_NUMBERS_395: &[u8] = &[3, 7, 6, 5];
const BITSET_NUMBERS_396: &[u8] = &[5, 1, 2, 7, 6];
const BITSET_NUMBERS_397: &[u8] = &[2, 6, 7, 5];
const BITSET_NUMBERS_398: &[u8] = &[5, 6, 7, 1];
const BITSET_NUMBERS_399: &[u8] = &[6, 7, 5];
const BITSET_NUMBERS_400: &[u8] = &[6, 4, 7, 2, 3, 1];
const BITSET_NUMBERS_401: &[u8] = &[7, 3, 4, 2, 6];
const BITSET_NUMBERS_402: &[u8] = &[6, 1, 3, 7, 4];
const BITSET_NUMBERS_403: &[u8] = &[3, 4, 6, 7];
const BITSET_NUMBERS_404: &[u8] = &[1, 2, 6, 7, 4];
const BITSET_NUMBERS_405: &[u8] = &[4, 2, 7, 6];
const BITSET_NUMBERS_406: &[u8] = &[4, 7, 6, 1];
const BITSET_NUMBERS_407: &[u8] = &[4, 6, 7];
const BITSET_NUMBERS_408: &[u8] = &[6, 1, 2, 3, 7];
const BITSET_NUMBERS_409: &[u8] = &[7, 6, 3, 2];
const BITSET_NUMBERS_410: &[u8] = &[3, 1, 6, 7];
const BITSET_NUMBERS_411: &[u8] = &[7, 6, 3];
const BITSET_NUMBERS_412: &[u8] = &[1, 6, 7, 2];
const BITSET_NUMBERS_413: &[u8] = &[6, 2, 7];
const BITSET_NUMBERS_414: &[u8] = &[7, 6, 1];
const BITSET_NUMBERS_415: &[u8] = &[7, 6];
const BITSET_NUMBERS_416: &[u8] = &[7, 3, 4, 1, 2, 5];
const BITSET_NUMBERS_417: &[u8] = &[3, 2, 5, 4, 7];
const BITSET_NUMBERS_418: &[u8] = &[1, 5, 4, 3, 7];
const BITSET_NUMBERS_419: &[u8] = &[4, 7, 5, 3];
const BITSET_NUMBERS_420: &[u8] = &[7, 5, 2, 4, 1];
const BITSET_NUMBERS_421: &[u8] = &[7, 2, 5, 4];
const BITSET_NUMBERS_422: &[u8] = &[5, 1, 4, 7];
const BITSET_NUMBERS_423: &[u8] = &[7, 4, 5];
const BITSET_NUMBERS_424: &[u8] = &[3, 2, 5, 1, 7];
const BITSET_NUMBERS_425: &[u8] = &[2, 5, 3, 7];
const BITSET_NUMBERS_426: &[u8] = &[3, 5, 1, 7];
const BITSET_NUMBERS_427: &[u8] = &[7, 5, 3];
const BITSET_NUMBERS_428: &[u8] = &[2, 1, 7, 5];
const BITSET_NUMBERS_429: &[u8] = &[5, 2, 7];
const BITSET_NUMBERS_430: &[u8] = &[1, 7, 5];
const BITSET_NUMBERS_431: &[u8] = &[5, 7];
const BITSET_NUMBERS_432: &[u8] = &[2, 4, 3, 7, 1];
const BITSET_NUMBERS_433: &[u8] = &[7, 2, 4, 3];
const BITSET_NUMBERS_434: &[u8] = &[1, 4, 3, 7];
const BITSET_NUMBERS_435: &[u8] = &[4, 3, 7];
const BITSET_NUMBERS_436: &[u8] = &[7, 1, 4, 2];
const BITSET_NUMBERS_437: &[u8] = &[2, 7, 4];
const BITSET_NUMBERS_438: &[u8] = &[1, 4, 7];
const BITSET_NUMBERS_439: &[u8] = &[4, 7];
const BITSET_NUMBERS_440: &[u8] = &[2, 3, 1, 7];
const BITSET_NUMBERS_441: &[u8] = &[7, 3, 2];
const BITSET_NUMBERS_442: &[u8] = &[1, 7, 3];
const BITSET_NUMBERS_443: &[u8] = &[3, 7];
const BITSET_NUMBERS_444: &[u8] = &[7, 1, 2];
const BITSET_NUMBERS_445: &[u8] = &[7, 2];
const BITSET_NUMBERS_446: &[u8] = &[7, 1];
const BITSET_NUMBERS_447: &[u8] = &[7];
const BITSET_NUMBERS_448: &[u8] = &[1, 4, 6, 5, 3, 2];
const BITSET_NUMBERS_449: &[u8] = &[2, 6, 5, 4, 3];
const BITSET_NUMBERS_450: &[u8] = &[6, 4, 1, 3, 5];
const BITSET_NUMBERS_451: &[u8] = &[6, 5, 4, 3];
const BITSET_NUMBERS_452: &[u8] = &[6, 1, 4, 2, 5];
const BITSET_NUMBERS_453: &[u8] = &[5, 4, 2, 6];
const BITSET_NUMBERS_454: &[u8] = &[1, 4, 5, 6];
const BITSET_NUMBERS_455: &[u8] = &[6, 4, 5];
const BITSET_NUMBERS_456: &[u8] = &[6, 3, 1, 2, 5];
const BITSET_NUMBERS_457: &[u8] = &[6, 5, 2, 3];
const BITSET_NUMBERS_458: &[u8] = &[1, 6, 3, 5];
const BITSET_NUMBERS_459: &[u8] = &[5, 6, 3];
const BITSET_NUMBERS_460: &[u8] = &[6, 1, 2, 5];
const BITSET_NUMBERS_461: &[u8] = &[5, 6, 2];
const BITSET_NUMBERS_462: &[u8] = &[6, 1, 5];
const BITSET_NUMBERS_463: &[u8] = &[5, 6];
const BITSET_NUMBERS_464: &[u8] = &[2, 4, 6, 3, 1];
const BITSET_NUMBERS_465: &[u8] = &[3, 6, 4, 2];
const BITSET_NUMBERS_466: &[u8] = &[6, 3, 1, 4];
const BITSET_NUMBERS_467: &[u8] = &[3, 4, 6];
const BITSET_NUMBERS_468: &[u8] = &[4, 2, 1, 6];
const BITSET_NUMBERS_469: &[u8] = &[4, 2, 6];
const BITSET_NUMBERS_470: &[u8] = &[1, 6, 4];
const BITSET_NUMBERS_471: &[u8] = &[6, 4];
const BITSET_NUMBERS_472: &[u8] = &[1, 6, 2, 3];
const BITSET_NUMBERS_473: &[u8] = &[2, 3, 6];
const BITSET_NUMBERS_474: &[u8] = &[1, 3, 6];
const BITSET_NUMBERS_475: &[u8] = &[3, 6];
const BITSET_NUMBERS_476: &[u8] = &[6, 1, 2];
const BITSET_NUMBERS_477: &[u8] = &[6, 2];
const BITSET_NUMBERS_478: &[u8] = &[1, 6];
const BITSET_NUMBERS_479: &[u8] = &[6];
const BITSET_NUMBERS_480: &[u8] = &[1, 3, 2, 4, 5];
const BITSET_NUMBERS_481: &[u8] = &[4, 2, 3, 5];
const BITSET_NUMBERS_482: &[u8] = &[1, 3, 4, 5];
const BITSET_NUMBERS_483: &[u8] = &[4, 3, 5];
const BITSET_NUMBERS_484: &[u8] = &[5, 1, 4, 2];
const BITSET_NUMBERS_485: &[u8] = &[4, 2, 5];
const BITSET_NUMBERS_486: &[u8] = &[5, 1, 4];
const BITSET_NUMBERS_487: &[u8] = &[4, 5];
const BITSET_NUMBERS_488: &[u8] = &[5, 3, 1, 2];
const BITSET_NUMBERS_489: &[u8] = &[2, 5, 3];
const BITSET_NUMBERS_490: &[u8] = &[3, 5, 1];
const BITSET_NUMBERS_491: &[u8] = &[3, 5];
const BITSET_NUMBERS_492: &[u8] = &[5, 2, 1];
const BITSET_NUMBERS_493: &[u8] = &[2, 5];
const BITSET_NUMBERS_494: &[u8] = &[1, 5];
const BITSET_NUMBERS_495: &[u8] = &[5];
const BITSET_NUMBERS_496: &[u8] = &[4, 2, 3, 1];
const BITSET_NUMBERS_497: &[u8] = &[4, 3, 2];
const BITSET_NUMBERS_498: &[u8] = &[1, 4, 3];
const BITSET_NUMBERS_499: &[u8] = &[4, 3];
const BITSET_NUMBERS_500: &[u8] = &[1, 4, 2];
const BITSET_NUMBERS_501: &[u8] = &[2, 4];
const BITSET_NUMBERS_502: &[u8] = &[1, 4];
const BITSET_NUMBERS_503: &[u8] = &[4];
const BITSET_NUMBERS_504: &[u8] = &[3, 1, 2];
const BITSET_NUMBERS_505: &[u8] = &[3, 2];
const BITSET_NUMBERS_506: &[u8] = &[1, 3];
const BITSET_NUMBERS_507: &[u8] = &[3];
const BITSET_NUMBERS_508: &[u8] = &[2, 1];
const BITSET_NUMBERS_509: &[u8] = &[2];
const BITSET_NUMBERS_510: &[u8] = &[1];

pub const BITSET_LENGTH: &[u8] = &[
    9, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5,
    4, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4,
    4, 3, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5,
    4, 4, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3,
    4, 3, 3, 2, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5,
    4, 5, 4, 4, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4,
    4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5,
    4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2,
    4, 3, 3, 2, 3, 2, 2, 1, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5,
    4, 6, 5, 5, 4, 5, 4, 4, 3, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4,
    4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5,
    4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3,
    4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5,
    4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4,
    4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5,
    4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
    4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
];

pub const BITSET_ARRAY: &[&[u8]] = &[
    BITSET_NUMBERS_000,
    BITSET_NUMBERS_001,
    BITSET_NUMBERS_002,
    BITSET_NUMBERS_003,
    BITSET_NUMBERS_004,
    BITSET_NUMBERS_005,
    BITSET_NUMBERS_006,
    BITSET_NUMBERS_007,
    BITSET_NUMBERS_008,
    BITSET_NUMBERS_009,
    BITSET_NUMBERS_010,
    BITSET_NUMBERS_011,
    BITSET_NUMBERS_012,
    BITSET_NUMBERS_013,
    BITSET_NUMBERS_014,
    BITSET_NUMBERS_015,
    BITSET_NUMBERS_016,
    BITSET_NUMBERS_017,
    BITSET_NUMBERS_018,
    BITSET_NUMBERS_019,
    BITSET_NUMBERS_020,
    BITSET_NUMBERS_021,
    BITSET_NUMBERS_022,
    BITSET_NUMBERS_023,
    BITSET_NUMBERS_024,
    BITSET_NUMBERS_025,
    BITSET_NUMBERS_026,
    BITSET_NUMBERS_027,
    BITSET_NUMBERS_028,
    BITSET_NUMBERS_029,
    BITSET_NUMBERS_030,
    BITSET_NUMBERS_031,
    BITSET_NUMBERS_032,
    BITSET_NUMBERS_033,
    BITSET_NUMBERS_034,
    BITSET_NUMBERS_035,
    BITSET_NUMBERS_036,
    BITSET_NUMBERS_037,
    BITSET_NUMBERS_038,
    BITSET_NUMBERS_039,
    BITSET_NUMBERS_040,
    BITSET_NUMBERS_041,
    BITSET_NUMBERS_042,
    BITSET_NUMBERS_043,
    BITSET_NUMBERS_044,
    BITSET_NUMBERS_045,
    BITSET_NUMBERS_046,
    BITSET_NUMBERS_047,
    BITSET_NUMBERS_048,
    BITSET_NUMBERS_049,
    BITSET_NUMBERS_050,
    BITSET_NUMBERS_051,
    BITSET_NUMBERS_052,
    BITSET_NUMBERS_053,
    BITSET_NUMBERS_054,
    BITSET_NUMBERS_055,
    BITSET_NUMBERS_056,
    BITSET_NUMBERS_057,
    BITSET_NUMBERS_058,
    BITSET_NUMBERS_059,
    BITSET_NUMBERS_060,
    BITSET_NUMBERS_061,
    BITSET_NUMBERS_062,
    BITSET_NUMBERS_063,
    BITSET_NUMBERS_064,
    BITSET_NUMBERS_065,
    BITSET_NUMBERS_066,
    BITSET_NUMBERS_067,
    BITSET_NUMBERS_068,
    BITSET_NUMBERS_069,
    BITSET_NUMBERS_070,
    BITSET_NUMBERS_071,
    BITSET_NUMBERS_072,
    BITSET_NUMBERS_073,
    BITSET_NUMBERS_074,
    BITSET_NUMBERS_075,
    BITSET_NUMBERS_076,
    BITSET_NUMBERS_077,
    BITSET_NUMBERS_078,
    BITSET_NUMBERS_079,
    BITSET_NUMBERS_080,
    BITSET_NUMBERS_081,
    BITSET_NUMBERS_082,
    BITSET_NUMBERS_083,
    BITSET_NUMBERS_084,
    BITSET_NUMBERS_085,
    BITSET_NUMBERS_086,
    BITSET_NUMBERS_087,
    BITSET_NUMBERS_088,
    BITSET_NUMBERS_089,
    BITSET_NUMBERS_090,
    BITSET_NUMBERS_091,
    BITSET_NUMBERS_092,
    BITSET_NUMBERS_093,
    BITSET_NUMBERS_094,
    BITSET_NUMBERS_095,
    BITSET_NUMBERS_096,
    BITSET_NUMBERS_097,
    BITSET_NUMBERS_098,
    BITSET_NUMBERS_099,
    BITSET_NUMBERS_100,
    BITSET_NUMBERS_101,
    BITSET_NUMBERS_102,
    BITSET_NUMBERS_103,
    BITSET_NUMBERS_104,
    BITSET_NUMBERS_105,
    BITSET_NUMBERS_106,
    BITSET_NUMBERS_107,
    BITSET_NUMBERS_108,
    BITSET_NUMBERS_109,
    BITSET_NUMBERS_110,
    BITSET_NUMBERS_111,
    BITSET_NUMBERS_112,
    BITSET_NUMBERS_113,
    BITSET_NUMBERS_114,
    BITSET_NUMBERS_115,
    BITSET_NUMBERS_116,
    BITSET_NUMBERS_117,
    BITSET_NUMBERS_118,
    BITSET_NUMBERS_119,
    BITSET_NUMBERS_120,
    BITSET_NUMBERS_121,
    BITSET_NUMBERS_122,
    BITSET_NUMBERS_123,
    BITSET_NUMBERS_124,
    BITSET_NUMBERS_125,
    BITSET_NUMBERS_126,
    BITSET_NUMBERS_127,
    BITSET_NUMBERS_128,
    BITSET_NUMBERS_129,
    BITSET_NUMBERS_130,
    BITSET_NUMBERS_131,
    BITSET_NUMBERS_132,
    BITSET_NUMBERS_133,
    BITSET_NUMBERS_134,
    BITSET_NUMBERS_135,
    BITSET_NUMBERS_136,
    BITSET_NUMBERS_137,
    BITSET_NUMBERS_138,
    BITSET_NUMBERS_139,
    BITSET_NUMBERS_140,
    BITSET_NUMBERS_141,
    BITSET_NUMBERS_142,
    BITSET_NUMBERS_143,
    BITSET_NUMBERS_144,
    BITSET_NUMBERS_145,
    BITSET_NUMBERS_146,
    BITSET_NUMBERS_147,
    BITSET_NUMBERS_148,
    BITSET_NUMBERS_149,
    BITSET_NUMBERS_150,
    BITSET_NUMBERS_151,
    BITSET_NUMBERS_152,
    BITSET_NUMBERS_153,
    BITSET_NUMBERS_154,
    BITSET_NUMBERS_155,
    BITSET_NUMBERS_156,
    BITSET_NUMBERS_157,
    BITSET_NUMBERS_158,
    BITSET_NUMBERS_159,
    BITSET_NUMBERS_160,
    BITSET_NUMBERS_161,
    BITSET_NUMBERS_162,
    BITSET_NUMBERS_163,
    BITSET_NUMBERS_164,
    BITSET_NUMBERS_165,
    BITSET_NUMBERS_166,
    BITSET_NUMBERS_167,
    BITSET_NUMBERS_168,
    BITSET_NUMBERS_169,
    BITSET_NUMBERS_170,
    BITSET_NUMBERS_171,
    BITSET_NUMBERS_172,
    BITSET_NUMBERS_173,
    BITSET_NUMBERS_174,
    BITSET_NUMBERS_175,
    BITSET_NUMBERS_176,
    BITSET_NUMBERS_177,
    BITSET_NUMBERS_178,
    BITSET_NUMBERS_179,
    BITSET_NUMBERS_180,
    BITSET_NUMBERS_181,
    BITSET_NUMBERS_182,
    BITSET_NUMBERS_183,
    BITSET_NUMBERS_184,
    BITSET_NUMBERS_185,
    BITSET_NUMBERS_186,
    BITSET_NUMBERS_187,
    BITSET_NUMBERS_188,
    BITSET_NUMBERS_189,
    BITSET_NUMBERS_190,
    BITSET_NUMBERS_191,
    BITSET_NUMBERS_192,
    BITSET_NUMBERS_193,
    BITSET_NUMBERS_194,
    BITSET_NUMBERS_195,
    BITSET_NUMBERS_196,
    BITSET_NUMBERS_197,
    BITSET_NUMBERS_198,
    BITSET_NUMBERS_199,
    BITSET_NUMBERS_200,
    BITSET_NUMBERS_201,
    BITSET_NUMBERS_202,
    BITSET_NUMBERS_203,
    BITSET_NUMBERS_204,
    BITSET_NUMBERS_205,
    BITSET_NUMBERS_206,
    BITSET_NUMBERS_207,
    BITSET_NUMBERS_208,
    BITSET_NUMBERS_209,
    BITSET_NUMBERS_210,
    BITSET_NUMBERS_211,
    BITSET_NUMBERS_212,
    BITSET_NUMBERS_213,
    BITSET_NUMBERS_214,
    BITSET_NUMBERS_215,
    BITSET_NUMBERS_216,
    BITSET_NUMBERS_217,
    BITSET_NUMBERS_218,
    BITSET_NUMBERS_219,
    BITSET_NUMBERS_220,
    BITSET_NUMBERS_221,
    BITSET_NUMBERS_222,
    BITSET_NUMBERS_223,
    BITSET_NUMBERS_224,
    BITSET_NUMBERS_225,
    BITSET_NUMBERS_226,
    BITSET_NUMBERS_227,
    BITSET_NUMBERS_228,
    BITSET_NUMBERS_229,
    BITSET_NUMBERS_230,
    BITSET_NUMBERS_231,
    BITSET_NUMBERS_232,
    BITSET_NUMBERS_233,
    BITSET_NUMBERS_234,
    BITSET_NUMBERS_235,
    BITSET_NUMBERS_236,
    BITSET_NUMBERS_237,
    BITSET_NUMBERS_238,
    BITSET_NUMBERS_239,
    BITSET_NUMBERS_240,
    BITSET_NUMBERS_241,
    BITSET_NUMBERS_242,
    BITSET_NUMBERS_243,
    BITSET_NUMBERS_244,
    BITSET_NUMBERS_245,
    BITSET_NUMBERS_246,
    BITSET_NUMBERS_247,
    BITSET_NUMBERS_248,
    BITSET_NUMBERS_249,
    BITSET_NUMBERS_250,
    BITSET_NUMBERS_251,
    BITSET_NUMBERS_252,
    BITSET_NUMBERS_253,
    BITSET_NUMBERS_254,
    BITSET_NUMBERS_255,
    BITSET_NUMBERS_256,
    BITSET_NUMBERS_257,
    BITSET_NUMBERS_258,
    BITSET_NUMBERS_259,
    BITSET_NUMBERS_260,
    BITSET_NUMBERS_261,
    BITSET_NUMBERS_262,
    BITSET_NUMBERS_263,
    BITSET_NUMBERS_264,
    BITSET_NUMBERS_265,
    BITSET_NUMBERS_266,
    BITSET_NUMBERS_267,
    BITSET_NUMBERS_268,
    BITSET_NUMBERS_269,
    BITSET_NUMBERS_270,
    BITSET_NUMBERS_271,
    BITSET_NUMBERS_272,
    BITSET_NUMBERS_273,
    BITSET_NUMBERS_274,
    BITSET_NUMBERS_275,
    BITSET_NUMBERS_276,
    BITSET_NUMBERS_277,
    BITSET_NUMBERS_278,
    BITSET_NUMBERS_279,
    BITSET_NUMBERS_280,
    BITSET_NUMBERS_281,
    BITSET_NUMBERS_282,
    BITSET_NUMBERS_283,
    BITSET_NUMBERS_284,
    BITSET_NUMBERS_285,
    BITSET_NUMBERS_286,
    BITSET_NUMBERS_287,
    BITSET_NUMBERS_288,
    BITSET_NUMBERS_289,
    BITSET_NUMBERS_290,
    BITSET_NUMBERS_291,
    BITSET_NUMBERS_292,
    BITSET_NUMBERS_293,
    BITSET_NUMBERS_294,
    BITSET_NUMBERS_295,
    BITSET_NUMBERS_296,
    BITSET_NUMBERS_297,
    BITSET_NUMBERS_298,
    BITSET_NUMBERS_299,
    BITSET_NUMBERS_300,
    BITSET_NUMBERS_301,
    BITSET_NUMBERS_302,
    BITSET_NUMBERS_303,
    BITSET_NUMBERS_304,
    BITSET_NUMBERS_305,
    BITSET_NUMBERS_306,
    BITSET_NUMBERS_307,
    BITSET_NUMBERS_308,
    BITSET_NUMBERS_309,
    BITSET_NUMBERS_310,
    BITSET_NUMBERS_311,
    BITSET_NUMBERS_312,
    BITSET_NUMBERS_313,
    BITSET_NUMBERS_314,
    BITSET_NUMBERS_315,
    BITSET_NUMBERS_316,
    BITSET_NUMBERS_317,
    BITSET_NUMBERS_318,
    BITSET_NUMBERS_319,
    BITSET_NUMBERS_320,
    BITSET_NUMBERS_321,
    BITSET_NUMBERS_322,
    BITSET_NUMBERS_323,
    BITSET_NUMBERS_324,
    BITSET_NUMBERS_325,
    BITSET_NUMBERS_326,
    BITSET_NUMBERS_327,
    BITSET_NUMBERS_328,
    BITSET_NUMBERS_329,
    BITSET_NUMBERS_330,
    BITSET_NUMBERS_331,
    BITSET_NUMBERS_332,
    BITSET_NUMBERS_333,
    BITSET_NUMBERS_334,
    BITSET_NUMBERS_335,
    BITSET_NUMBERS_336,
    BITSET_NUMBERS_337,
    BITSET_NUMBERS_338,
    BITSET_NUMBERS_339,
    BITSET_NUMBERS_340,
    BITSET_NUMBERS_341,
    BITSET_NUMBERS_342,
    BITSET_NUMBERS_343,
    BITSET_NUMBERS_344,
    BITSET_NUMBERS_345,
    BITSET_NUMBERS_346,
    BITSET_NUMBERS_347,
    BITSET_NUMBERS_348,
    BITSET_NUMBERS_349,
    BITSET_NUMBERS_350,
    BITSET_NUMBERS_351,
    BITSET_NUMBERS_352,
    BITSET_NUMBERS_353,
    BITSET_NUMBERS_354,
    BITSET_NUMBERS_355,
    BITSET_NUMBERS_356,
    BITSET_NUMBERS_357,
    BITSET_NUMBERS_358,
    BITSET_NUMBERS_359,
    BITSET_NUMBERS_360,
    BITSET_NUMBERS_361,
    BITSET_NUMBERS_362,
    BITSET_NUMBERS_363,
    BITSET_NUMBERS_364,
    BITSET_NUMBERS_365,
    BITSET_NUMBERS_366,
    BITSET_NUMBERS_367,
    BITSET_NUMBERS_368,
    BITSET_NUMBERS_369,
    BITSET_NUMBERS_370,
    BITSET_NUMBERS_371,
    BITSET_NUMBERS_372,
    BITSET_NUMBERS_373,
    BITSET_NUMBERS_374,
    BITSET_NUMBERS_375,
    BITSET_NUMBERS_376,
    BITSET_NUMBERS_377,
    BITSET_NUMBERS_378,
    BITSET_NUMBERS_379,
    BITSET_NUMBERS_380,
    BITSET_NUMBERS_381,
    BITSET_NUMBERS_382,
    BITSET_NUMBERS_383,
    BITSET_NUMBERS_384,
    BITSET_NUMBERS_385,
    BITSET_NUMBERS_386,
    BITSET_NUMBERS_387,
    BITSET_NUMBERS_388,
    BITSET_NUMBERS_389,
    BITSET_NUMBERS_390,
    BITSET_NUMBERS_391,
    BITSET_NUMBERS_392,
    BITSET_NUMBERS_393,
    BITSET_NUMBERS_394,
    BITSET_NUMBERS_395,
    BITSET_NUMBERS_396,
    BITSET_NUMBERS_397,
    BITSET_NUMBERS_398,
    BITSET_NUMBERS_399,
    BITSET_NUMBERS_400,
    BITSET_NUMBERS_401,
    BITSET_NUMBERS_402,
    BITSET_NUMBERS_403,
    BITSET_NUMBERS_404,
    BITSET_NUMBERS_405,
    BITSET_NUMBERS_406,
    BITSET_NUMBERS_407,
    BITSET_NUMBERS_408,
    BITSET_NUMBERS_409,
    BITSET_NUMBERS_410,
    BITSET_NUMBERS_411,
    BITSET_NUMBERS_412,
    BITSET_NUMBERS_413,
    BITSET_NUMBERS_414,
    BITSET_NUMBERS_415,
    BITSET_NUMBERS_416,
    BITSET_NUMBERS_417,
    BITSET_NUMBERS_418,
    BITSET_NUMBERS_419,
    BITSET_NUMBERS_420,
    BITSET_NUMBERS_421,
    BITSET_NUMBERS_422,
    BITSET_NUMBERS_423,
    BITSET_NUMBERS_424,
    BITSET_NUMBERS_425,
    BITSET_NUMBERS_426,
    BITSET_NUMBERS_427,
    BITSET_NUMBERS_428,
    BITSET_NUMBERS_429,
    BITSET_NUMBERS_430,
    BITSET_NUMBERS_431,
    BITSET_NUMBERS_432,
    BITSET_NUMBERS_433,
    BITSET_NUMBERS_434,
    BITSET_NUMBERS_435,
    BITSET_NUMBERS_436,
    BITSET_NUMBERS_437,
    BITSET_NUMBERS_438,
    BITSET_NUMBERS_439,
    BITSET_NUMBERS_440,
    BITSET_NUMBERS_441,
    BITSET_NUMBERS_442,
    BITSET_NUMBERS_443,
    BITSET_NUMBERS_444,
    BITSET_NUMBERS_445,
    BITSET_NUMBERS_446,
    BITSET_NUMBERS_447,
    BITSET_NUMBERS_448,
    BITSET_NUMBERS_449,
    BITSET_NUMBERS_450,
    BITSET_NUMBERS_451,
    BITSET_NUMBERS_452,
    BITSET_NUMBERS_453,
    BITSET_NUMBERS_454,
    BITSET_NUMBERS_455,
    BITSET_NUMBERS_456,
    BITSET_NUMBERS_457,
    BITSET_NUMBERS_458,
    BITSET_NUMBERS_459,
    BITSET_NUMBERS_460,
    BITSET_NUMBERS_461,
    BITSET_NUMBERS_462,
    BITSET_NUMBERS_463,
    BITSET_NUMBERS_464,
    BITSET_NUMBERS_465,
    BITSET_NUMBERS_466,
    BITSET_NUMBERS_467,
    BITSET_NUMBERS_468,
    BITSET_NUMBERS_469,
    BITSET_NUMBERS_470,
    BITSET_NUMBERS_471,
    BITSET_NUMBERS_472,
    BITSET_NUMBERS_473,
    BITSET_NUMBERS_474,
    BITSET_NUMBERS_475,
    BITSET_NUMBERS_476,
    BITSET_NUMBERS_477,
    BITSET_NUMBERS_478,
    BITSET_NUMBERS_479,
    BITSET_NUMBERS_480,
    BITSET_NUMBERS_481,
    BITSET_NUMBERS_482,
    BITSET_NUMBERS_483,
    BITSET_NUMBERS_484,
    BITSET_NUMBERS_485,
    BITSET_NUMBERS_486,
    BITSET_NUMBERS_487,
    BITSET_NUMBERS_488,
    BITSET_NUMBERS_489,
    BITSET_NUMBERS_490,
    BITSET_NUMBERS_491,
    BITSET_NUMBERS_492,
    BITSET_NUMBERS_493,
    BITSET_NUMBERS_494,
    BITSET_NUMBERS_495,
    BITSET_NUMBERS_496,
    BITSET_NUMBERS_497,
    BITSET_NUMBERS_498,
    BITSET_NUMBERS_499,
    BITSET_NUMBERS_500,
    BITSET_NUMBERS_501,
    BITSET_NUMBERS_502,
    BITSET_NUMBERS_503,
    BITSET_NUMBERS_504,
    BITSET_NUMBERS_505,
    BITSET_NUMBERS_506,
    BITSET_NUMBERS_507,
    BITSET_NUMBERS_508,
    BITSET_NUMBERS_509,
    BITSET_NUMBERS_510
];
