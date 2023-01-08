/*
 * sudoku - Sudoku solver for comparison Golang with Scala and Rust
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
package main

const (
	NEW_SUDOKU_SEPARATOR = "Grid"
	CELL_COUNT           = 81
	PUZZLE_SIZE          = 9
	SQUARE_SIZE          = 3
	CHECK_BITS           = 511 // bit pattern 0000...111111111, 9 ones

	PARALLELIZATION_COUNT = 65536
)

// unfortunately golang does not support array constants therefore
// vars are used instead:
var (
	BITSET_NUMBERS_000 = []uint8{4, 7, 8, 3, 5, 1, 9, 6, 2}
	BITSET_NUMBERS_001 = []uint8{8, 6, 7, 5, 9, 4, 2, 3}
	BITSET_NUMBERS_002 = []uint8{3, 4, 5, 1, 9, 6, 7, 8}
	BITSET_NUMBERS_003 = []uint8{7, 8, 9, 5, 4, 6, 3}
	BITSET_NUMBERS_004 = []uint8{9, 1, 8, 6, 7, 5, 4, 2}
	BITSET_NUMBERS_005 = []uint8{8, 6, 5, 7, 4, 2, 9}
	BITSET_NUMBERS_006 = []uint8{6, 9, 4, 5, 8, 7, 1}
	BITSET_NUMBERS_007 = []uint8{9, 7, 4, 6, 8, 5}
	BITSET_NUMBERS_008 = []uint8{1, 8, 9, 3, 2, 5, 7, 6}
	BITSET_NUMBERS_009 = []uint8{3, 7, 2, 5, 9, 8, 6}
	BITSET_NUMBERS_010 = []uint8{7, 6, 9, 3, 1, 8, 5}
	BITSET_NUMBERS_011 = []uint8{7, 8, 3, 6, 5, 9}
	BITSET_NUMBERS_012 = []uint8{8, 2, 6, 7, 9, 1, 5}
	BITSET_NUMBERS_013 = []uint8{7, 5, 9, 8, 6, 2}
	BITSET_NUMBERS_014 = []uint8{9, 8, 7, 1, 5, 6}
	BITSET_NUMBERS_015 = []uint8{6, 8, 7, 9, 5}
	BITSET_NUMBERS_016 = []uint8{8, 2, 9, 7, 1, 3, 6, 4}
	BITSET_NUMBERS_017 = []uint8{3, 4, 7, 8, 2, 6, 9}
	BITSET_NUMBERS_018 = []uint8{6, 1, 4, 3, 9, 7, 8}
	BITSET_NUMBERS_019 = []uint8{9, 3, 7, 4, 6, 8}
	BITSET_NUMBERS_020 = []uint8{7, 2, 6, 9, 4, 1, 8}
	BITSET_NUMBERS_021 = []uint8{4, 9, 6, 2, 7, 8}
	BITSET_NUMBERS_022 = []uint8{7, 8, 4, 1, 6, 9}
	BITSET_NUMBERS_023 = []uint8{9, 4, 8, 7, 6}
	BITSET_NUMBERS_024 = []uint8{7, 9, 8, 2, 3, 6, 1}
	BITSET_NUMBERS_025 = []uint8{3, 6, 9, 8, 7, 2}
	BITSET_NUMBERS_026 = []uint8{3, 6, 1, 8, 9, 7}
	BITSET_NUMBERS_027 = []uint8{8, 3, 7, 6, 9}
	BITSET_NUMBERS_028 = []uint8{8, 1, 2, 7, 9, 6}
	BITSET_NUMBERS_029 = []uint8{6, 8, 9, 7, 2}
	BITSET_NUMBERS_030 = []uint8{1, 8, 9, 7, 6}
	BITSET_NUMBERS_031 = []uint8{8, 9, 7, 6}
	BITSET_NUMBERS_032 = []uint8{8, 7, 3, 1, 2, 5, 9, 4}
	BITSET_NUMBERS_033 = []uint8{9, 3, 5, 4, 2, 7, 8}
	BITSET_NUMBERS_034 = []uint8{7, 3, 8, 4, 9, 5, 1}
	BITSET_NUMBERS_035 = []uint8{9, 4, 3, 7, 5, 8}
	BITSET_NUMBERS_036 = []uint8{2, 7, 8, 4, 9, 1, 5}
	BITSET_NUMBERS_037 = []uint8{4, 7, 5, 8, 2, 9}
	BITSET_NUMBERS_038 = []uint8{5, 7, 9, 8, 1, 4}
	BITSET_NUMBERS_039 = []uint8{5, 8, 9, 7, 4}
	BITSET_NUMBERS_040 = []uint8{7, 5, 8, 9, 3, 1, 2}
	BITSET_NUMBERS_041 = []uint8{9, 8, 3, 7, 5, 2}
	BITSET_NUMBERS_042 = []uint8{5, 3, 9, 7, 8, 1}
	BITSET_NUMBERS_043 = []uint8{5, 3, 9, 8, 7}
	BITSET_NUMBERS_044 = []uint8{5, 2, 9, 7, 1, 8}
	BITSET_NUMBERS_045 = []uint8{5, 8, 9, 2, 7}
	BITSET_NUMBERS_046 = []uint8{1, 5, 7, 9, 8}
	BITSET_NUMBERS_047 = []uint8{7, 8, 9, 5}
	BITSET_NUMBERS_048 = []uint8{7, 9, 8, 2, 1, 4, 3}
	BITSET_NUMBERS_049 = []uint8{8, 9, 3, 7, 4, 2}
	BITSET_NUMBERS_050 = []uint8{1, 3, 4, 7, 8, 9}
	BITSET_NUMBERS_051 = []uint8{9, 4, 7, 3, 8}
	BITSET_NUMBERS_052 = []uint8{7, 4, 8, 2, 1, 9}
	BITSET_NUMBERS_053 = []uint8{4, 2, 7, 8, 9}
	BITSET_NUMBERS_054 = []uint8{1, 9, 7, 8, 4}
	BITSET_NUMBERS_055 = []uint8{4, 9, 8, 7}
	BITSET_NUMBERS_056 = []uint8{2, 9, 1, 3, 8, 7}
	BITSET_NUMBERS_057 = []uint8{2, 8, 9, 7, 3}
	BITSET_NUMBERS_058 = []uint8{8, 9, 1, 7, 3}
	BITSET_NUMBERS_059 = []uint8{3, 7, 9, 8}
	BITSET_NUMBERS_060 = []uint8{8, 7, 1, 9, 2}
	BITSET_NUMBERS_061 = []uint8{2, 7, 8, 9}
	BITSET_NUMBERS_062 = []uint8{1, 8, 7, 9}
	BITSET_NUMBERS_063 = []uint8{9, 7, 8}
	BITSET_NUMBERS_064 = []uint8{6, 5, 3, 8, 2, 4, 1, 9}
	BITSET_NUMBERS_065 = []uint8{9, 2, 5, 3, 6, 8, 4}
	BITSET_NUMBERS_066 = []uint8{5, 4, 6, 8, 1, 9, 3}
	BITSET_NUMBERS_067 = []uint8{6, 8, 4, 5, 9, 3}
	BITSET_NUMBERS_068 = []uint8{1, 8, 5, 2, 6, 9, 4}
	BITSET_NUMBERS_069 = []uint8{6, 9, 5, 2, 8, 4}
	BITSET_NUMBERS_070 = []uint8{8, 9, 5, 6, 1, 4}
	BITSET_NUMBERS_071 = []uint8{9, 8, 6, 5, 4}
	BITSET_NUMBERS_072 = []uint8{2, 5, 9, 3, 1, 6, 8}
	BITSET_NUMBERS_073 = []uint8{5, 6, 8, 3, 2, 9}
	BITSET_NUMBERS_074 = []uint8{6, 5, 8, 9, 3, 1}
	BITSET_NUMBERS_075 = []uint8{3, 9, 5, 6, 8}
	BITSET_NUMBERS_076 = []uint8{1, 2, 6, 5, 9, 8}
	BITSET_NUMBERS_077 = []uint8{5, 8, 9, 6, 2}
	BITSET_NUMBERS_078 = []uint8{5, 9, 1, 8, 6}
	BITSET_NUMBERS_079 = []uint8{5, 8, 9, 6}
	BITSET_NUMBERS_080 = []uint8{8, 1, 2, 6, 9, 3, 4}
	BITSET_NUMBERS_081 = []uint8{3, 4, 8, 6, 2, 9}
	BITSET_NUMBERS_082 = []uint8{3, 8, 4, 1, 6, 9}
	BITSET_NUMBERS_083 = []uint8{9, 6, 3, 8, 4}
	BITSET_NUMBERS_084 = []uint8{9, 1, 4, 6, 2, 8}
	BITSET_NUMBERS_085 = []uint8{2, 8, 6, 9, 4}
	BITSET_NUMBERS_086 = []uint8{6, 9, 8, 1, 4}
	BITSET_NUMBERS_087 = []uint8{9, 4, 8, 6}
	BITSET_NUMBERS_088 = []uint8{1, 2, 6, 9, 3, 8}
	BITSET_NUMBERS_089 = []uint8{3, 8, 9, 6, 2}
	BITSET_NUMBERS_090 = []uint8{9, 3, 6, 1, 8}
	BITSET_NUMBERS_091 = []uint8{8, 3, 9, 6}
	BITSET_NUMBERS_092 = []uint8{1, 6, 2, 9, 8}
	BITSET_NUMBERS_093 = []uint8{8, 6, 2, 9}
	BITSET_NUMBERS_094 = []uint8{9, 1, 8, 6}
	BITSET_NUMBERS_095 = []uint8{8, 9, 6}
	BITSET_NUMBERS_096 = []uint8{1, 2, 5, 8, 9, 3, 4}
	BITSET_NUMBERS_097 = []uint8{9, 5, 2, 8, 3, 4}
	BITSET_NUMBERS_098 = []uint8{3, 1, 5, 4, 8, 9}
	BITSET_NUMBERS_099 = []uint8{8, 5, 9, 3, 4}
	BITSET_NUMBERS_100 = []uint8{9, 1, 5, 8, 4, 2}
	BITSET_NUMBERS_101 = []uint8{8, 2, 5, 9, 4}
	BITSET_NUMBERS_102 = []uint8{8, 4, 1, 9, 5}
	BITSET_NUMBERS_103 = []uint8{5, 4, 9, 8}
	BITSET_NUMBERS_104 = []uint8{2, 8, 5, 1, 9, 3}
	BITSET_NUMBERS_105 = []uint8{2, 8, 3, 9, 5}
	BITSET_NUMBERS_106 = []uint8{1, 9, 5, 8, 3}
	BITSET_NUMBERS_107 = []uint8{9, 5, 3, 8}
	BITSET_NUMBERS_108 = []uint8{5, 2, 8, 9, 1}
	BITSET_NUMBERS_109 = []uint8{8, 5, 9, 2}
	BITSET_NUMBERS_110 = []uint8{9, 1, 5, 8}
	BITSET_NUMBERS_111 = []uint8{9, 5, 8}
	BITSET_NUMBERS_112 = []uint8{2, 9, 4, 1, 8, 3}
	BITSET_NUMBERS_113 = []uint8{3, 2, 4, 9, 8}
	BITSET_NUMBERS_114 = []uint8{8, 3, 4, 9, 1}
	BITSET_NUMBERS_115 = []uint8{9, 8, 4, 3}
	BITSET_NUMBERS_116 = []uint8{9, 8, 1, 2, 4}
	BITSET_NUMBERS_117 = []uint8{4, 2, 9, 8}
	BITSET_NUMBERS_118 = []uint8{9, 1, 8, 4}
	BITSET_NUMBERS_119 = []uint8{9, 8, 4}
	BITSET_NUMBERS_120 = []uint8{3, 9, 8, 1, 2}
	BITSET_NUMBERS_121 = []uint8{8, 2, 3, 9}
	BITSET_NUMBERS_122 = []uint8{8, 3, 1, 9}
	BITSET_NUMBERS_123 = []uint8{3, 8, 9}
	BITSET_NUMBERS_124 = []uint8{1, 9, 2, 8}
	BITSET_NUMBERS_125 = []uint8{2, 9, 8}
	BITSET_NUMBERS_126 = []uint8{8, 1, 9}
	BITSET_NUMBERS_127 = []uint8{8, 9}
	BITSET_NUMBERS_128 = []uint8{7, 5, 1, 2, 3, 4, 9, 6}
	BITSET_NUMBERS_129 = []uint8{7, 2, 5, 4, 9, 6, 3}
	BITSET_NUMBERS_130 = []uint8{4, 7, 1, 3, 9, 6, 5}
	BITSET_NUMBERS_131 = []uint8{3, 6, 7, 9, 4, 5}
	BITSET_NUMBERS_132 = []uint8{9, 6, 7, 2, 4, 5, 1}
	BITSET_NUMBERS_133 = []uint8{5, 7, 4, 6, 2, 9}
	BITSET_NUMBERS_134 = []uint8{6, 4, 5, 9, 1, 7}
	BITSET_NUMBERS_135 = []uint8{4, 5, 6, 9, 7}
	BITSET_NUMBERS_136 = []uint8{1, 2, 7, 6, 3, 9, 5}
	BITSET_NUMBERS_137 = []uint8{7, 5, 3, 9, 2, 6}
	BITSET_NUMBERS_138 = []uint8{6, 7, 9, 3, 1, 5}
	BITSET_NUMBERS_139 = []uint8{5, 7, 3, 9, 6}
	BITSET_NUMBERS_140 = []uint8{5, 2, 1, 6, 9, 7}
	BITSET_NUMBERS_141 = []uint8{9, 7, 5, 6, 2}
	BITSET_NUMBERS_142 = []uint8{6, 1, 7, 5, 9}
	BITSET_NUMBERS_143 = []uint8{5, 7, 9, 6}
	BITSET_NUMBERS_144 = []uint8{4, 9, 2, 6, 7, 3, 1}
	BITSET_NUMBERS_145 = []uint8{3, 4, 7, 2, 6, 9}
	BITSET_NUMBERS_146 = []uint8{3, 1, 6, 7, 9, 4}
	BITSET_NUMBERS_147 = []uint8{4, 3, 7, 6, 9}
	BITSET_NUMBERS_148 = []uint8{4, 7, 9, 2, 6, 1}
	BITSET_NUMBERS_149 = []uint8{6, 9, 4, 2, 7}
	BITSET_NUMBERS_150 = []uint8{7, 4, 9, 6, 1}
	BITSET_NUMBERS_151 = []uint8{6, 7, 4, 9}
	BITSET_NUMBERS_152 = []uint8{7, 2, 1, 6, 3, 9}
	BITSET_NUMBERS_153 = []uint8{3, 9, 6, 2, 7}
	BITSET_NUMBERS_154 = []uint8{3, 6, 9, 7, 1}
	BITSET_NUMBERS_155 = []uint8{9, 6, 3, 7}
	BITSET_NUMBERS_156 = []uint8{9, 2, 1, 6, 7}
	BITSET_NUMBERS_157 = []uint8{9, 7, 2, 6}
	BITSET_NUMBERS_158 = []uint8{9, 7, 1, 6}
	BITSET_NUMBERS_159 = []uint8{9, 7, 6}
	BITSET_NUMBERS_160 = []uint8{2, 9, 4, 1, 7, 3, 5}
	BITSET_NUMBERS_161 = []uint8{2, 5, 3, 7, 9, 4}
	BITSET_NUMBERS_162 = []uint8{9, 7, 3, 5, 1, 4}
	BITSET_NUMBERS_163 = []uint8{9, 4, 7, 5, 3}
	BITSET_NUMBERS_164 = []uint8{9, 4, 1, 2, 7, 5}
	BITSET_NUMBERS_165 = []uint8{4, 9, 7, 2, 5}
	BITSET_NUMBERS_166 = []uint8{4, 7, 9, 1, 5}
	BITSET_NUMBERS_167 = []uint8{9, 5, 7, 4}
	BITSET_NUMBERS_168 = []uint8{5, 1, 9, 2, 7, 3}
	BITSET_NUMBERS_169 = []uint8{9, 3, 2, 5, 7}
	BITSET_NUMBERS_170 = []uint8{7, 5, 9, 3, 1}
	BITSET_NUMBERS_171 = []uint8{3, 9, 5, 7}
	BITSET_NUMBERS_172 = []uint8{5, 7, 2, 1, 9}
	BITSET_NUMBERS_173 = []uint8{7, 9, 5, 2}
	BITSET_NUMBERS_174 = []uint8{9, 7, 5, 1}
	BITSET_NUMBERS_175 = []uint8{5, 9, 7}
	BITSET_NUMBERS_176 = []uint8{1, 3, 2, 4, 9, 7}
	BITSET_NUMBERS_177 = []uint8{2, 4, 7, 9, 3}
	BITSET_NUMBERS_178 = []uint8{7, 4, 1, 3, 9}
	BITSET_NUMBERS_179 = []uint8{9, 7, 4, 3}
	BITSET_NUMBERS_180 = []uint8{2, 1, 4, 9, 7}
	BITSET_NUMBERS_181 = []uint8{4, 9, 7, 2}
	BITSET_NUMBERS_182 = []uint8{9, 4, 1, 7}
	BITSET_NUMBERS_183 = []uint8{7, 9, 4}
	BITSET_NUMBERS_184 = []uint8{7, 3, 9, 1, 2}
	BITSET_NUMBERS_185 = []uint8{2, 7, 9, 3}
	BITSET_NUMBERS_186 = []uint8{1, 3, 9, 7}
	BITSET_NUMBERS_187 = []uint8{3, 9, 7}
	BITSET_NUMBERS_188 = []uint8{2, 1, 9, 7}
	BITSET_NUMBERS_189 = []uint8{2, 7, 9}
	BITSET_NUMBERS_190 = []uint8{1, 9, 7}
	BITSET_NUMBERS_191 = []uint8{9, 7}
	BITSET_NUMBERS_192 = []uint8{6, 9, 3, 5, 4, 1, 2}
	BITSET_NUMBERS_193 = []uint8{9, 3, 6, 4, 2, 5}
	BITSET_NUMBERS_194 = []uint8{1, 5, 3, 4, 9, 6}
	BITSET_NUMBERS_195 = []uint8{5, 3, 9, 4, 6}
	BITSET_NUMBERS_196 = []uint8{4, 6, 9, 1, 5, 2}
	BITSET_NUMBERS_197 = []uint8{9, 4, 6, 5, 2}
	BITSET_NUMBERS_198 = []uint8{9, 1, 5, 4, 6}
	BITSET_NUMBERS_199 = []uint8{6, 9, 4, 5}
	BITSET_NUMBERS_200 = []uint8{6, 3, 5, 9, 1, 2}
	BITSET_NUMBERS_201 = []uint8{3, 5, 2, 9, 6}
	BITSET_NUMBERS_202 = []uint8{5, 3, 6, 1, 9}
	BITSET_NUMBERS_203 = []uint8{6, 5, 3, 9}
	BITSET_NUMBERS_204 = []uint8{9, 2, 1, 6, 5}
	BITSET_NUMBERS_205 = []uint8{9, 5, 2, 6}
	BITSET_NUMBERS_206 = []uint8{6, 1, 9, 5}
	BITSET_NUMBERS_207 = []uint8{5, 6, 9}
	BITSET_NUMBERS_208 = []uint8{4, 9, 2, 1, 3, 6}
	BITSET_NUMBERS_209 = []uint8{2, 9, 4, 3, 6}
	BITSET_NUMBERS_210 = []uint8{4, 6, 3, 1, 9}
	BITSET_NUMBERS_211 = []uint8{3, 4, 6, 9}
	BITSET_NUMBERS_212 = []uint8{4, 2, 6, 1, 9}
	BITSET_NUMBERS_213 = []uint8{2, 6, 4, 9}
	BITSET_NUMBERS_214 = []uint8{4, 6, 9, 1}
	BITSET_NUMBERS_215 = []uint8{4, 9, 6}
	BITSET_NUMBERS_216 = []uint8{6, 1, 9, 2, 3}
	BITSET_NUMBERS_217 = []uint8{9, 6, 3, 2}
	BITSET_NUMBERS_218 = []uint8{9, 3, 6, 1}
	BITSET_NUMBERS_219 = []uint8{9, 6, 3}
	BITSET_NUMBERS_220 = []uint8{6, 2, 9, 1}
	BITSET_NUMBERS_221 = []uint8{2, 9, 6}
	BITSET_NUMBERS_222 = []uint8{1, 6, 9}
	BITSET_NUMBERS_223 = []uint8{6, 9}
	BITSET_NUMBERS_224 = []uint8{4, 5, 2, 9, 1, 3}
	BITSET_NUMBERS_225 = []uint8{4, 9, 3, 5, 2}
	BITSET_NUMBERS_226 = []uint8{9, 3, 1, 4, 5}
	BITSET_NUMBERS_227 = []uint8{4, 5, 9, 3}
	BITSET_NUMBERS_228 = []uint8{2, 1, 4, 5, 9}
	BITSET_NUMBERS_229 = []uint8{5, 9, 2, 4}
	BITSET_NUMBERS_230 = []uint8{1, 9, 5, 4}
	BITSET_NUMBERS_231 = []uint8{5, 9, 4}
	BITSET_NUMBERS_232 = []uint8{2, 9, 1, 5, 3}
	BITSET_NUMBERS_233 = []uint8{5, 2, 9, 3}
	BITSET_NUMBERS_234 = []uint8{1, 3, 9, 5}
	BITSET_NUMBERS_235 = []uint8{9, 5, 3}
	BITSET_NUMBERS_236 = []uint8{9, 2, 5, 1}
	BITSET_NUMBERS_237 = []uint8{9, 2, 5}
	BITSET_NUMBERS_238 = []uint8{9, 5, 1}
	BITSET_NUMBERS_239 = []uint8{5, 9}
	BITSET_NUMBERS_240 = []uint8{3, 9, 4, 1, 2}
	BITSET_NUMBERS_241 = []uint8{2, 9, 3, 4}
	BITSET_NUMBERS_242 = []uint8{3, 4, 1, 9}
	BITSET_NUMBERS_243 = []uint8{3, 9, 4}
	BITSET_NUMBERS_244 = []uint8{4, 2, 1, 9}
	BITSET_NUMBERS_245 = []uint8{9, 4, 2}
	BITSET_NUMBERS_246 = []uint8{1, 9, 4}
	BITSET_NUMBERS_247 = []uint8{9, 4}
	BITSET_NUMBERS_248 = []uint8{3, 1, 2, 9}
	BITSET_NUMBERS_249 = []uint8{2, 9, 3}
	BITSET_NUMBERS_250 = []uint8{1, 9, 3}
	BITSET_NUMBERS_251 = []uint8{9, 3}
	BITSET_NUMBERS_252 = []uint8{2, 9, 1}
	BITSET_NUMBERS_253 = []uint8{2, 9}
	BITSET_NUMBERS_254 = []uint8{9, 1}
	BITSET_NUMBERS_255 = []uint8{9}
	BITSET_NUMBERS_256 = []uint8{1, 3, 6, 2, 5, 4, 8, 7}
	BITSET_NUMBERS_257 = []uint8{5, 4, 8, 3, 6, 2, 7}
	BITSET_NUMBERS_258 = []uint8{3, 4, 1, 5, 7, 6, 8}
	BITSET_NUMBERS_259 = []uint8{4, 6, 8, 7, 5, 3}
	BITSET_NUMBERS_260 = []uint8{7, 8, 2, 4, 1, 6, 5}
	BITSET_NUMBERS_261 = []uint8{2, 7, 8, 5, 6, 4}
	BITSET_NUMBERS_262 = []uint8{8, 5, 4, 6, 7, 1}
	BITSET_NUMBERS_263 = []uint8{6, 5, 7, 4, 8}
	BITSET_NUMBERS_264 = []uint8{7, 3, 5, 8, 6, 1, 2}
	BITSET_NUMBERS_265 = []uint8{6, 7, 3, 8, 2, 5}
	BITSET_NUMBERS_266 = []uint8{6, 1, 5, 3, 7, 8}
	BITSET_NUMBERS_267 = []uint8{8, 7, 6, 3, 5}
	BITSET_NUMBERS_268 = []uint8{5, 2, 8, 1, 6, 7}
	BITSET_NUMBERS_269 = []uint8{5, 8, 7, 6, 2}
	BITSET_NUMBERS_270 = []uint8{1, 7, 8, 6, 5}
	BITSET_NUMBERS_271 = []uint8{7, 6, 5, 8}
	BITSET_NUMBERS_272 = []uint8{7, 1, 3, 4, 6, 2, 8}
	BITSET_NUMBERS_273 = []uint8{7, 3, 6, 8, 4, 2}
	BITSET_NUMBERS_274 = []uint8{3, 6, 1, 7, 8, 4}
	BITSET_NUMBERS_275 = []uint8{3, 8, 4, 7, 6}
	BITSET_NUMBERS_276 = []uint8{7, 8, 6, 2, 1, 4}
	BITSET_NUMBERS_277 = []uint8{6, 8, 4, 2, 7}
	BITSET_NUMBERS_278 = []uint8{6, 8, 1, 4, 7}
	BITSET_NUMBERS_279 = []uint8{7, 6, 4, 8}
	BITSET_NUMBERS_280 = []uint8{1, 2, 7, 6, 3, 8}
	BITSET_NUMBERS_281 = []uint8{7, 3, 6, 8, 2}
	BITSET_NUMBERS_282 = []uint8{6, 1, 8, 7, 3}
	BITSET_NUMBERS_283 = []uint8{8, 6, 7, 3}
	BITSET_NUMBERS_284 = []uint8{7, 2, 1, 6, 8}
	BITSET_NUMBERS_285 = []uint8{2, 8, 6, 7}
	BITSET_NUMBERS_286 = []uint8{8, 1, 7, 6}
	BITSET_NUMBERS_287 = []uint8{6, 7, 8}
	BITSET_NUMBERS_288 = []uint8{8, 7, 2, 4, 3, 1, 5}
	BITSET_NUMBERS_289 = []uint8{4, 5, 7, 2, 3, 8}
	BITSET_NUMBERS_290 = []uint8{3, 8, 5, 1, 7, 4}
	BITSET_NUMBERS_291 = []uint8{4, 3, 8, 5, 7}
	BITSET_NUMBERS_292 = []uint8{7, 2, 1, 8, 4, 5}
	BITSET_NUMBERS_293 = []uint8{4, 7, 5, 8, 2}
	BITSET_NUMBERS_294 = []uint8{5, 1, 7, 4, 8}
	BITSET_NUMBERS_295 = []uint8{5, 7, 4, 8}
	BITSET_NUMBERS_296 = []uint8{7, 3, 5, 1, 2, 8}
	BITSET_NUMBERS_297 = []uint8{7, 8, 2, 3, 5}
	BITSET_NUMBERS_298 = []uint8{7, 1, 3, 8, 5}
	BITSET_NUMBERS_299 = []uint8{3, 7, 8, 5}
	BITSET_NUMBERS_300 = []uint8{2, 1, 8, 7, 5}
	BITSET_NUMBERS_301 = []uint8{2, 8, 5, 7}
	BITSET_NUMBERS_302 = []uint8{1, 5, 7, 8}
	BITSET_NUMBERS_303 = []uint8{5, 8, 7}
	BITSET_NUMBERS_304 = []uint8{7, 8, 2, 4, 3, 1}
	BITSET_NUMBERS_305 = []uint8{7, 2, 3, 8, 4}
	BITSET_NUMBERS_306 = []uint8{8, 7, 4, 1, 3}
	BITSET_NUMBERS_307 = []uint8{7, 8, 4, 3}
	BITSET_NUMBERS_308 = []uint8{7, 4, 8, 2, 1}
	BITSET_NUMBERS_309 = []uint8{4, 7, 2, 8}
	BITSET_NUMBERS_310 = []uint8{4, 7, 1, 8}
	BITSET_NUMBERS_311 = []uint8{8, 4, 7}
	BITSET_NUMBERS_312 = []uint8{7, 1, 2, 3, 8}
	BITSET_NUMBERS_313 = []uint8{2, 3, 7, 8}
	BITSET_NUMBERS_314 = []uint8{1, 7, 8, 3}
	BITSET_NUMBERS_315 = []uint8{8, 3, 7}
	BITSET_NUMBERS_316 = []uint8{1, 2, 8, 7}
	BITSET_NUMBERS_317 = []uint8{2, 7, 8}
	BITSET_NUMBERS_318 = []uint8{7, 1, 8}
	BITSET_NUMBERS_319 = []uint8{8, 7}
	BITSET_NUMBERS_320 = []uint8{3, 2, 5, 8, 1, 6, 4}
	BITSET_NUMBERS_321 = []uint8{8, 3, 2, 6, 4, 5}
	BITSET_NUMBERS_322 = []uint8{6, 4, 3, 8, 1, 5}
	BITSET_NUMBERS_323 = []uint8{3, 6, 8, 4, 5}
	BITSET_NUMBERS_324 = []uint8{6, 5, 4, 2, 8, 1}
	BITSET_NUMBERS_325 = []uint8{6, 5, 2, 4, 8}
	BITSET_NUMBERS_326 = []uint8{6, 1, 8, 4, 5}
	BITSET_NUMBERS_327 = []uint8{5, 8, 4, 6}
	BITSET_NUMBERS_328 = []uint8{8, 1, 5, 2, 3, 6}
	BITSET_NUMBERS_329 = []uint8{3, 6, 5, 2, 8}
	BITSET_NUMBERS_330 = []uint8{6, 5, 8, 1, 3}
	BITSET_NUMBERS_331 = []uint8{8, 5, 3, 6}
	BITSET_NUMBERS_332 = []uint8{5, 1, 6, 2, 8}
	BITSET_NUMBERS_333 = []uint8{6, 5, 8, 2}
	BITSET_NUMBERS_334 = []uint8{6, 5, 8, 1}
	BITSET_NUMBERS_335 = []uint8{8, 5, 6}
	BITSET_NUMBERS_336 = []uint8{1, 6, 3, 8, 4, 2}
	BITSET_NUMBERS_337 = []uint8{4, 3, 2, 8, 6}
	BITSET_NUMBERS_338 = []uint8{8, 4, 3, 6, 1}
	BITSET_NUMBERS_339 = []uint8{4, 8, 3, 6}
	BITSET_NUMBERS_340 = []uint8{6, 1, 4, 2, 8}
	BITSET_NUMBERS_341 = []uint8{6, 8, 2, 4}
	BITSET_NUMBERS_342 = []uint8{4, 6, 8, 1}
	BITSET_NUMBERS_343 = []uint8{4, 8, 6}
	BITSET_NUMBERS_344 = []uint8{8, 6, 3, 1, 2}
	BITSET_NUMBERS_345 = []uint8{3, 8, 2, 6}
	BITSET_NUMBERS_346 = []uint8{8, 3, 6, 1}
	BITSET_NUMBERS_347 = []uint8{8, 3, 6}
	BITSET_NUMBERS_348 = []uint8{8, 6, 2, 1}
	BITSET_NUMBERS_349 = []uint8{8, 6, 2}
	BITSET_NUMBERS_350 = []uint8{1, 8, 6}
	BITSET_NUMBERS_351 = []uint8{6, 8}
	BITSET_NUMBERS_352 = []uint8{1, 3, 4, 2, 8, 5}
	BITSET_NUMBERS_353 = []uint8{4, 3, 8, 2, 5}
	BITSET_NUMBERS_354 = []uint8{4, 5, 1, 3, 8}
	BITSET_NUMBERS_355 = []uint8{5, 8, 3, 4}
	BITSET_NUMBERS_356 = []uint8{2, 1, 4, 5, 8}
	BITSET_NUMBERS_357 = []uint8{4, 5, 2, 8}
	BITSET_NUMBERS_358 = []uint8{4, 1, 8, 5}
	BITSET_NUMBERS_359 = []uint8{5, 4, 8}
	BITSET_NUMBERS_360 = []uint8{3, 1, 5, 8, 2}
	BITSET_NUMBERS_361 = []uint8{8, 2, 3, 5}
	BITSET_NUMBERS_362 = []uint8{3, 8, 1, 5}
	BITSET_NUMBERS_363 = []uint8{3, 5, 8}
	BITSET_NUMBERS_364 = []uint8{2, 8, 1, 5}
	BITSET_NUMBERS_365 = []uint8{8, 2, 5}
	BITSET_NUMBERS_366 = []uint8{5, 1, 8}
	BITSET_NUMBERS_367 = []uint8{5, 8}
	BITSET_NUMBERS_368 = []uint8{4, 1, 3, 8, 2}
	BITSET_NUMBERS_369 = []uint8{2, 4, 3, 8}
	BITSET_NUMBERS_370 = []uint8{1, 4, 3, 8}
	BITSET_NUMBERS_371 = []uint8{8, 3, 4}
	BITSET_NUMBERS_372 = []uint8{2, 4, 8, 1}
	BITSET_NUMBERS_373 = []uint8{2, 8, 4}
	BITSET_NUMBERS_374 = []uint8{8, 1, 4}
	BITSET_NUMBERS_375 = []uint8{8, 4}
	BITSET_NUMBERS_376 = []uint8{3, 8, 2, 1}
	BITSET_NUMBERS_377 = []uint8{8, 3, 2}
	BITSET_NUMBERS_378 = []uint8{1, 8, 3}
	BITSET_NUMBERS_379 = []uint8{8, 3}
	BITSET_NUMBERS_380 = []uint8{1, 2, 8}
	BITSET_NUMBERS_381 = []uint8{2, 8}
	BITSET_NUMBERS_382 = []uint8{1, 8}
	BITSET_NUMBERS_383 = []uint8{8}
	BITSET_NUMBERS_384 = []uint8{1, 4, 6, 2, 5, 7, 3}
	BITSET_NUMBERS_385 = []uint8{7, 3, 6, 4, 5, 2}
	BITSET_NUMBERS_386 = []uint8{3, 4, 1, 6, 5, 7}
	BITSET_NUMBERS_387 = []uint8{6, 5, 3, 7, 4}
	BITSET_NUMBERS_388 = []uint8{7, 2, 1, 4, 6, 5}
	BITSET_NUMBERS_389 = []uint8{7, 4, 6, 2, 5}
	BITSET_NUMBERS_390 = []uint8{5, 6, 1, 4, 7}
	BITSET_NUMBERS_391 = []uint8{7, 4, 6, 5}
	BITSET_NUMBERS_392 = []uint8{2, 6, 1, 5, 7, 3}
	BITSET_NUMBERS_393 = []uint8{3, 2, 5, 7, 6}
	BITSET_NUMBERS_394 = []uint8{3, 5, 6, 1, 7}
	BITSET_NUMBERS_395 = []uint8{3, 7, 6, 5}
	BITSET_NUMBERS_396 = []uint8{5, 1, 2, 7, 6}
	BITSET_NUMBERS_397 = []uint8{2, 6, 7, 5}
	BITSET_NUMBERS_398 = []uint8{5, 6, 7, 1}
	BITSET_NUMBERS_399 = []uint8{6, 7, 5}
	BITSET_NUMBERS_400 = []uint8{6, 4, 7, 2, 3, 1}
	BITSET_NUMBERS_401 = []uint8{7, 3, 4, 2, 6}
	BITSET_NUMBERS_402 = []uint8{6, 1, 3, 7, 4}
	BITSET_NUMBERS_403 = []uint8{3, 4, 6, 7}
	BITSET_NUMBERS_404 = []uint8{1, 2, 6, 7, 4}
	BITSET_NUMBERS_405 = []uint8{4, 2, 7, 6}
	BITSET_NUMBERS_406 = []uint8{4, 7, 6, 1}
	BITSET_NUMBERS_407 = []uint8{4, 6, 7}
	BITSET_NUMBERS_408 = []uint8{6, 1, 2, 3, 7}
	BITSET_NUMBERS_409 = []uint8{7, 6, 3, 2}
	BITSET_NUMBERS_410 = []uint8{3, 1, 6, 7}
	BITSET_NUMBERS_411 = []uint8{7, 6, 3}
	BITSET_NUMBERS_412 = []uint8{1, 6, 7, 2}
	BITSET_NUMBERS_413 = []uint8{6, 2, 7}
	BITSET_NUMBERS_414 = []uint8{7, 6, 1}
	BITSET_NUMBERS_415 = []uint8{7, 6}
	BITSET_NUMBERS_416 = []uint8{7, 3, 4, 1, 2, 5}
	BITSET_NUMBERS_417 = []uint8{3, 2, 5, 4, 7}
	BITSET_NUMBERS_418 = []uint8{1, 5, 4, 3, 7}
	BITSET_NUMBERS_419 = []uint8{4, 7, 5, 3}
	BITSET_NUMBERS_420 = []uint8{7, 5, 2, 4, 1}
	BITSET_NUMBERS_421 = []uint8{7, 2, 5, 4}
	BITSET_NUMBERS_422 = []uint8{5, 1, 4, 7}
	BITSET_NUMBERS_423 = []uint8{7, 4, 5}
	BITSET_NUMBERS_424 = []uint8{3, 2, 5, 1, 7}
	BITSET_NUMBERS_425 = []uint8{2, 5, 3, 7}
	BITSET_NUMBERS_426 = []uint8{3, 5, 1, 7}
	BITSET_NUMBERS_427 = []uint8{7, 5, 3}
	BITSET_NUMBERS_428 = []uint8{2, 1, 7, 5}
	BITSET_NUMBERS_429 = []uint8{5, 2, 7}
	BITSET_NUMBERS_430 = []uint8{1, 7, 5}
	BITSET_NUMBERS_431 = []uint8{5, 7}
	BITSET_NUMBERS_432 = []uint8{2, 4, 3, 7, 1}
	BITSET_NUMBERS_433 = []uint8{7, 2, 4, 3}
	BITSET_NUMBERS_434 = []uint8{1, 4, 3, 7}
	BITSET_NUMBERS_435 = []uint8{4, 3, 7}
	BITSET_NUMBERS_436 = []uint8{7, 1, 4, 2}
	BITSET_NUMBERS_437 = []uint8{2, 7, 4}
	BITSET_NUMBERS_438 = []uint8{1, 4, 7}
	BITSET_NUMBERS_439 = []uint8{4, 7}
	BITSET_NUMBERS_440 = []uint8{2, 3, 1, 7}
	BITSET_NUMBERS_441 = []uint8{7, 3, 2}
	BITSET_NUMBERS_442 = []uint8{1, 7, 3}
	BITSET_NUMBERS_443 = []uint8{3, 7}
	BITSET_NUMBERS_444 = []uint8{7, 1, 2}
	BITSET_NUMBERS_445 = []uint8{7, 2}
	BITSET_NUMBERS_446 = []uint8{7, 1}
	BITSET_NUMBERS_447 = []uint8{7}
	BITSET_NUMBERS_448 = []uint8{1, 4, 6, 5, 3, 2}
	BITSET_NUMBERS_449 = []uint8{2, 6, 5, 4, 3}
	BITSET_NUMBERS_450 = []uint8{6, 4, 1, 3, 5}
	BITSET_NUMBERS_451 = []uint8{6, 5, 4, 3}
	BITSET_NUMBERS_452 = []uint8{6, 1, 4, 2, 5}
	BITSET_NUMBERS_453 = []uint8{5, 4, 2, 6}
	BITSET_NUMBERS_454 = []uint8{1, 4, 5, 6}
	BITSET_NUMBERS_455 = []uint8{6, 4, 5}
	BITSET_NUMBERS_456 = []uint8{6, 3, 1, 2, 5}
	BITSET_NUMBERS_457 = []uint8{6, 5, 2, 3}
	BITSET_NUMBERS_458 = []uint8{1, 6, 3, 5}
	BITSET_NUMBERS_459 = []uint8{5, 6, 3}
	BITSET_NUMBERS_460 = []uint8{6, 1, 2, 5}
	BITSET_NUMBERS_461 = []uint8{5, 6, 2}
	BITSET_NUMBERS_462 = []uint8{6, 1, 5}
	BITSET_NUMBERS_463 = []uint8{5, 6}
	BITSET_NUMBERS_464 = []uint8{2, 4, 6, 3, 1}
	BITSET_NUMBERS_465 = []uint8{3, 6, 4, 2}
	BITSET_NUMBERS_466 = []uint8{6, 3, 1, 4}
	BITSET_NUMBERS_467 = []uint8{3, 4, 6}
	BITSET_NUMBERS_468 = []uint8{4, 2, 1, 6}
	BITSET_NUMBERS_469 = []uint8{4, 2, 6}
	BITSET_NUMBERS_470 = []uint8{1, 6, 4}
	BITSET_NUMBERS_471 = []uint8{6, 4}
	BITSET_NUMBERS_472 = []uint8{1, 6, 2, 3}
	BITSET_NUMBERS_473 = []uint8{2, 3, 6}
	BITSET_NUMBERS_474 = []uint8{1, 3, 6}
	BITSET_NUMBERS_475 = []uint8{3, 6}
	BITSET_NUMBERS_476 = []uint8{6, 1, 2}
	BITSET_NUMBERS_477 = []uint8{6, 2}
	BITSET_NUMBERS_478 = []uint8{1, 6}
	BITSET_NUMBERS_479 = []uint8{6}
	BITSET_NUMBERS_480 = []uint8{1, 3, 2, 4, 5}
	BITSET_NUMBERS_481 = []uint8{4, 2, 3, 5}
	BITSET_NUMBERS_482 = []uint8{1, 3, 4, 5}
	BITSET_NUMBERS_483 = []uint8{4, 3, 5}
	BITSET_NUMBERS_484 = []uint8{5, 1, 4, 2}
	BITSET_NUMBERS_485 = []uint8{4, 2, 5}
	BITSET_NUMBERS_486 = []uint8{5, 1, 4}
	BITSET_NUMBERS_487 = []uint8{4, 5}
	BITSET_NUMBERS_488 = []uint8{5, 3, 1, 2}
	BITSET_NUMBERS_489 = []uint8{2, 5, 3}
	BITSET_NUMBERS_490 = []uint8{3, 5, 1}
	BITSET_NUMBERS_491 = []uint8{3, 5}
	BITSET_NUMBERS_492 = []uint8{5, 2, 1}
	BITSET_NUMBERS_493 = []uint8{2, 5}
	BITSET_NUMBERS_494 = []uint8{1, 5}
	BITSET_NUMBERS_495 = []uint8{5}
	BITSET_NUMBERS_496 = []uint8{4, 2, 3, 1}
	BITSET_NUMBERS_497 = []uint8{4, 3, 2}
	BITSET_NUMBERS_498 = []uint8{1, 4, 3}
	BITSET_NUMBERS_499 = []uint8{4, 3}
	BITSET_NUMBERS_500 = []uint8{1, 4, 2}
	BITSET_NUMBERS_501 = []uint8{2, 4}
	BITSET_NUMBERS_502 = []uint8{1, 4}
	BITSET_NUMBERS_503 = []uint8{4}
	BITSET_NUMBERS_504 = []uint8{3, 1, 2}
	BITSET_NUMBERS_505 = []uint8{3, 2}
	BITSET_NUMBERS_506 = []uint8{1, 3}
	BITSET_NUMBERS_507 = []uint8{3}
	BITSET_NUMBERS_508 = []uint8{2, 1}
	BITSET_NUMBERS_509 = []uint8{2}
	BITSET_NUMBERS_510 = []uint8{1}

	BITSET_LENGTH = []uint8{
		9, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4,
		8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
		8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
		7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
		8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
		7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
		7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
		6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
		8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 4, 7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
		7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
		7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
		6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
		7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
		6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
		6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
		5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
	}

	BITSET_ARRAY = [][]uint8{
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
		BITSET_NUMBERS_510,
	}
)