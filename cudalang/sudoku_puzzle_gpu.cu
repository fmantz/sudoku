/*
 * sudoku - Sudoku solver for comparison Scala / Rust and CUDA
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
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>

#define CELL_COUNT   81
#define PUZZLE_SIZE   9
#define SQUARE_SIZE   3
#define NEW_SUDOKU_SEPARATOR "Grid"

typedef struct {
    bool my_is_solvable;
    bool my_is_solved;
    char puzzle[CELL_COUNT];
} SudokuPuzzleData;

// all possible precomputed numbers for a field.
// numbers are assigned according to bitvectors.
__constant__ char BITSET_NUMBERS_000[] = {4, 7, 8, 3, 5, 1, 9, 6, 2};
__constant__ char BITSET_NUMBERS_001[] = {8, 6, 7, 5, 9, 4, 2, 3};
__constant__ char BITSET_NUMBERS_002[] = {3, 4, 5, 1, 9, 6, 7, 8};
__constant__ char BITSET_NUMBERS_003[] = {7, 8, 9, 5, 4, 6, 3};
__constant__ char BITSET_NUMBERS_004[] = {9, 1, 8, 6, 7, 5, 4, 2};
__constant__ char BITSET_NUMBERS_005[] = {8, 6, 5, 7, 4, 2, 9};
__constant__ char BITSET_NUMBERS_006[] = {6, 9, 4, 5, 8, 7, 1};
__constant__ char BITSET_NUMBERS_007[] = {9, 7, 4, 6, 8, 5};
__constant__ char BITSET_NUMBERS_008[] = {1, 8, 9, 3, 2, 5, 7, 6};
__constant__ char BITSET_NUMBERS_009[] = {3, 7, 2, 5, 9, 8, 6};
__constant__ char BITSET_NUMBERS_010[] = {7, 6, 9, 3, 1, 8, 5};
__constant__ char BITSET_NUMBERS_011[] = {7, 8, 3, 6, 5, 9};
__constant__ char BITSET_NUMBERS_012[] = {8, 2, 6, 7, 9, 1, 5};
__constant__ char BITSET_NUMBERS_013[] = {7, 5, 9, 8, 6, 2};
__constant__ char BITSET_NUMBERS_014[] = {9, 8, 7, 1, 5, 6};
__constant__ char BITSET_NUMBERS_015[] = {6, 8, 7, 9, 5};
__constant__ char BITSET_NUMBERS_016[] = {8, 2, 9, 7, 1, 3, 6, 4};
__constant__ char BITSET_NUMBERS_017[] = {3, 4, 7, 8, 2, 6, 9};
__constant__ char BITSET_NUMBERS_018[] = {6, 1, 4, 3, 9, 7, 8};
__constant__ char BITSET_NUMBERS_019[] = {9, 3, 7, 4, 6, 8};
__constant__ char BITSET_NUMBERS_020[] = {7, 2, 6, 9, 4, 1, 8};
__constant__ char BITSET_NUMBERS_021[] = {4, 9, 6, 2, 7, 8};
__constant__ char BITSET_NUMBERS_022[] = {7, 8, 4, 1, 6, 9};
__constant__ char BITSET_NUMBERS_023[] = {9, 4, 8, 7, 6};
__constant__ char BITSET_NUMBERS_024[] = {7, 9, 8, 2, 3, 6, 1};
__constant__ char BITSET_NUMBERS_025[] = {3, 6, 9, 8, 7, 2};
__constant__ char BITSET_NUMBERS_026[] = {3, 6, 1, 8, 9, 7};
__constant__ char BITSET_NUMBERS_027[] = {8, 3, 7, 6, 9};
__constant__ char BITSET_NUMBERS_028[] = {8, 1, 2, 7, 9, 6};
__constant__ char BITSET_NUMBERS_029[] = {6, 8, 9, 7, 2};
__constant__ char BITSET_NUMBERS_030[] = {1, 8, 9, 7, 6};
__constant__ char BITSET_NUMBERS_031[] = {8, 9, 7, 6};
__constant__ char BITSET_NUMBERS_032[] = {8, 7, 3, 1, 2, 5, 9, 4};
__constant__ char BITSET_NUMBERS_033[] = {9, 3, 5, 4, 2, 7, 8};
__constant__ char BITSET_NUMBERS_034[] = {7, 3, 8, 4, 9, 5, 1};
__constant__ char BITSET_NUMBERS_035[] = {9, 4, 3, 7, 5, 8};
__constant__ char BITSET_NUMBERS_036[] = {2, 7, 8, 4, 9, 1, 5};
__constant__ char BITSET_NUMBERS_037[] = {4, 7, 5, 8, 2, 9};
__constant__ char BITSET_NUMBERS_038[] = {5, 7, 9, 8, 1, 4};
__constant__ char BITSET_NUMBERS_039[] = {5, 8, 9, 7, 4};
__constant__ char BITSET_NUMBERS_040[] = {7, 5, 8, 9, 3, 1, 2};
__constant__ char BITSET_NUMBERS_041[] = {9, 8, 3, 7, 5, 2};
__constant__ char BITSET_NUMBERS_042[] = {5, 3, 9, 7, 8, 1};
__constant__ char BITSET_NUMBERS_043[] = {5, 3, 9, 8, 7};
__constant__ char BITSET_NUMBERS_044[] = {5, 2, 9, 7, 1, 8};
__constant__ char BITSET_NUMBERS_045[] = {5, 8, 9, 2, 7};
__constant__ char BITSET_NUMBERS_046[] = {1, 5, 7, 9, 8};
__constant__ char BITSET_NUMBERS_047[] = {7, 8, 9, 5};
__constant__ char BITSET_NUMBERS_048[] = {7, 9, 8, 2, 1, 4, 3};
__constant__ char BITSET_NUMBERS_049[] = {8, 9, 3, 7, 4, 2};
__constant__ char BITSET_NUMBERS_050[] = {1, 3, 4, 7, 8, 9};
__constant__ char BITSET_NUMBERS_051[] = {9, 4, 7, 3, 8};
__constant__ char BITSET_NUMBERS_052[] = {7, 4, 8, 2, 1, 9};
__constant__ char BITSET_NUMBERS_053[] = {4, 2, 7, 8, 9};
__constant__ char BITSET_NUMBERS_054[] = {1, 9, 7, 8, 4};
__constant__ char BITSET_NUMBERS_055[] = {4, 9, 8, 7};
__constant__ char BITSET_NUMBERS_056[] = {2, 9, 1, 3, 8, 7};
__constant__ char BITSET_NUMBERS_057[] = {2, 8, 9, 7, 3};
__constant__ char BITSET_NUMBERS_058[] = {8, 9, 1, 7, 3};
__constant__ char BITSET_NUMBERS_059[] = {3, 7, 9, 8};
__constant__ char BITSET_NUMBERS_060[] = {8, 7, 1, 9, 2};
__constant__ char BITSET_NUMBERS_061[] = {2, 7, 8, 9};
__constant__ char BITSET_NUMBERS_062[] = {1, 8, 7, 9};
__constant__ char BITSET_NUMBERS_063[] = {9, 7, 8};
__constant__ char BITSET_NUMBERS_064[] = {6, 5, 3, 8, 2, 4, 1, 9};
__constant__ char BITSET_NUMBERS_065[] = {9, 2, 5, 3, 6, 8, 4};
__constant__ char BITSET_NUMBERS_066[] = {5, 4, 6, 8, 1, 9, 3};
__constant__ char BITSET_NUMBERS_067[] = {6, 8, 4, 5, 9, 3};
__constant__ char BITSET_NUMBERS_068[] = {1, 8, 5, 2, 6, 9, 4};
__constant__ char BITSET_NUMBERS_069[] = {6, 9, 5, 2, 8, 4};
__constant__ char BITSET_NUMBERS_070[] = {8, 9, 5, 6, 1, 4};
__constant__ char BITSET_NUMBERS_071[] = {9, 8, 6, 5, 4};
__constant__ char BITSET_NUMBERS_072[] = {2, 5, 9, 3, 1, 6, 8};
__constant__ char BITSET_NUMBERS_073[] = {5, 6, 8, 3, 2, 9};
__constant__ char BITSET_NUMBERS_074[] = {6, 5, 8, 9, 3, 1};
__constant__ char BITSET_NUMBERS_075[] = {3, 9, 5, 6, 8};
__constant__ char BITSET_NUMBERS_076[] = {1, 2, 6, 5, 9, 8};
__constant__ char BITSET_NUMBERS_077[] = {5, 8, 9, 6, 2};
__constant__ char BITSET_NUMBERS_078[] = {5, 9, 1, 8, 6};
__constant__ char BITSET_NUMBERS_079[] = {5, 8, 9, 6};
__constant__ char BITSET_NUMBERS_080[] = {8, 1, 2, 6, 9, 3, 4};
__constant__ char BITSET_NUMBERS_081[] = {3, 4, 8, 6, 2, 9};
__constant__ char BITSET_NUMBERS_082[] = {3, 8, 4, 1, 6, 9};
__constant__ char BITSET_NUMBERS_083[] = {9, 6, 3, 8, 4};
__constant__ char BITSET_NUMBERS_084[] = {9, 1, 4, 6, 2, 8};
__constant__ char BITSET_NUMBERS_085[] = {2, 8, 6, 9, 4};
__constant__ char BITSET_NUMBERS_086[] = {6, 9, 8, 1, 4};
__constant__ char BITSET_NUMBERS_087[] = {9, 4, 8, 6};
__constant__ char BITSET_NUMBERS_088[] = {1, 2, 6, 9, 3, 8};
__constant__ char BITSET_NUMBERS_089[] = {3, 8, 9, 6, 2};
__constant__ char BITSET_NUMBERS_090[] = {9, 3, 6, 1, 8};
__constant__ char BITSET_NUMBERS_091[] = {8, 3, 9, 6};
__constant__ char BITSET_NUMBERS_092[] = {1, 6, 2, 9, 8};
__constant__ char BITSET_NUMBERS_093[] = {8, 6, 2, 9};
__constant__ char BITSET_NUMBERS_094[] = {9, 1, 8, 6};
__constant__ char BITSET_NUMBERS_095[] = {8, 9, 6};
__constant__ char BITSET_NUMBERS_096[] = {1, 2, 5, 8, 9, 3, 4};
__constant__ char BITSET_NUMBERS_097[] = {9, 5, 2, 8, 3, 4};
__constant__ char BITSET_NUMBERS_098[] = {3, 1, 5, 4, 8, 9};
__constant__ char BITSET_NUMBERS_099[] = {8, 5, 9, 3, 4};
__constant__ char BITSET_NUMBERS_100[] = {9, 1, 5, 8, 4, 2};
__constant__ char BITSET_NUMBERS_101[] = {8, 2, 5, 9, 4};
__constant__ char BITSET_NUMBERS_102[] = {8, 4, 1, 9, 5};
__constant__ char BITSET_NUMBERS_103[] = {5, 4, 9, 8};
__constant__ char BITSET_NUMBERS_104[] = {2, 8, 5, 1, 9, 3};
__constant__ char BITSET_NUMBERS_105[] = {2, 8, 3, 9, 5};
__constant__ char BITSET_NUMBERS_106[] = {1, 9, 5, 8, 3};
__constant__ char BITSET_NUMBERS_107[] = {9, 5, 3, 8};
__constant__ char BITSET_NUMBERS_108[] = {5, 2, 8, 9, 1};
__constant__ char BITSET_NUMBERS_109[] = {8, 5, 9, 2};
__constant__ char BITSET_NUMBERS_110[] = {9, 1, 5, 8};
__constant__ char BITSET_NUMBERS_111[] = {9, 5, 8};
__constant__ char BITSET_NUMBERS_112[] = {2, 9, 4, 1, 8, 3};
__constant__ char BITSET_NUMBERS_113[] = {3, 2, 4, 9, 8};
__constant__ char BITSET_NUMBERS_114[] = {8, 3, 4, 9, 1};
__constant__ char BITSET_NUMBERS_115[] = {9, 8, 4, 3};
__constant__ char BITSET_NUMBERS_116[] = {9, 8, 1, 2, 4};
__constant__ char BITSET_NUMBERS_117[] = {4, 2, 9, 8};
__constant__ char BITSET_NUMBERS_118[] = {9, 1, 8, 4};
__constant__ char BITSET_NUMBERS_119[] = {9, 8, 4};
__constant__ char BITSET_NUMBERS_120[] = {3, 9, 8, 1, 2};
__constant__ char BITSET_NUMBERS_121[] = {8, 2, 3, 9};
__constant__ char BITSET_NUMBERS_122[] = {8, 3, 1, 9};
__constant__ char BITSET_NUMBERS_123[] = {3, 8, 9};
__constant__ char BITSET_NUMBERS_124[] = {1, 9, 2, 8};
__constant__ char BITSET_NUMBERS_125[] = {2, 9, 8};
__constant__ char BITSET_NUMBERS_126[] = {8, 1, 9};
__constant__ char BITSET_NUMBERS_127[] = {8, 9};
__constant__ char BITSET_NUMBERS_128[] = {7, 5, 1, 2, 3, 4, 9, 6};
__constant__ char BITSET_NUMBERS_129[] = {7, 2, 5, 4, 9, 6, 3};
__constant__ char BITSET_NUMBERS_130[] = {4, 7, 1, 3, 9, 6, 5};
__constant__ char BITSET_NUMBERS_131[] = {3, 6, 7, 9, 4, 5};
__constant__ char BITSET_NUMBERS_132[] = {9, 6, 7, 2, 4, 5, 1};
__constant__ char BITSET_NUMBERS_133[] = {5, 7, 4, 6, 2, 9};
__constant__ char BITSET_NUMBERS_134[] = {6, 4, 5, 9, 1, 7};
__constant__ char BITSET_NUMBERS_135[] = {4, 5, 6, 9, 7};
__constant__ char BITSET_NUMBERS_136[] = {1, 2, 7, 6, 3, 9, 5};
__constant__ char BITSET_NUMBERS_137[] = {7, 5, 3, 9, 2, 6};
__constant__ char BITSET_NUMBERS_138[] = {6, 7, 9, 3, 1, 5};
__constant__ char BITSET_NUMBERS_139[] = {5, 7, 3, 9, 6};
__constant__ char BITSET_NUMBERS_140[] = {5, 2, 1, 6, 9, 7};
__constant__ char BITSET_NUMBERS_141[] = {9, 7, 5, 6, 2};
__constant__ char BITSET_NUMBERS_142[] = {6, 1, 7, 5, 9};
__constant__ char BITSET_NUMBERS_143[] = {5, 7, 9, 6};
__constant__ char BITSET_NUMBERS_144[] = {4, 9, 2, 6, 7, 3, 1};
__constant__ char BITSET_NUMBERS_145[] = {3, 4, 7, 2, 6, 9};
__constant__ char BITSET_NUMBERS_146[] = {3, 1, 6, 7, 9, 4};
__constant__ char BITSET_NUMBERS_147[] = {4, 3, 7, 6, 9};
__constant__ char BITSET_NUMBERS_148[] = {4, 7, 9, 2, 6, 1};
__constant__ char BITSET_NUMBERS_149[] = {6, 9, 4, 2, 7};
__constant__ char BITSET_NUMBERS_150[] = {7, 4, 9, 6, 1};
__constant__ char BITSET_NUMBERS_151[] = {6, 7, 4, 9};
__constant__ char BITSET_NUMBERS_152[] = {7, 2, 1, 6, 3, 9};
__constant__ char BITSET_NUMBERS_153[] = {3, 9, 6, 2, 7};
__constant__ char BITSET_NUMBERS_154[] = {3, 6, 9, 7, 1};
__constant__ char BITSET_NUMBERS_155[] = {9, 6, 3, 7};
__constant__ char BITSET_NUMBERS_156[] = {9, 2, 1, 6, 7};
__constant__ char BITSET_NUMBERS_157[] = {9, 7, 2, 6};
__constant__ char BITSET_NUMBERS_158[] = {9, 7, 1, 6};
__constant__ char BITSET_NUMBERS_159[] = {9, 7, 6};
__constant__ char BITSET_NUMBERS_160[] = {2, 9, 4, 1, 7, 3, 5};
__constant__ char BITSET_NUMBERS_161[] = {2, 5, 3, 7, 9, 4};
__constant__ char BITSET_NUMBERS_162[] = {9, 7, 3, 5, 1, 4};
__constant__ char BITSET_NUMBERS_163[] = {9, 4, 7, 5, 3};
__constant__ char BITSET_NUMBERS_164[] = {9, 4, 1, 2, 7, 5};
__constant__ char BITSET_NUMBERS_165[] = {4, 9, 7, 2, 5};
__constant__ char BITSET_NUMBERS_166[] = {4, 7, 9, 1, 5};
__constant__ char BITSET_NUMBERS_167[] = {9, 5, 7, 4};
__constant__ char BITSET_NUMBERS_168[] = {5, 1, 9, 2, 7, 3};
__constant__ char BITSET_NUMBERS_169[] = {9, 3, 2, 5, 7};
__constant__ char BITSET_NUMBERS_170[] = {7, 5, 9, 3, 1};
__constant__ char BITSET_NUMBERS_171[] = {3, 9, 5, 7};
__constant__ char BITSET_NUMBERS_172[] = {5, 7, 2, 1, 9};
__constant__ char BITSET_NUMBERS_173[] = {7, 9, 5, 2};
__constant__ char BITSET_NUMBERS_174[] = {9, 7, 5, 1};
__constant__ char BITSET_NUMBERS_175[] = {5, 9, 7};
__constant__ char BITSET_NUMBERS_176[] = {1, 3, 2, 4, 9, 7};
__constant__ char BITSET_NUMBERS_177[] = {2, 4, 7, 9, 3};
__constant__ char BITSET_NUMBERS_178[] = {7, 4, 1, 3, 9};
__constant__ char BITSET_NUMBERS_179[] = {9, 7, 4, 3};
__constant__ char BITSET_NUMBERS_180[] = {2, 1, 4, 9, 7};
__constant__ char BITSET_NUMBERS_181[] = {4, 9, 7, 2};
__constant__ char BITSET_NUMBERS_182[] = {9, 4, 1, 7};
__constant__ char BITSET_NUMBERS_183[] = {7, 9, 4};
__constant__ char BITSET_NUMBERS_184[] = {7, 3, 9, 1, 2};
__constant__ char BITSET_NUMBERS_185[] = {2, 7, 9, 3};
__constant__ char BITSET_NUMBERS_186[] = {1, 3, 9, 7};
__constant__ char BITSET_NUMBERS_187[] = {3, 9, 7};
__constant__ char BITSET_NUMBERS_188[] = {2, 1, 9, 7};
__constant__ char BITSET_NUMBERS_189[] = {2, 7, 9};
__constant__ char BITSET_NUMBERS_190[] = {1, 9, 7};
__constant__ char BITSET_NUMBERS_191[] = {9, 7};
__constant__ char BITSET_NUMBERS_192[] = {6, 9, 3, 5, 4, 1, 2};
__constant__ char BITSET_NUMBERS_193[] = {9, 3, 6, 4, 2, 5};
__constant__ char BITSET_NUMBERS_194[] = {1, 5, 3, 4, 9, 6};
__constant__ char BITSET_NUMBERS_195[] = {5, 3, 9, 4, 6};
__constant__ char BITSET_NUMBERS_196[] = {4, 6, 9, 1, 5, 2};
__constant__ char BITSET_NUMBERS_197[] = {9, 4, 6, 5, 2};
__constant__ char BITSET_NUMBERS_198[] = {9, 1, 5, 4, 6};
__constant__ char BITSET_NUMBERS_199[] = {6, 9, 4, 5};
__constant__ char BITSET_NUMBERS_200[] = {6, 3, 5, 9, 1, 2};
__constant__ char BITSET_NUMBERS_201[] = {3, 5, 2, 9, 6};
__constant__ char BITSET_NUMBERS_202[] = {5, 3, 6, 1, 9};
__constant__ char BITSET_NUMBERS_203[] = {6, 5, 3, 9};
__constant__ char BITSET_NUMBERS_204[] = {9, 2, 1, 6, 5};
__constant__ char BITSET_NUMBERS_205[] = {9, 5, 2, 6};
__constant__ char BITSET_NUMBERS_206[] = {6, 1, 9, 5};
__constant__ char BITSET_NUMBERS_207[] = {5, 6, 9};
__constant__ char BITSET_NUMBERS_208[] = {4, 9, 2, 1, 3, 6};
__constant__ char BITSET_NUMBERS_209[] = {2, 9, 4, 3, 6};
__constant__ char BITSET_NUMBERS_210[] = {4, 6, 3, 1, 9};
__constant__ char BITSET_NUMBERS_211[] = {3, 4, 6, 9};
__constant__ char BITSET_NUMBERS_212[] = {4, 2, 6, 1, 9};
__constant__ char BITSET_NUMBERS_213[] = {2, 6, 4, 9};
__constant__ char BITSET_NUMBERS_214[] = {4, 6, 9, 1};
__constant__ char BITSET_NUMBERS_215[] = {4, 9, 6};
__constant__ char BITSET_NUMBERS_216[] = {6, 1, 9, 2, 3};
__constant__ char BITSET_NUMBERS_217[] = {9, 6, 3, 2};
__constant__ char BITSET_NUMBERS_218[] = {9, 3, 6, 1};
__constant__ char BITSET_NUMBERS_219[] = {9, 6, 3};
__constant__ char BITSET_NUMBERS_220[] = {6, 2, 9, 1};
__constant__ char BITSET_NUMBERS_221[] = {2, 9, 6};
__constant__ char BITSET_NUMBERS_222[] = {1, 6, 9};
__constant__ char BITSET_NUMBERS_223[] = {6, 9};
__constant__ char BITSET_NUMBERS_224[] = {4, 5, 2, 9, 1, 3};
__constant__ char BITSET_NUMBERS_225[] = {4, 9, 3, 5, 2};
__constant__ char BITSET_NUMBERS_226[] = {9, 3, 1, 4, 5};
__constant__ char BITSET_NUMBERS_227[] = {4, 5, 9, 3};
__constant__ char BITSET_NUMBERS_228[] = {2, 1, 4, 5, 9};
__constant__ char BITSET_NUMBERS_229[] = {5, 9, 2, 4};
__constant__ char BITSET_NUMBERS_230[] = {1, 9, 5, 4};
__constant__ char BITSET_NUMBERS_231[] = {5, 9, 4};
__constant__ char BITSET_NUMBERS_232[] = {2, 9, 1, 5, 3};
__constant__ char BITSET_NUMBERS_233[] = {5, 2, 9, 3};
__constant__ char BITSET_NUMBERS_234[] = {1, 3, 9, 5};
__constant__ char BITSET_NUMBERS_235[] = {9, 5, 3};
__constant__ char BITSET_NUMBERS_236[] = {9, 2, 5, 1};
__constant__ char BITSET_NUMBERS_237[] = {9, 2, 5};
__constant__ char BITSET_NUMBERS_238[] = {9, 5, 1};
__constant__ char BITSET_NUMBERS_239[] = {5, 9};
__constant__ char BITSET_NUMBERS_240[] = {3, 9, 4, 1, 2};
__constant__ char BITSET_NUMBERS_241[] = {2, 9, 3, 4};
__constant__ char BITSET_NUMBERS_242[] = {3, 4, 1, 9};
__constant__ char BITSET_NUMBERS_243[] = {3, 9, 4};
__constant__ char BITSET_NUMBERS_244[] = {4, 2, 1, 9};
__constant__ char BITSET_NUMBERS_245[] = {9, 4, 2};
__constant__ char BITSET_NUMBERS_246[] = {1, 9, 4};
__constant__ char BITSET_NUMBERS_247[] = {9, 4};
__constant__ char BITSET_NUMBERS_248[] = {3, 1, 2, 9};
__constant__ char BITSET_NUMBERS_249[] = {2, 9, 3};
__constant__ char BITSET_NUMBERS_250[] = {1, 9, 3};
__constant__ char BITSET_NUMBERS_251[] = {9, 3};
__constant__ char BITSET_NUMBERS_252[] = {2, 9, 1};
__constant__ char BITSET_NUMBERS_253[] = {2, 9};
__constant__ char BITSET_NUMBERS_254[] = {9, 1};
__constant__ char BITSET_NUMBERS_255[] = {9};
__constant__ char BITSET_NUMBERS_256[] = {1, 3, 6, 2, 5, 4, 8, 7};
__constant__ char BITSET_NUMBERS_257[] = {5, 4, 8, 3, 6, 2, 7};
__constant__ char BITSET_NUMBERS_258[] = {3, 4, 1, 5, 7, 6, 8};
__constant__ char BITSET_NUMBERS_259[] = {4, 6, 8, 7, 5, 3};
__constant__ char BITSET_NUMBERS_260[] = {7, 8, 2, 4, 1, 6, 5};
__constant__ char BITSET_NUMBERS_261[] = {2, 7, 8, 5, 6, 4};
__constant__ char BITSET_NUMBERS_262[] = {8, 5, 4, 6, 7, 1};
__constant__ char BITSET_NUMBERS_263[] = {6, 5, 7, 4, 8};
__constant__ char BITSET_NUMBERS_264[] = {7, 3, 5, 8, 6, 1, 2};
__constant__ char BITSET_NUMBERS_265[] = {6, 7, 3, 8, 2, 5};
__constant__ char BITSET_NUMBERS_266[] = {6, 1, 5, 3, 7, 8};
__constant__ char BITSET_NUMBERS_267[] = {8, 7, 6, 3, 5};
__constant__ char BITSET_NUMBERS_268[] = {5, 2, 8, 1, 6, 7};
__constant__ char BITSET_NUMBERS_269[] = {5, 8, 7, 6, 2};
__constant__ char BITSET_NUMBERS_270[] = {1, 7, 8, 6, 5};
__constant__ char BITSET_NUMBERS_271[] = {7, 6, 5, 8};
__constant__ char BITSET_NUMBERS_272[] = {7, 1, 3, 4, 6, 2, 8};
__constant__ char BITSET_NUMBERS_273[] = {7, 3, 6, 8, 4, 2};
__constant__ char BITSET_NUMBERS_274[] = {3, 6, 1, 7, 8, 4};
__constant__ char BITSET_NUMBERS_275[] = {3, 8, 4, 7, 6};
__constant__ char BITSET_NUMBERS_276[] = {7, 8, 6, 2, 1, 4};
__constant__ char BITSET_NUMBERS_277[] = {6, 8, 4, 2, 7};
__constant__ char BITSET_NUMBERS_278[] = {6, 8, 1, 4, 7};
__constant__ char BITSET_NUMBERS_279[] = {7, 6, 4, 8};
__constant__ char BITSET_NUMBERS_280[] = {1, 2, 7, 6, 3, 8};
__constant__ char BITSET_NUMBERS_281[] = {7, 3, 6, 8, 2};
__constant__ char BITSET_NUMBERS_282[] = {6, 1, 8, 7, 3};
__constant__ char BITSET_NUMBERS_283[] = {8, 6, 7, 3};
__constant__ char BITSET_NUMBERS_284[] = {7, 2, 1, 6, 8};
__constant__ char BITSET_NUMBERS_285[] = {2, 8, 6, 7};
__constant__ char BITSET_NUMBERS_286[] = {8, 1, 7, 6};
__constant__ char BITSET_NUMBERS_287[] = {6, 7, 8};
__constant__ char BITSET_NUMBERS_288[] = {8, 7, 2, 4, 3, 1, 5};
__constant__ char BITSET_NUMBERS_289[] = {4, 5, 7, 2, 3, 8};
__constant__ char BITSET_NUMBERS_290[] = {3, 8, 5, 1, 7, 4};
__constant__ char BITSET_NUMBERS_291[] = {4, 3, 8, 5, 7};
__constant__ char BITSET_NUMBERS_292[] = {7, 2, 1, 8, 4, 5};
__constant__ char BITSET_NUMBERS_293[] = {4, 7, 5, 8, 2};
__constant__ char BITSET_NUMBERS_294[] = {5, 1, 7, 4, 8};
__constant__ char BITSET_NUMBERS_295[] = {5, 7, 4, 8};
__constant__ char BITSET_NUMBERS_296[] = {7, 3, 5, 1, 2, 8};
__constant__ char BITSET_NUMBERS_297[] = {7, 8, 2, 3, 5};
__constant__ char BITSET_NUMBERS_298[] = {7, 1, 3, 8, 5};
__constant__ char BITSET_NUMBERS_299[] = {3, 7, 8, 5};
__constant__ char BITSET_NUMBERS_300[] = {2, 1, 8, 7, 5};
__constant__ char BITSET_NUMBERS_301[] = {2, 8, 5, 7};
__constant__ char BITSET_NUMBERS_302[] = {1, 5, 7, 8};
__constant__ char BITSET_NUMBERS_303[] = {5, 8, 7};
__constant__ char BITSET_NUMBERS_304[] = {7, 8, 2, 4, 3, 1};
__constant__ char BITSET_NUMBERS_305[] = {7, 2, 3, 8, 4};
__constant__ char BITSET_NUMBERS_306[] = {8, 7, 4, 1, 3};
__constant__ char BITSET_NUMBERS_307[] = {7, 8, 4, 3};
__constant__ char BITSET_NUMBERS_308[] = {7, 4, 8, 2, 1};
__constant__ char BITSET_NUMBERS_309[] = {4, 7, 2, 8};
__constant__ char BITSET_NUMBERS_310[] = {4, 7, 1, 8};
__constant__ char BITSET_NUMBERS_311[] = {8, 4, 7};
__constant__ char BITSET_NUMBERS_312[] = {7, 1, 2, 3, 8};
__constant__ char BITSET_NUMBERS_313[] = {2, 3, 7, 8};
__constant__ char BITSET_NUMBERS_314[] = {1, 7, 8, 3};
__constant__ char BITSET_NUMBERS_315[] = {8, 3, 7};
__constant__ char BITSET_NUMBERS_316[] = {1, 2, 8, 7};
__constant__ char BITSET_NUMBERS_317[] = {2, 7, 8};
__constant__ char BITSET_NUMBERS_318[] = {7, 1, 8};
__constant__ char BITSET_NUMBERS_319[] = {8, 7};
__constant__ char BITSET_NUMBERS_320[] = {3, 2, 5, 8, 1, 6, 4};
__constant__ char BITSET_NUMBERS_321[] = {8, 3, 2, 6, 4, 5};
__constant__ char BITSET_NUMBERS_322[] = {6, 4, 3, 8, 1, 5};
__constant__ char BITSET_NUMBERS_323[] = {3, 6, 8, 4, 5};
__constant__ char BITSET_NUMBERS_324[] = {6, 5, 4, 2, 8, 1};
__constant__ char BITSET_NUMBERS_325[] = {6, 5, 2, 4, 8};
__constant__ char BITSET_NUMBERS_326[] = {6, 1, 8, 4, 5};
__constant__ char BITSET_NUMBERS_327[] = {5, 8, 4, 6};
__constant__ char BITSET_NUMBERS_328[] = {8, 1, 5, 2, 3, 6};
__constant__ char BITSET_NUMBERS_329[] = {3, 6, 5, 2, 8};
__constant__ char BITSET_NUMBERS_330[] = {6, 5, 8, 1, 3};
__constant__ char BITSET_NUMBERS_331[] = {8, 5, 3, 6};
__constant__ char BITSET_NUMBERS_332[] = {5, 1, 6, 2, 8};
__constant__ char BITSET_NUMBERS_333[] = {6, 5, 8, 2};
__constant__ char BITSET_NUMBERS_334[] = {6, 5, 8, 1};
__constant__ char BITSET_NUMBERS_335[] = {8, 5, 6};
__constant__ char BITSET_NUMBERS_336[] = {1, 6, 3, 8, 4, 2};
__constant__ char BITSET_NUMBERS_337[] = {4, 3, 2, 8, 6};
__constant__ char BITSET_NUMBERS_338[] = {8, 4, 3, 6, 1};
__constant__ char BITSET_NUMBERS_339[] = {4, 8, 3, 6};
__constant__ char BITSET_NUMBERS_340[] = {6, 1, 4, 2, 8};
__constant__ char BITSET_NUMBERS_341[] = {6, 8, 2, 4};
__constant__ char BITSET_NUMBERS_342[] = {4, 6, 8, 1};
__constant__ char BITSET_NUMBERS_343[] = {4, 8, 6};
__constant__ char BITSET_NUMBERS_344[] = {8, 6, 3, 1, 2};
__constant__ char BITSET_NUMBERS_345[] = {3, 8, 2, 6};
__constant__ char BITSET_NUMBERS_346[] = {8, 3, 6, 1};
__constant__ char BITSET_NUMBERS_347[] = {8, 3, 6};
__constant__ char BITSET_NUMBERS_348[] = {8, 6, 2, 1};
__constant__ char BITSET_NUMBERS_349[] = {8, 6, 2};
__constant__ char BITSET_NUMBERS_350[] = {1, 8, 6};
__constant__ char BITSET_NUMBERS_351[] = {6, 8};
__constant__ char BITSET_NUMBERS_352[] = {1, 3, 4, 2, 8, 5};
__constant__ char BITSET_NUMBERS_353[] = {4, 3, 8, 2, 5};
__constant__ char BITSET_NUMBERS_354[] = {4, 5, 1, 3, 8};
__constant__ char BITSET_NUMBERS_355[] = {5, 8, 3, 4};
__constant__ char BITSET_NUMBERS_356[] = {2, 1, 4, 5, 8};
__constant__ char BITSET_NUMBERS_357[] = {4, 5, 2, 8};
__constant__ char BITSET_NUMBERS_358[] = {4, 1, 8, 5};
__constant__ char BITSET_NUMBERS_359[] = {5, 4, 8};
__constant__ char BITSET_NUMBERS_360[] = {3, 1, 5, 8, 2};
__constant__ char BITSET_NUMBERS_361[] = {8, 2, 3, 5};
__constant__ char BITSET_NUMBERS_362[] = {3, 8, 1, 5};
__constant__ char BITSET_NUMBERS_363[] = {3, 5, 8};
__constant__ char BITSET_NUMBERS_364[] = {2, 8, 1, 5};
__constant__ char BITSET_NUMBERS_365[] = {8, 2, 5};
__constant__ char BITSET_NUMBERS_366[] = {5, 1, 8};
__constant__ char BITSET_NUMBERS_367[] = {5, 8};
__constant__ char BITSET_NUMBERS_368[] = {4, 1, 3, 8, 2};
__constant__ char BITSET_NUMBERS_369[] = {2, 4, 3, 8};
__constant__ char BITSET_NUMBERS_370[] = {1, 4, 3, 8};
__constant__ char BITSET_NUMBERS_371[] = {8, 3, 4};
__constant__ char BITSET_NUMBERS_372[] = {2, 4, 8, 1};
__constant__ char BITSET_NUMBERS_373[] = {2, 8, 4};
__constant__ char BITSET_NUMBERS_374[] = {8, 1, 4};
__constant__ char BITSET_NUMBERS_375[] = {8, 4};
__constant__ char BITSET_NUMBERS_376[] = {3, 8, 2, 1};
__constant__ char BITSET_NUMBERS_377[] = {8, 3, 2};
__constant__ char BITSET_NUMBERS_378[] = {1, 8, 3};
__constant__ char BITSET_NUMBERS_379[] = {8, 3};
__constant__ char BITSET_NUMBERS_380[] = {1, 2, 8};
__constant__ char BITSET_NUMBERS_381[] = {2, 8};
__constant__ char BITSET_NUMBERS_382[] = {1, 8};
__constant__ char BITSET_NUMBERS_383[] = {8};
__constant__ char BITSET_NUMBERS_384[] = {1, 4, 6, 2, 5, 7, 3};
__constant__ char BITSET_NUMBERS_385[] = {7, 3, 6, 4, 5, 2};
__constant__ char BITSET_NUMBERS_386[] = {3, 4, 1, 6, 5, 7};
__constant__ char BITSET_NUMBERS_387[] = {6, 5, 3, 7, 4};
__constant__ char BITSET_NUMBERS_388[] = {7, 2, 1, 4, 6, 5};
__constant__ char BITSET_NUMBERS_389[] = {7, 4, 6, 2, 5};
__constant__ char BITSET_NUMBERS_390[] = {5, 6, 1, 4, 7};
__constant__ char BITSET_NUMBERS_391[] = {7, 4, 6, 5};
__constant__ char BITSET_NUMBERS_392[] = {2, 6, 1, 5, 7, 3};
__constant__ char BITSET_NUMBERS_393[] = {3, 2, 5, 7, 6};
__constant__ char BITSET_NUMBERS_394[] = {3, 5, 6, 1, 7};
__constant__ char BITSET_NUMBERS_395[] = {3, 7, 6, 5};
__constant__ char BITSET_NUMBERS_396[] = {5, 1, 2, 7, 6};
__constant__ char BITSET_NUMBERS_397[] = {2, 6, 7, 5};
__constant__ char BITSET_NUMBERS_398[] = {5, 6, 7, 1};
__constant__ char BITSET_NUMBERS_399[] = {6, 7, 5};
__constant__ char BITSET_NUMBERS_400[] = {6, 4, 7, 2, 3, 1};
__constant__ char BITSET_NUMBERS_401[] = {7, 3, 4, 2, 6};
__constant__ char BITSET_NUMBERS_402[] = {6, 1, 3, 7, 4};
__constant__ char BITSET_NUMBERS_403[] = {3, 4, 6, 7};
__constant__ char BITSET_NUMBERS_404[] = {1, 2, 6, 7, 4};
__constant__ char BITSET_NUMBERS_405[] = {4, 2, 7, 6};
__constant__ char BITSET_NUMBERS_406[] = {4, 7, 6, 1};
__constant__ char BITSET_NUMBERS_407[] = {4, 6, 7};
__constant__ char BITSET_NUMBERS_408[] = {6, 1, 2, 3, 7};
__constant__ char BITSET_NUMBERS_409[] = {7, 6, 3, 2};
__constant__ char BITSET_NUMBERS_410[] = {3, 1, 6, 7};
__constant__ char BITSET_NUMBERS_411[] = {7, 6, 3};
__constant__ char BITSET_NUMBERS_412[] = {1, 6, 7, 2};
__constant__ char BITSET_NUMBERS_413[] = {6, 2, 7};
__constant__ char BITSET_NUMBERS_414[] = {7, 6, 1};
__constant__ char BITSET_NUMBERS_415[] = {7, 6};
__constant__ char BITSET_NUMBERS_416[] = {7, 3, 4, 1, 2, 5};
__constant__ char BITSET_NUMBERS_417[] = {3, 2, 5, 4, 7};
__constant__ char BITSET_NUMBERS_418[] = {1, 5, 4, 3, 7};
__constant__ char BITSET_NUMBERS_419[] = {4, 7, 5, 3};
__constant__ char BITSET_NUMBERS_420[] = {7, 5, 2, 4, 1};
__constant__ char BITSET_NUMBERS_421[] = {7, 2, 5, 4};
__constant__ char BITSET_NUMBERS_422[] = {5, 1, 4, 7};
__constant__ char BITSET_NUMBERS_423[] = {7, 4, 5};
__constant__ char BITSET_NUMBERS_424[] = {3, 2, 5, 1, 7};
__constant__ char BITSET_NUMBERS_425[] = {2, 5, 3, 7};
__constant__ char BITSET_NUMBERS_426[] = {3, 5, 1, 7};
__constant__ char BITSET_NUMBERS_427[] = {7, 5, 3};
__constant__ char BITSET_NUMBERS_428[] = {2, 1, 7, 5};
__constant__ char BITSET_NUMBERS_429[] = {5, 2, 7};
__constant__ char BITSET_NUMBERS_430[] = {1, 7, 5};
__constant__ char BITSET_NUMBERS_431[] = {5, 7};
__constant__ char BITSET_NUMBERS_432[] = {2, 4, 3, 7, 1};
__constant__ char BITSET_NUMBERS_433[] = {7, 2, 4, 3};
__constant__ char BITSET_NUMBERS_434[] = {1, 4, 3, 7};
__constant__ char BITSET_NUMBERS_435[] = {4, 3, 7};
__constant__ char BITSET_NUMBERS_436[] = {7, 1, 4, 2};
__constant__ char BITSET_NUMBERS_437[] = {2, 7, 4};
__constant__ char BITSET_NUMBERS_438[] = {1, 4, 7};
__constant__ char BITSET_NUMBERS_439[] = {4, 7};
__constant__ char BITSET_NUMBERS_440[] = {2, 3, 1, 7};
__constant__ char BITSET_NUMBERS_441[] = {7, 3, 2};
__constant__ char BITSET_NUMBERS_442[] = {1, 7, 3};
__constant__ char BITSET_NUMBERS_443[] = {3, 7};
__constant__ char BITSET_NUMBERS_444[] = {7, 1, 2};
__constant__ char BITSET_NUMBERS_445[] = {7, 2};
__constant__ char BITSET_NUMBERS_446[] = {7, 1};
__constant__ char BITSET_NUMBERS_447[] = {7};
__constant__ char BITSET_NUMBERS_448[] = {1, 4, 6, 5, 3, 2};
__constant__ char BITSET_NUMBERS_449[] = {2, 6, 5, 4, 3};
__constant__ char BITSET_NUMBERS_450[] = {6, 4, 1, 3, 5};
__constant__ char BITSET_NUMBERS_451[] = {6, 5, 4, 3};
__constant__ char BITSET_NUMBERS_452[] = {6, 1, 4, 2, 5};
__constant__ char BITSET_NUMBERS_453[] = {5, 4, 2, 6};
__constant__ char BITSET_NUMBERS_454[] = {1, 4, 5, 6};
__constant__ char BITSET_NUMBERS_455[] = {6, 4, 5};
__constant__ char BITSET_NUMBERS_456[] = {6, 3, 1, 2, 5};
__constant__ char BITSET_NUMBERS_457[] = {6, 5, 2, 3};
__constant__ char BITSET_NUMBERS_458[] = {1, 6, 3, 5};
__constant__ char BITSET_NUMBERS_459[] = {5, 6, 3};
__constant__ char BITSET_NUMBERS_460[] = {6, 1, 2, 5};
__constant__ char BITSET_NUMBERS_461[] = {5, 6, 2};
__constant__ char BITSET_NUMBERS_462[] = {6, 1, 5};
__constant__ char BITSET_NUMBERS_463[] = {5, 6};
__constant__ char BITSET_NUMBERS_464[] = {2, 4, 6, 3, 1};
__constant__ char BITSET_NUMBERS_465[] = {3, 6, 4, 2};
__constant__ char BITSET_NUMBERS_466[] = {6, 3, 1, 4};
__constant__ char BITSET_NUMBERS_467[] = {3, 4, 6};
__constant__ char BITSET_NUMBERS_468[] = {4, 2, 1, 6};
__constant__ char BITSET_NUMBERS_469[] = {4, 2, 6};
__constant__ char BITSET_NUMBERS_470[] = {1, 6, 4};
__constant__ char BITSET_NUMBERS_471[] = {6, 4};
__constant__ char BITSET_NUMBERS_472[] = {1, 6, 2, 3};
__constant__ char BITSET_NUMBERS_473[] = {2, 3, 6};
__constant__ char BITSET_NUMBERS_474[] = {1, 3, 6};
__constant__ char BITSET_NUMBERS_475[] = {3, 6};
__constant__ char BITSET_NUMBERS_476[] = {6, 1, 2};
__constant__ char BITSET_NUMBERS_477[] = {6, 2};
__constant__ char BITSET_NUMBERS_478[] = {1, 6};
__constant__ char BITSET_NUMBERS_479[] = {6};
__constant__ char BITSET_NUMBERS_480[] = {1, 3, 2, 4, 5};
__constant__ char BITSET_NUMBERS_481[] = {4, 2, 3, 5};
__constant__ char BITSET_NUMBERS_482[] = {1, 3, 4, 5};
__constant__ char BITSET_NUMBERS_483[] = {4, 3, 5};
__constant__ char BITSET_NUMBERS_484[] = {5, 1, 4, 2};
__constant__ char BITSET_NUMBERS_485[] = {4, 2, 5};
__constant__ char BITSET_NUMBERS_486[] = {5, 1, 4};
__constant__ char BITSET_NUMBERS_487[] = {4, 5};
__constant__ char BITSET_NUMBERS_488[] = {5, 3, 1, 2};
__constant__ char BITSET_NUMBERS_489[] = {2, 5, 3};
__constant__ char BITSET_NUMBERS_490[] = {3, 5, 1};
__constant__ char BITSET_NUMBERS_491[] = {3, 5};
__constant__ char BITSET_NUMBERS_492[] = {5, 2, 1};
__constant__ char BITSET_NUMBERS_493[] = {2, 5};
__constant__ char BITSET_NUMBERS_494[] = {1, 5};
__constant__ char BITSET_NUMBERS_495[] = {5};
__constant__ char BITSET_NUMBERS_496[] = {4, 2, 3, 1};
__constant__ char BITSET_NUMBERS_497[] = {4, 3, 2};
__constant__ char BITSET_NUMBERS_498[] = {1, 4, 3};
__constant__ char BITSET_NUMBERS_499[] = {4, 3};
__constant__ char BITSET_NUMBERS_500[] = {1, 4, 2};
__constant__ char BITSET_NUMBERS_501[] = {2, 4};
__constant__ char BITSET_NUMBERS_502[] = {1, 4};
__constant__ char BITSET_NUMBERS_503[] = {4};
__constant__ char BITSET_NUMBERS_504[] = {3, 1, 2};
__constant__ char BITSET_NUMBERS_505[] = {3, 2};
__constant__ char BITSET_NUMBERS_506[] = {1, 3};
__constant__ char BITSET_NUMBERS_507[] = {3};
__constant__ char BITSET_NUMBERS_508[] = {2, 1};
__constant__ char BITSET_NUMBERS_509[] = {2};
__constant__ char BITSET_NUMBERS_510[] = {1};

// all precomputed array length oft he arrays above.
// for e.g. BITSET_NUMBERS_000.length == BITSET_LENGTH[0].
__constant__ char BITSET_LENGTH[] = {
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
};

// all arrays of possible next values stored in an array of arrays so that
// each array can be found by its computed index.
__constant__ char* BITSET_ARRAY[] = {
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
};

// functions to calculate indices in puzzle:

__device__ char calculate_row_index(char index){
    return index / PUZZLE_SIZE;
};

__device__ char calculate_col_index(char index){
    return index % PUZZLE_SIZE;
};

__device__ char calculate_square_index(char row_index, char col_index){
    return row_index / SQUARE_SIZE * SQUARE_SIZE + col_index / SQUARE_SIZE; // attention: int arithmetic
};

// get count of possible numbers of cell.
__device__ char get_possible_counts(
    SudokuPuzzleData* p,
    char index,
    unsigned short* row_nums,
    unsigned short* col_nums,
    unsigned short* square_nums
) {
    if (p->puzzle[index] == 0) {
        // a puzzle is stored in an linear array, therefore calculated their
        // virtual 2D indices:
        char row_index = calculate_row_index(index);
        char col_index = calculate_col_index(index);
        char square_index = calculate_square_index(row_index, col_index);
        // row_nums, col_nums, square_nums, contains bitsets were each bit represent a number that is not
        // possible since it is already contaied in a row, col, or square.
        // calculate the possiblities by combining these bit vectors:
        unsigned short possible_number_index = row_nums[row_index] | col_nums[col_index] | square_nums[square_index];
        return BITSET_LENGTH[possible_number_index];
    } else {
        return 0;
    }
}

// store a number in a bitset. return true if the number was not already contained.
__device__ bool set_and_check_bit(unsigned short check_bit, unsigned short* array, char index){
    int old_value = array[index];
    array[index] |= check_bit;
    return old_value != array[index];
}

// store a value in row_nums, col_nums, square_nums.
// if not all bit set can store the value mark the puzzle as unsolvable.
__device__ void save_value_for_cell_and_check_is_solvable(
    SudokuPuzzleData* p,
    char value,
    char index,
    unsigned short* row_nums,
    unsigned short* col_nums,
    unsigned short* square_nums
){
    char row_index = calculate_row_index(index);
    char col_index = calculate_col_index(index);
    char square_index = calculate_square_index(row_index, col_index);
    unsigned short check_bit = 1 << (value - 1);
    p->my_is_solvable &= set_and_check_bit(check_bit, row_nums, row_index);
    p->my_is_solvable &= set_and_check_bit(check_bit, col_nums, col_index);
    p->my_is_solvable &= set_and_check_bit(check_bit, square_nums, square_index);
}

// save for each cell all possible values.
__device__  void find_all_possible_values_for_each_empty_cell(
    SudokuPuzzleData* p,
    unsigned short* row_nums,
    unsigned short* col_nums,
    unsigned short* square_nums
){
    for(int i = 0; i < CELL_COUNT; i++){
        char cur_value = p->puzzle[i];
        if(cur_value > 0){
            save_value_for_cell_and_check_is_solvable(p, cur_value, i, row_nums, col_nums, square_nums);
        }
    }
}

// rearange the puzzle layout so that each index in the puzzle array represents an virtual index stored in an indices array.
__device__ void sort_puzzle(
    SudokuPuzzleData* p,
    char * puzzle_sorted,
    char * indices
){
    for(int i = 0; i < CELL_COUNT; i++){
        puzzle_sorted[i] = p->puzzle[indices[i]];
    }
}

// revert method 'sort_puzzle' so that the layout of the puzzle represents its real 1D layout
__device__ void fill_positions(
    SudokuPuzzleData* p,
    char * puzzle_sorted,
    char * indices
){
    for(int i = 0; i < CELL_COUNT; i++){
        p->puzzle[indices[i]] = puzzle_sorted[i];
    }
}

// calculate 1D array index from 2D array index
__device__ int get_single_array_index(char row, char col){
    return row * PUZZLE_SIZE + col;
}

// rearange puzzle to avoid jumping in the puzzle array
__device__ void prepare_puzzle_for_solving(
    SudokuPuzzleData* p,
    char* puzzle_sorted,
    char* indices,
    unsigned short* row_nums,
    unsigned short* col_nums,
    unsigned short* square_nums
) {
    // init
    int number_off_sets[PUZZLE_SIZE + 2];     // numbers 0 - 9 + 1 for offset = puzzleSize + 2 (9 + 2 = 11)
    for(int i = 0; i < PUZZLE_SIZE + 2; i++){ // c does not auto init with default number zero
        number_off_sets[i] = 0;
    }

    // count number of possible solutions for each cell.
    // number_off_sets[1] == 81 means each of all 81 cells have 0 possible numbers
    for(int i = 0; i < CELL_COUNT; i++){
        int count_of_index = get_possible_counts(p, i, row_nums, col_nums, square_nums);
        number_off_sets[count_of_index + 1]++; // note the offset of one
    }

    // test if sudoku is already solved.
    p->my_is_solved = number_off_sets[1] == CELL_COUNT; // all cells have already a solution!

    // correct offsets.
    for(int i = 1; i < PUZZLE_SIZE + 2; i++){
        // there are 'number_off_sets[0]' fields that are already solved,
        // 'number_off_sets[1]' with zero or one possibilities left
        // 'number_off_sets[2]' with zero, one or two possibilities left
        number_off_sets[i] += number_off_sets[i - 1];
    }

    // create an 'indices' array so that the the puzzle layout can be easily changed
    // afterwards to an layout where the sudoku fields with the least possible numbers are tested first (in a linear iteration)
    for(int i = 0; i < CELL_COUNT; i++){
        int count_of_index = get_possible_counts(p, i, row_nums, col_nums, square_nums);
        char off_set = number_off_sets[count_of_index];
        indices[off_set] = i;
        number_off_sets[count_of_index]++;
    }

     // rearange puzzle values.
    sort_puzzle(p, puzzle_sorted, indices);
}

// find a solution for a sudoku by traversing and backtracking the sudoku array.
// note that the sodoku has a special layout.
__device__ void find_solution_non_recursively(
    SudokuPuzzleData* p,
    char* puzzle_sorted,
    char* indices,
    unsigned short* row_nums,
    unsigned short* col_nums,
    unsigned short* square_nums
){
    int indices_current[CELL_COUNT];
    for(int i = 0; i < CELL_COUNT; i++){
        indices_current[i]=-1;
    }
    int i = 0;
    while(i < CELL_COUNT){

        char cur_value = puzzle_sorted[i];

        // value of puzzle_sorted[i] is not set
        if(cur_value == 0){

            // is there a current guess possible?
            int puzzle_index = indices[i];
            int row_index = calculate_row_index(puzzle_index);
            int col_index = calculate_col_index(puzzle_index);
            int square_index = calculate_square_index(row_index, col_index);

            // calculate the array index of BITSET_ARRAY containing all possible numbers for the current field.
            unsigned short possible_number_index = row_nums[row_index] | col_nums[col_index] | square_nums[square_index];

            // store the next index to try of all possible numbers of the current field.
            char next_number_index = (indices_current[i] + 1);

            // try possible numbers until all possible numbers have been tried.
            if(next_number_index < BITSET_LENGTH[possible_number_index]){

                // next possible number to try found.
                char* next_numbers = BITSET_ARRAY[possible_number_index];
                char next_number = next_numbers[next_number_index];
                puzzle_sorted[i] = next_number;

                // save value for cell.
                unsigned short check_bit = 1 << (next_number - 1);
                row_nums[row_index] |= check_bit;
                col_nums[col_index] |= check_bit;
                square_nums[square_index] |= check_bit;

                // success.
                indices_current[i] = next_number_index;

                //go to next cell.
                i += 1;

            // backtrack (another field was set to an incorrect number).
            } else {

                // back track position of possible numbers.
                indices_current[i] = -1;

                // not given values are in the head of array 'puzzle_sorted', therefore we can simply go one step back!
                i -= 1;
                char last_invalid_try = puzzle_sorted[i];
                char last_puzzle_index = indices[i];

                // find in the next step a new solution for i.
                puzzle_sorted[i] = 0;

                // revert last value.
                int last_row_index = calculate_row_index(last_puzzle_index);
                int last_col_index = calculate_col_index(last_puzzle_index);
                int last_square_index = calculate_square_index(last_row_index, last_col_index);
                unsigned short last_check_bit = 1 << (last_invalid_try - 1);
                row_nums[last_row_index] ^= last_check_bit;
                col_nums[last_col_index] ^= last_check_bit;
                square_nums[last_square_index] ^= last_check_bit;

            }
        } else {
            i += 1;
        }
    }
    // rearange sudoku to original layout.
    fill_positions(p, puzzle_sorted, indices);

    p->my_is_solved = true;
}

// solve single sudoku on device.
__device__ bool solve_one_sudokus_on_device(SudokuPuzzleData* current, int i){

     //printf("TEST %d\n", i);

    // early out.
    if(!current->my_is_solvable || current->my_is_solved){
        return current->my_is_solved;
    }

    // temporary memory to compute solution.
    char puzzle_sorted[CELL_COUNT];
    char indices[CELL_COUNT];
    unsigned short row_nums[PUZZLE_SIZE];
    unsigned short col_nums[PUZZLE_SIZE];
    unsigned short square_nums[PUZZLE_SIZE];

    // c does not auto init with default.
    for(int i = 0; i < PUZZLE_SIZE; i++){
        row_nums[i] = 0;
        col_nums[i] = 0;
        square_nums[i] = 0;
    }

    // calculate some cell statistics.
    find_all_possible_values_for_each_empty_cell(current, row_nums, col_nums, square_nums);

    // rearange puzzle for solving.
    prepare_puzzle_for_solving(current, puzzle_sorted, indices, row_nums, col_nums, square_nums);

   // solve puzzle.
   if(current->my_is_solvable && !(current->my_is_solved)) {
      find_solution_non_recursively(current, puzzle_sorted, indices, row_nums, col_nums, square_nums);
   }

   return current->my_is_solved;
}

// solve sudokus in parallel.
__global__ void solve_sudokus_in_parallel(SudokuPuzzleData* p, int count){
    int i = threadIdx.x;
    if(i < count) {
        solve_one_sudokus_on_device(&p[i], i);
    }
}

// check if cuda is available and print some information
// extern "C"  //prevent C++ name mangling!
bool is_cuda_available(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        printf("NO CUDA device FOUND!\n");
    } else {
        int devNo = 0; //simply use first device found! could be improved for later versions
        int driverVersion = 0;
        int runtimeVersion = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devNo);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("CUDA Device 0 (currently only first device is supported!)\n");
        printf("CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
    }
    return deviceCount > 0;
}

// function to remove white spaces on both sides of a string i.e trim.
char * trim (char *s){
    int i;
    // skip left side white spaces.
    while (isspace (*s)) s++;
     // skip right side white spaces.
    for (i = strlen (s) - 1; (isspace (s[i])); i--) ;
    s[i + 1] = '\0';
    return s;
}

// check string 'a' contain prefix 'b'.
bool startsWith(const char *a, const char *b){
   if(strncmp(a, b, strlen(b)) == 0) return 1;
   return 0;
}

// read sudoku from file into an array.
void read_sudokus(char * input_file, int count, SudokuPuzzleData * result){

    FILE * fp;
    char * line = NULL;
    int line_len = 0;
    size_t len = 0;

    fp = fopen(input_file, "r");
    if (fp == NULL){
        exit(EXIT_FAILURE);
    }

    int c = 0;
    int lineCount = 0;
    while ((getline(&line, &len, fp)) != -1) {

        line = trim(line);
        line_len = strlen(line);

        if(line_len == 0 || startsWith(line, NEW_SUDOKU_SEPARATOR)){
            continue;
        } else if (line_len != 9) {
            printf("input file in unsupported format!");
            exit(EXIT_FAILURE);
        } else {
            SudokuPuzzleData* current = &result[c];
            int curOffset = lineCount * PUZZLE_SIZE;
            for(int i = 0; i < PUZZLE_SIZE; i++) {
               char check_char = line[i];
               if(check_char >= '0' && check_char <= '9'){
                current->puzzle[i + curOffset] = line[i] - '0';
               } else {
                current->puzzle[i + curOffset] = 0;
               }
            }
            lineCount++;
            if(lineCount == 9){
                current->my_is_solvable = true;
                current->my_is_solved = false;
                c++;  //count only sudokus;
                lineCount = 0;
            }
            if(c == count){
                break;
            }
        }
    }

    fclose(fp);
}

// note only works well for up to 100 sudokus (its very slow)
int main(int argc, char **argv){

    clock_t begin = clock();
    if(!is_cuda_available()){
        printf("SETUP CUDA FIRST!\n\n");
    }

    if(argc < 3){
        printf("sudoku_cuda FILENAME COUNT\n\n");
        printf("./sudoku_cuda p096_sudoku.txt 10\n\n");
        exit(EXIT_SUCCESS);
    }

   char * input_file = argv[1];
   int count = atoi(argv[2]);
   SudokuPuzzleData* puzzle_data_read = (SudokuPuzzleData*) malloc(count * sizeof(SudokuPuzzleData));
   SudokuPuzzleData* puzzle_data_result = (SudokuPuzzleData*) malloc(count * sizeof(SudokuPuzzleData));
   read_sudokus(input_file, count, puzzle_data_read);

   int sent_to_gpu = 0;
   int batch_size = 256;
   int loop_count = 0;
   int loop_success_count = 0;

   printf("Start with CUDA...");
   while(sent_to_gpu < count){

       // copy slice of array.
       int sudokus_still_to_be_send = count - sent_to_gpu;
       int current_batch_size = (sudokus_still_to_be_send > batch_size) ? batch_size : sudokus_still_to_be_send;
       SudokuPuzzleData * puzzle_data = (SudokuPuzzleData*) malloc(current_batch_size * sizeof(SudokuPuzzleData));

       for(int i = 0; i < current_batch_size; i++){
          memcpy(&puzzle_data[i], &puzzle_data_read[i + sent_to_gpu], sizeof(SudokuPuzzleData));
       }

       printf("\n.. try to run on GPU (batchsize %i)! ...\n", current_batch_size);

       // allocate GPU memory.
       SudokuPuzzleData * device_puzzle_data = 0;
       cudaMalloc((void **) & device_puzzle_data, current_batch_size * sizeof(SudokuPuzzleData));
       cudaMemcpy(device_puzzle_data, puzzle_data, current_batch_size * sizeof(SudokuPuzzleData), cudaMemcpyHostToDevice);

       // run in parallel.
       dim3 grid_size(1); dim3 block_size(current_batch_size);
       solve_sudokus_in_parallel<<<grid_size, block_size>>>(device_puzzle_data, current_batch_size);

       // overwrite old data.
       cudaMemcpy(puzzle_data, device_puzzle_data, current_batch_size * sizeof(SudokuPuzzleData), cudaMemcpyDeviceToHost); //copy data back

       // deep copy to result.
       for(int i = 0; i < current_batch_size; i++){
          memcpy(&puzzle_data_result[i + sent_to_gpu], &puzzle_data[i], sizeof(SudokuPuzzleData));
       }

       // check all solved.
       bool loop_success = false;
       for(int i = 0; i < current_batch_size; i++) {
           SudokuPuzzleData* current = &puzzle_data_result[i + sent_to_gpu];
           if(current->my_is_solved){
              loop_success=true;
              loop_success_count++;
              break;
           }
       }

       if(loop_success){
           printf("... SUCCEED!\n");
       } else {
           printf("... FAILED!\n");
           cudaError_t error = cudaGetLastError();
           const char * error_message = cudaGetErrorString(error);
           printf("Loop %d Error=%s\n", loop_count, error_message);
       }

       // free GPU memory.
       cudaFree(device_puzzle_data);
       free(puzzle_data);

       sent_to_gpu+=current_batch_size;
       loop_count++;

   }

   // print sudoku.
   printf("output on host:\n");
   for(int i = 0; i < count; i++) {
       SudokuPuzzleData* current = &puzzle_data_result[i];
         for(int j = 0; j < CELL_COUNT; j++) {
             if(j % PUZZLE_SIZE == 0){
               printf("\n");
             }
             printf("%d", current->puzzle[j]);
         }
         printf("\n");
   }

   free(puzzle_data_read);
   free(puzzle_data_result);

   clock_t end = clock();
   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("CUDA spend %f seconds!\n", time_spent);

   bool success = loop_count == loop_success_count;
   printf("Run on GPU success=%d\n", success);
   if(!success){
       cudaError_t error = cudaGetLastError();
       const char * error_message = cudaGetErrorString(error);
       printf("Loop %d Error=%s\n", loop_count, error_message);
   }
   exit(EXIT_SUCCESS);

}
