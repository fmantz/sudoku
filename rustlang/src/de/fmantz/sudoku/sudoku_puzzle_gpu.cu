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
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

struct SudokuPuzzleData {
    bool my_is_solvable;
    bool my_is_solved;
    char puzzle[81];
};

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

//solve one single sudoku one device:
__device__ void solve_one_sudokus_one_device(SudokuPuzzleData *current){
    //try to change data and return changed data to rust! works :-)
    //TODO: work with pointern in print sudoku struct is copied!
    for(int i = 0; i < 9; i++) {
        printf("1. Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
        current->puzzle[i] = 9;
        printf("%d\n", current->puzzle[i]);
        printf("2. Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
    }
}

//solve sudokus in parallel:
__global__ void solve_sudokus_in_parallel(SudokuPuzzleData *p, int count){
    for(int i = 0; i < count; i++) {
        solve_one_sudokus_one_device(&p[i]);
    }
}

//library function to call from rust:  //TODO add another function to check compute compability and device found!
extern "C"  //prevent C++ name mangling!
int solve_on_cuda(SudokuPuzzleData* puzzle_data, int count){ //library method

   printf("Hello World from CPU!\n");

   //print sudoku:
   printf("input:\n");
   for(int i = 0; i < count; i++) {
        SudokuPuzzleData current = puzzle_data[i];
        for(int j = 0; j < 81; j++) {
            if(j % 9 == 0){
              printf("\n");
            }
            printf("%d", current.puzzle[j]);
        }
        printf("\n-----------");
   }

   // Allocate GPU memory.
   SudokuPuzzleData *device_puzzle_data = 0;
   cudaMalloc((void **) & device_puzzle_data, count * sizeof(SudokuPuzzleData));
   cudaMemcpy(device_puzzle_data, puzzle_data, count * sizeof(SudokuPuzzleData), cudaMemcpyHostToDevice);

   //Run in parallel:
   solve_sudokus_in_parallel<<<1,count>>>(device_puzzle_data, count);

   //Free old data:
   free(puzzle_data);
   cudaMemcpy(puzzle_data, device_puzzle_data, count * sizeof(SudokuPuzzleData), cudaMemcpyDeviceToHost); //copy data back

   // Free GPU memory:
   cudaFree(device_puzzle_data);

   //print sudoku:
   printf("output:\n");
   for(int i = 0; i < count; i++) {
       SudokuPuzzleData current = puzzle_data[i];
         for(int j = 0; j < 81; j++) {
             if(j % 9 == 0){
               printf("\n");
             }
             printf("%d", current.puzzle[j]);
         }
         printf("\n-----------");
   }

   return EXIT_SUCCESS;
}
