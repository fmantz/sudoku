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
   cudaDeviceSynchronize(); //that output data is transfered back

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
