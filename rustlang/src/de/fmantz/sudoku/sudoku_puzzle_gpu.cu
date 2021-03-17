#include <iostream>
#include <stdlib.h>
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
}

struct SudokuPuzzleData {
    bool my_is_solvable;
    bool my_is_solved;
    char puzzle[81];
};

//simple dummy hello world:
extern "C"  //prevent C++ name mangling!
int solve_on_cuda(SudokuPuzzleData p[], int count){ //library method

    for(int i = 0; i < count; i++) {
        SudokuPuzzleData current = p[i];
        for(int j = 0; j < 81; j++) {
            if(j % 9 == 0){
              printf("\n");
            }
            printf("%d", current.puzzle[j]);
        }
        printf("\n-----------");
    }

    printf("Hello World from CPU!\n");
//    cuda_hello<<<500,1024>>>();
    cuda_hello<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
