#include <iostream>
#include <stdlib.h>
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
}

struct SudokuPuzzleData {
    bool my_is_solvable;
    bool my_is_solved;
    unsigned char puzzle[81];
};

//simple dummy hello world:
extern "C"  //prevent C++ name mangling!
int solve_on_cuda(SudokuPuzzleData p[], unsigned int count){ //library method

    for(int i = 0; i < 3; i++) {
        printf("parameter %d \n", p[i].my_is_solvable);
    }

    printf("Hello World from CPU!\n");
//    cuda_hello<<<500,1024>>>();
    cuda_hello<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
