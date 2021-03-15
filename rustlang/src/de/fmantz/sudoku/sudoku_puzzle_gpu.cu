#include <iostream>
#include <stdlib.h>
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU! %d\n", threadIdx.x*gridDim.x);
}

//simple dummy hello world:
extern "C"  //prevent C++ name mangling!
int solve_on_cuda(){ //library method
    printf("Hello World from CPU!\n");
//    cuda_hello<<<500,1024>>>();
    cuda_hello<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
