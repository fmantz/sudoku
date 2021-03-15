#Simple MakeFile for CUDA sudoku_gpu_lib
CUDA_ROOT_DIR ?= "/opt/cuda"
HOST_COMPILER := $(CUDA_ROOT_DIR)/bin/gcc
LIB_FLAGS := -shared --compiler-options '-fPIE -fPIC'
NVCC := $(CUDA_ROOT_DIR)/bin/nvcc -ccbin $(HOST_COMPILER) $(LIB_FLAGS)

#Note: taps are required instead of spaces AND name must start with lib:
sudoku_gpu_lib:
	$(NVCC) src/de/fmantz/sudoku/sudoku_puzzle_gpu.cu -o lib/libsudoku_puzzle_gpu.so
clean:
	rm lib/libsudoku_puzzle_gpu.so
