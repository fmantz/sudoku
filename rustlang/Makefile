#Simple MakeFile for CUDA sudoku_gpu_lib
CUDA_ROOT_DIR ?= "/opt/cuda"
HOST_COMPILER := g++
NVCC := $(CUDA_ROOT_DIR)/bin/nvcc -ccbin $(HOST_COMPILER)

#Note: taps are required instead of spaces:
sudoku_gpu_lib:
	$(NVCC) src/de/fmantz/sudoku/sudoku_puzzle_gpu.cu -o target/sudoku_puzzle_gpu
clean:
	rm target/sudoku_puzzle_gpu
