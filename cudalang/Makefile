# simple MakeFile for CUDA sudoku_cuda //sudoku_gpu_lib
CUDA_ROOT_DIR ?= "/opt/cuda"
HOST_COMPILER := $(CUDA_ROOT_DIR)/bin/gcc
#LIB_FLAGS := -shared --compiler-options '-fPIC' //Comment in to use CUDA in a shared lib
LIB_FLAGS :=
NVCC := $(CUDA_ROOT_DIR)/bin/nvcc -ccbin $(HOST_COMPILER) $(LIB_FLAGS) -arch=compute_61 -rdc=true
MKDIR_P = mkdir -p
INPUT_DIR := "./"
OUTPUT_DIR := "./"
#OUTPUT_NAME := "libsudoku_puzzle_gpu.so" //Comment in to use CUDA in a shared lib
OUTPUT_NAME := "sudoku_cuda"

#Note: taps are required instead of spaces AND name must start with lib:
sudoku_gpu_lib:
	$(MKDIR_P) $(OUTPUT_DIR)
	$(NVCC) $(INPUT_DIR)/sudoku_puzzle_gpu.cu -o $(OUTPUT_DIR)/$(OUTPUT_NAME)
clean:
	rm  $(OUTPUT_DIR)/libsudoku_puzzle_gpu.so
