CUDA_ROOT_DIR ?= /opt/cuda
#HOST_COMPILER := /usr/sbin/gcc-13
#NVCC          := $(CUDA_ROOT_DIR)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC          := $(CUDA_ROOT_DIR)/bin/nvcc
MKDIR_P       := mkdir -p
INPUT_DIR     := ./
OUTPUT_DIR    := ./
OUTPUT_NAME   := sudoku_cuda

sudoku_gpu_lib:
	$(MKDIR_P) $(OUTPUT_DIR)
	$(NVCC) $(INPUT_DIR)sudoku_puzzle_gpu.cu -o $(OUTPUT_DIR)$(OUTPUT_NAME)
