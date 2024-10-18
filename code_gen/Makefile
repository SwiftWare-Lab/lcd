# Makefile to manage the project

# Default target
all: help

kernel:
	python3 generate_triton_kernel.py

# Helper target to display usage
help:
	@echo "Available targets:"
	@echo "  make matrices     - Generate matrices by running the Python script"
	@echo "  make test_cuda    - Compile and run CUDA kernels"
	@echo "  make test_print   - Run Python test_unroll_coarsen_print.py"
	@echo "  make test_atomic  - Run Python test_unroll_coarsen_atomic.py"
	@echo "  make run_all      - Run all kernels by executing run_experiment2.py"
	@echo "  make clean        - Clean up (if needed)"

# Clean target to remove output files and matrices folder
clean:
	rm -rf benchmark_kernels/*.o benchmark_kernels/*.so benchmark_kernels/test_cuda_kernels \
		matrices
	@echo "Cleaned up generated files and matrices folder."