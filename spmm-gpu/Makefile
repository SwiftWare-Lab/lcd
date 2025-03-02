# Makefile to manage the project

# Default target
all: help

# Target to generate matrices by running the Python script in utils directory
matrices:
	cd utils && python3 gen_matrices.py

# Target to compile and run CUDA kernels
test_cuda:
	cd benchmark_kernels && \
	nvcc -c -o CUDA_wrapper.o CUDA_wrapper.cu -lcusparse -lcublas -lcudart && \
	nvcc -c -o test_cuda_kernels.o test_cuda_kernels.cu -lcusparse -lcublas -lcudart && \
	nvcc -o test_cuda_kernels CUDA_wrapper.o test_cuda_kernels.o -lcusparse -lcublas -lcudart && \
	./test_cuda_kernels

# Target to run Python test_unroll_coarsen_print.py
test_print:
	cd benchmark_kernels && python3 test_unroll_coarsen_print.py

# Target to run Python test_unroll_coarsen_atomic.py
test_atomic:
	cd benchmark_kernels && python3 test_unroll_coarsen_atomic.py

# Target to run all kernels by executing run_experiment2.py
run_all:
	cd benchmark_kernels && \
	run using nvcc -o cuda_lib.so. CUDA_wrapper.cu -shared -lcusparse -lcublas -lcudart -Xcompiler -fPIC && \
	python3 run_experiment2.py

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