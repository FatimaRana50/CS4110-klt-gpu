######################################################################
# Choose your favorite C compiler
CC = gcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
# Compiler and flags
NVCC = nvcc
# Match the command we've been using for builds (includes device debug and host flags)
NVCC_FLAGS = -arch=sm_75 -O3 -g -G -DDEBUG -lineinfo -Xcompiler "-fopenmp"
CFLAGS = -O3 -DDEBUG -fopenmp

# Source files
SOURCES = example3.c klt.c klt_util.c pnmio.c convolve.cu selectGoodFeatures.cu \
          error.c pyramid.c trackFeatures.cu writeFeatures.c storeFeatures.c

# Object files (replace .cu with .o and .c with .o)
OBJECTS = $(SOURCES:.c=.o)
OBJECTS := $(OBJECTS:.cu=.o)

# Executable
TARGET = example3_gpu

# Image files for testing (update these to your actual image paths)
IMAGE1 = img0.pgm
IMAGE2 = img1.pgm

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJECTS) -lm

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C files  (we use nvcc for consistent linking of CUDA runtime)
%.o: %.c
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Run the program and save output to features.txt
run: $(TARGET)
	./$(TARGET) $(IMAGE1) $(IMAGE2) | tee features.txt

# NSYS profiling - system-level performance analysis
profile: $(TARGET)
	nsys profile --stats=true --trace=cuda,nvtx --force-overwrite true -o profile_report ./$(TARGET) $(IMAGE1) $(IMAGE2) | tee features.txt

# NCU profiling - kernel-level performance analysis  
ncu: $(TARGET)
	ncu --set full --force-overwrite -o ncu_report ./$(TARGET) $(IMAGE1) $(IMAGE2) | tee features.txt

# Quick NCU with basic metrics
ncu-quick: $(TARGET)
	ncu --metrics gpu__time_duration.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed --force-overwrite -o ncu_quick_report ./$(TARGET) $(IMAGE1) $(IMAGE2) | tee features.txt

# Generate profiling/gpu_timing.txt using the instrumented timing code and then run the Python graph generator
profile_gpu: $(TARGET)
	@mkdir -p profiling
	: > profiling/gpu_timing.txt
	CUDA_LAUNCH_BLOCKING=1 KLT_DEVICE_CHECKS=1 KLT_HOST_FALLBACK=1 ./$(TARGET) $(IMAGE1) $(IMAGE2) >> profiling/gpu_timing.txt 2>&1 || true
	python3 make_gpu_graph_from_timings.py

# Simulate a graph (useful when CUDA runtime isn't available)
simulate_graph:
	python3 simulate_gpu_graph.py

# Clean
clean:
	rm -f $(TARGET) $(OBJECTS) features.txt
	rm -f profiling/gpu_timing.txt profiling/gpu_callgraph_from_timings.* profiling/gpu_callgraph_simulated.*
	rm -rf profile_report.nsys-rep ncu_report.ncu-rep ncu_quick_report.ncu-rep

.PHONY: all run profile ncu ncu-quick profile_gpu simulate_graph clean


/*
commands:
1. make
2. : > profiling/gpu_timing.tx
3. CUDA_LAUNCH_BLOCKING=1 KLT_DEVICE_CHECKS=1 KLT_HOST_FALLBACK=1   ./example3_gpu img0.pgm img1.pgm >> profiling/gpu_timing.txt 2>&1
4. python3 make_gpu_graph_from_timings.py
5. make simulate_graph

compile -> remove existing profiling.txt for rerun -> rerun -> generate graph from timings -> simulate graph
*/


