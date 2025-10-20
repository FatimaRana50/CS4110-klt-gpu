######################################################################
# Compiler setup
CC = gcc
NVCC = nvcc

######################################################################
# Flags
NVCC_FLAGS = -arch=sm_75 -O3 -g -G -DDEBUG -lineinfo -Xcompiler "-fopenmp"
CFLAGS = -O3 -DDEBUG -fopenmp

######################################################################
# Source and target setup
SOURCES = example3.c klt.c klt_util.c pnmio.c convolve.cu selectGoodFeatures.cu \
          error.c pyramid.c trackFeatures.cu writeFeatures.c storeFeatures.c

OBJECTS = $(SOURCES:.c=.o)
OBJECTS := $(OBJECTS:.cu=.o)

TARGET = example3_gpu

######################################################################
# Default target
all: $(TARGET)

######################################################################
# Build rules
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJECTS) -lm

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.c
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

######################################################################
# Run on all 10 images (img0 → img1, img1 → img2, ...)
run: $(TARGET)
	@echo "🔹 Running KLT tracker on all image pairs (img0–img9)..."
	@rm -f features.txt
	@for i in 0 1 2 3 4 5 6 7 8; do \
		next=$$((i+1)); \
		echo "▶ Processing img$$i.pgm → img$$next.pgm"; \
		./$(TARGET) img$$i.pgm img$$next.pgm | tee -a features.txt; \
	done
	@echo " Tracking complete. Results saved to features.txt."

######################################################################
# Full GPU profiling and graph generation on all 10 images
profile_gpu: $(TARGET)
	@echo "🔹 Starting GPU profiling on all images..."
	@mkdir -p profiling
	@rm -f profiling/gpu_timing.txt
	@for i in 0 1 2 3 4 5 6 7 8; do \
		next=$$((i+1)); \
		echo "▶ Profiling img$$i.pgm → img$$next.pgm"; \
		CUDA_LAUNCH_BLOCKING=1 KLT_DEVICE_CHECKS=1 KLT_HOST_FALLBACK=1 ./$(TARGET) img$$i.pgm img$$next.pgm >> profiling/gpu_timing.txt 2>&1 || true; \
	done
	@echo "Profiling complete. Generating timing graph..."
	@python3 make_gpu_graph_from_timings.py
	@echo " Graph generated in profiling/ directory."

######################################################################
# Simulated graph (no GPU required)
simulate_graph:
	python3 simulate_gpu_graph.py

######################################################################
# Clean all builds and profiling outputs
clean:
	rm -f $(TARGET) $(OBJECTS) features.txt
	rm -f profiling/gpu_timing.txt profiling/gpu_callgraph_from_timings.* profiling/gpu_callgraph_simulated.*
	rm -rf profile_report.nsys-rep ncu_report.ncu-rep ncu_quick_report.ncu-rep

.PHONY: all run profile_gpu simulate_graph clean
#make run to compile
#make profile_gpu for graph 
