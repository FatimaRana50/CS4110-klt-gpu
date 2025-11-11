# ===============================================================
#  Makefile for full KLT + CUDA Tracker (example3.c)
# ===============================================================

CC       = gcc
CXX      = g++
NVCC     = nvcc

CFLAGS    = -DNDEBUG -O3 -Wall
# Re-enable fast math for performance
NVCCFLAGS := -lineinfo -Xptxas -O3 -arch=sm_75 --compiler-options '-O3,-fPIC' -DUSE_CUDA --use_fast_math

# Added -lcublas and -lcusolver for GPU linear algebra
LDFLAGS   = -lm -lcuda -lcudart -lcublas -lcusolver

CSRCS  = pyramid.c selectGoodFeatures.c error.c klt_util.c klt.c \
          pnmio.c storeFeatures.c writeFeatures.c
CUSRCS = convolve_gpu.cu trackFeatures.cu min_eigenvalue_kernel.cu

OBJS = $(CSRCS:.c=.o) $(CUSRCS:.cu=.o)

MAIN = example3Masooma.c
TARGET = klt_tracker

all: $(TARGET)

$(TARGET): $(OBJS) $(MAIN)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(MAIN) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f *.o *.ppm $(TARGET)
# ===============================================================

# ============================================================
# Auto-generate GPU kernel timing graph from report.txt
# ============================================================

EXTRACT_SCRIPT := extract_gpu_timings.py
GRAPH_SCRIPT := make_gpu_graph_from_timings.py
REPORT_FILE := report.txt
TIMING_FILE := profiling/gpu_timing.txt
GRAPH_PNG := profiling/gpu_callgraph_from_timings.png

.PHONY: graph
graph:
	@echo ">>> Generating GPU callgraph automatically from report.txt..."
	@if [ ! -f $(REPORT_FILE) ]; then \
		echo "[Error] $(REPORT_FILE) not found."; \
		exit 1; \
	fi
	@python3 $(EXTRACT_SCRIPT)
	@if [ -f $(TIMING_FILE) ]; then \
		echo "[Info] Parsed kernel timings successfully."; \
		python3 $(GRAPH_SCRIPT); \
		if [ -f $(GRAPH_PNG) ]; then \
			echo "[OK] Graph generated at $(GRAPH_PNG)"; \
		else \
			echo "[Error] Graph generation failed. Check script output."; \
		fi; \
	else \
		echo "[Error] Failed to extract kernel timings from $(REPORT_FILE)."; \
	fi
# ============================================================
