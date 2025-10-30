# =============================================================
# V3 Optimized GPU Implementation Makefile (SAFE + FINAL)
# =============================================================

CC = gcc
NVCC = nvcc
CUDA_PATH ?= /usr/local/cuda

# Compiler flags
CFLAGS = -O3 -Wall -DUSE_CUDA -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -arch=sm_75 -use_fast_math --compiler-options '-fPIC' -DUSE_CUDA
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm

# =============================================================
# Source files
# =============================================================

C_SRCS = klt.c error.c klt_util.c pyramid.c \
         selectGoodFeatures.c storeFeatures.c \
         writeFeatures.c pnmio.c example3.c

CU_SRCS = convolve_gpu.cu trackFeatures.cu

C_OBJS = $(C_SRCS:.c=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)

TARGET = klt_tracker

# =============================================================
# Build rules
# =============================================================

all: $(TARGET)

$(TARGET): $(C_OBJS) $(CU_OBJS)
	$(NVCC) -o $@ $(C_OBJS) $(CU_OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# =============================================================
# Safe cleanup (does not touch .cu/.h source files)
# =============================================================
clean:
	rm -f $(C_OBJS) $(CU_OBJS) $(TARGET) *.ppm *.txt *.ft *.nsys-rep *.sqlite

# =============================================================
# Utility targets
# =============================================================
run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	nsys profile --stats=true -o profile ./$<

.PHONY: all clean run profile