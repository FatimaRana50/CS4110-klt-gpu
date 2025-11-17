# ==========================================================
# Makefile.example3 - Build, Run, and Profile with OpenACC
# ==========================================================

# OpenACC compiler configuration (NVIDIA HPC SDK)
CC_ACC      = nvc
CFLAGS_ACC  = -acc -gpu=mem:managed -Minfo=accel -fast -O3
LIBS        = -lm

# Sources
SRC = klt.c klt_util.c pnmio.c convolve.c error.c pyramid.c \
      selectGoodFeatures.c trackFeatures.c writeFeatures.c storeFeatures.c

TARGET      = example3shakir
PROF_DIR    = profiling
OUTPUT_FILE = run_output.txt

# Default target
all: $(TARGET)

# Build with OpenACC (single compilation command)
$(TARGET): $(TARGET).c $(SRC)
	@echo "===================================================="
	@echo "Building $(TARGET) with OpenACC (nvc)..."
	@echo "===================================================="
	$(CC_ACC) $(CFLAGS_ACC) $^ -o $@ $(LIBS)
	@echo "✅ Build complete: $(TARGET)"
	@echo ""
	mkdir -p $(PROF_DIR)

# Run and save output
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET) | tee $(OUTPUT_FILE)

# Test OpenACC (kernel launch messages)
test-acc: $(TARGET)
	@echo "===================================================="
	@echo "Testing OpenACC Execution with ACC_NOTIFY"
	@echo "===================================================="
	ACC_NOTIFY=3 ./$(TARGET) 2>&1 | head -100

# Nsight Systems profiling
profile: $(TARGET)
	@echo "===================================================="
	@echo "Profiling with Nsight Systems"
	@echo "===================================================="
	mkdir -p $(PROF_DIR)
	nsys profile --stats=true --force-overwrite=true \
	             -o $(PROF_DIR)/$(TARGET)_profile ./$(TARGET)
	@echo ""
	@if [ -f $(PROF_DIR)/$(TARGET)_profile.nsys-rep ]; then \
		echo "✅ Profile saved: $(PROF_DIR)/$(TARGET)_profile.nsys-rep"; \
		echo "Generating text reports..."; \
		nsys stats --report gputrace --format csv \
		           $(PROF_DIR)/$(TARGET)_profile.nsys-rep \
		           > $(PROF_DIR)/gpu_trace.csv 2>&1 || true; \
		nsys stats --report cudaapisum \
		           $(PROF_DIR)/$(TARGET)_profile.nsys-rep \
		           > $(PROF_DIR)/cuda_api.txt 2>&1 || true; \
		echo "✅ Text reports saved to $(PROF_DIR)/"; \
	fi

# Quick GPU activity check
profile-quick: $(TARGET)
	@echo "Quick GPU activity check..."
	@nsys profile --stats=true -o /tmp/quick_check ./$(TARGET) \
	    2>&1 | grep -i "kernel\|GPU\|launch" | head -20
	@rm -f /tmp/quick_check.*

# CPU-only version (optional)
cpu:
	gcc -O3 -Wall $(SRC) $(TARGET).c -o $(TARGET)_cpu -lm
	@echo "✅ Built CPU-only version: $(TARGET)_cpu"

# System info
info:
	@echo "===================================================="
	@echo "System Information"
	@echo "===================================================="
	@echo "Compiler:"
	@$(CC_ACC) --version | head -2
	@echo ""
	@echo "GPU Info:"
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1 \
	    || echo "nvidia-smi not available"
	@echo ""
	@echo "Nsight Systems:"
	@nsys --version 2>&1 | head -1 || echo "nsys not available"

# Verify OpenACC compilation messages
check-compile:
	@echo "===================================================="
	@echo "Checking OpenACC Compilation"
	@echo "===================================================="
	@$(MAKE) clean --no-print-directory
	@$(MAKE) all 2>&1 | tee compile_check.log
	@echo ""
	@echo "Checking for OpenACC accelerator generation..."
	@grep -i "Generating" compile_check.log \
	    || echo "⚠️ No OpenACC accelerator output found"
	@rm -f compile_check.log
	
	
# rm -rf $(PROF_DIR) - add this if you want to remove profiling directory on clean
clean:
	rm -f $(TARGET) $(TARGET)_cpu $(OUTPUT_FILE) *.o *.ppm *.pgm
	@echo "🧹 Cleaned all build and image files"


.PHONY: all run test-acc profile profile-quick cpu info \
        check-compile clean
