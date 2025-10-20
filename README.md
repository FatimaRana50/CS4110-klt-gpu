# CS4110 KLT GPU Project
# 🧠 CS4110 – KLT Feature Tracker Profiling (Deliverable 1)

## Deliverable1

## 📘 Overview
This project focuses on *profiling and performance analysis* of the *Kanade–Lucas–Tomasi (KLT)* feature tracking algorithm, which is widely used in computer vision for tracking distinct points (features) across image frames.

The goal of this deliverable is to:
- Compile and execute the *KLT example programs* (specifically example3).
- Generate a *gprof performance profile* and *function call graph*.
- Identify computational hotspots for later optimization (e.g., GPU acceleration).

---

## ⚙️ How to Build and Profile

### 🔧 1. Build & Profile example3
Run the following command inside src/V1/klt:

```bash
make -f Makefile.example3 callgraph
```

This single command will:

Compile the source code (example3.c and related KLT modules).

Run the program to produce profiling data (gmon.out).

Generate a gprof performance report (example3_analysis.txt).

Create a visual call graph (example3_callgraph.png) using gprof2dot and Graphviz.

All generated profiling files are automatically moved to the /profiling folder.

🖼️ Profiling Output Preview
+<p align="center"> <img src="src/V1/klt/images/pick1.png" alt="KLT Profiling Graph Preview" width="600"> </p>

📊 Performance Summary (from example3_analysis.txt)

| Function                 | % Time     | Description                                 |
| ------------------------ | ---------- | ------------------------------------------- |
| _convolveImageHoriz    | *42.86%* | Horizontal convolution (Gaussian smoothing) |
| _convolveImageVert     | *23.81%* | Vertical convolution (image pyramid)        |
| _interpolate           | *14.29%* | Pixel interpolation for subpixel accuracy   |
| _KLTSelectGoodFeatures | *4.76%*  | Detecting high-quality feature points       |
| _computeGradientSum    | *4.76%*  | Computing image gradients                   |
| _quicksort             | *4.76%*  | Sorting features by response strength       |

🔍 Key Observations

The convolution and interpolation stages dominate total execution time.

These operations are repeated for every image pyramid level and pixel neighborhood.

Potential optimization targets:

Parallelize convolution (OpenMP / CUDA).

Use optimized image filtering libraries.

Cache gradient and pyramid computations to avoid recomputation.

🚀 Next Steps

Accelerate heavy functions (_convolveImageHoriz, _convolveImageVert, _interpolate) using GPU or SIMD parallelization.

Compare pre- and post-optimization profiles to quantify performance gains.

📂 Directory Structure
```
CS4110-klt-gpu/
│
├── src/
│   └── V1/
│       └── klt/
│           ├── Makefile.example3         # Build + profile script
│           ├── example3.c                # Example program used
│           ├── profiling/                # Output folder for profiling results
│           │   ├── example3_analysis.txt
│           │   ├── example3_callgraph.dot
│           │   └── example3_callgraph.png
│           └── images/
│               └── pick1.png             # Preview image (included above)
│
└── README.md
```

👩‍💻 Contributors

Fatima Farrukh Rana
Fatima Shakir
Faateh Haneef
Course: CS4110 – High Performance Computing
Deliverable 1: Profiling & Hotspot Analysis of KLT Feature Tracker
