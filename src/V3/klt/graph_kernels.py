#!/usr/bin/env python3
"""
extract_gpu_timings.py
----------------------
Parse Nsight Systems 'report.txt' and extract the CUDA GPU Kernel Summary
into profiling/gpu_timing.txt, formatted as:
<kernel_name> <total_time_ms>
"""

import re, os, sys

REPORT_FILE = "report.txt"
OUT_DIR = "profiling"
OUT_FILE = os.path.join(OUT_DIR, "gpu_timing.txt")

if not os.path.exists(REPORT_FILE):
    print(f"[Error] {REPORT_FILE} not found.")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

lines = open(REPORT_FILE, "r", errors="ignore").read().splitlines()

collect = False
kern_lines = []
for line in lines:
    if re.search(r"\*\*\s*CUDA GPU Kernel Summary", line):
        collect = True
        continue
    if collect:
        if line.strip().startswith("Processing ["):
            break
        if not line.strip() or line.strip().startswith("Time") or line.strip().startswith("-"):
            continue
        kern_lines.append(line)

if not kern_lines:
    print("[Warning] No CUDA kernel section found.")
    sys.exit(0)

row_re = re.compile(
    r"^\s*(\d+\.\d+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+[ \d.]+\s+[ \d.]+\s+[ \d.]+\s+[ \d.]+\s+(.+)$"
)
out_data = []
for ln in kern_lines:
    m = row_re.match(ln)
    if not m:
        continue
    pct = float(m.group(1))
    total_ns = int(m.group(2).replace(",", ""))
    name = m.group(5).strip()
    # Clean up kernel name
    name = re.sub(r"\(.*?\)", "", name).replace("…", "").strip()
    total_ms = total_ns / 1e6
    out_data.append((name, total_ms))

if not out_data:
    print("[Warning] No kernel entries parsed.")
    sys.exit(0)

# Write timings file
with open(OUT_FILE, "w") as f:
    for name, t in out_data:
        f.write(f"{name} {t:.3f}\n")

print(f"[OK] Extracted {len(out_data)} kernels to {OUT_FILE}")
