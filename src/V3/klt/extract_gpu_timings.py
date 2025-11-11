#!/usr/bin/env python3
"""
extract_gpu_timings.py
----------------------
Parse Nsight Systems 'report.txt' and extract the
CUDA GPU Kernel Summary section into profiling/gpu_timing.txt
Format:
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

with open(REPORT_FILE, "r", errors="ignore") as f:
    lines = f.readlines()

kern_lines = []
collect = False
for line in lines:
    if re.search(r"\*\*\s*CUDA GPU Kernel Summary", line):
        collect = True
        continue
    if collect:
        # Stop when next section begins
        if line.strip().startswith("Processing ["):
            break
        # Skip header and divider lines
        if not line.strip() or line.strip().startswith("Time") or line.strip().startswith("-"):
            continue
        kern_lines.append(line.rstrip())

if not kern_lines:
    print("[Warning] No CUDA kernel summary section found.")
    sys.exit(0)

# Simplified pattern: <pct> <total_ns> ... <name>
pattern = re.compile(r"^\s*(\d+\.\d+)\s+([\d,]+).*?\s([A-Za-z0-9_]+)\s*(?:\(.*)?$")

parsed = []
for ln in kern_lines:
    m = pattern.match(ln)
    if not m:
        continue
    pct = float(m.group(1))
    total_ns = int(m.group(2).replace(",", ""))
    name = m.group(3).strip()
    total_ms = total_ns / 1e6
    parsed.append((name, total_ms, pct))

if not parsed:
    print("[Warning] No kernel entries parsed.")
    sys.exit(0)

# Write output file
with open(OUT_FILE, "w") as f:
    for name, ms, pct in parsed:
        f.write(f"{name} {ms:.3f}\n")

print(f"[OK] Extracted {len(parsed)} kernels to {OUT_FILE}")
