import sys
import os
import subprocess
from collections import defaultdict
import re

def demangle(name):
    try:
        out = subprocess.run(['c++filt', name], capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return name

def normalize_func_name(name):
    name = name.strip()
    if name.startswith("_"):
        name = name.lstrip("_")
    if name.startswith("std::"):
        name = name.replace("std::", "")
    return name

def parse_gprof_file(filename):
    """Parse gprof file robustly by splitting on whitespace."""
    data = {}

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("Each sample") or "name" in line:
                continue  # skip headers and empty lines

            parts = line.split()
            if len(parts) < 7:
                continue  # not a valid data line

            try:
                percent = float(parts[0])
                cumulative = float(parts[1])
                self_sec = float(parts[2])
                # function name is always last column
                func_name = parts[-1]
                func_name = demangle(normalize_func_name(func_name))
                data[func_name] = {
                    "percent": percent,
                    "self_sec": self_sec,
                    "cum_sec": cumulative
                }
            except ValueError:
                print(f"Warning: Could not parse line in {filename}: {line.strip()}")
                #continue  # skip lines that don’t match numeric structure
    return data

def average_gprof(files):
    sums = defaultdict(lambda: {"percent": 0.0, "self_sec": 0.0, "cum_sec": 0.0})
    counts = defaultdict(int)

    for file in files:
        file_data = parse_gprof_file(file)
        for func, vals in file_data.items():
            sums[func]["percent"] += vals["percent"]
            sums[func]["self_sec"] += vals["self_sec"]
            sums[func]["cum_sec"] += vals["cum_sec"]
            counts[func] += 1

    averages = {}
    for func in sums:
        n = counts[func]
        avg_percent = sums[func]["percent"] / n
        avg_self = sums[func]["self_sec"] / n
        avg_cum = sums[func]["cum_sec"] / n
        hotspot_score = avg_percent + (avg_self * 100)
        averages[func] = (avg_percent, avg_self, avg_cum, hotspot_score)
    return averages

def save_results(results, output_file):
    with open(output_file, "w") as f:
        f.write("Function\tAvg %Time\tAvg Self (s)\tAvg Cumulative (s)\tHotspot Score\n")
        f.write("=" * 90 + "\n")
        for func, (pct, self_s, cum_s, score) in sorted(results.items(), key=lambda x: x[1][3], reverse=True):
            f.write(f"{func}\t{pct:.2f}\t{self_s:.4f}\t{cum_s:.4f}\t{score:.2f}\n")
    print(f"\n✅ Averaged results saved to: {output_file}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 average_gprof_hotspots.py run1.txt run2.txt run3.txt")
        sys.exit(1)

    input_files = sys.argv[1:]
    results = average_gprof(input_files)
    
    output_file = os.path.join(os.path.dirname(input_files[0]), "averaged_results.txt")
    save_results(results, output_file)

    print("\n Top Hotspots (by Hotspot Score):")
    for func, (pct, self_s, cum_s, score) in sorted(results.items(), key=lambda x: x[1][3], reverse=True)[:10]:
        print(f"{func:<35} score={score:.2f}  avg_self={self_s:.4f}s  avg_pct={pct:.2f}%")

if __name__ == "__main__":
    main()
