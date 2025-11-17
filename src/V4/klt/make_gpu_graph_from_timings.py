#!/usr/bin/env python3
import os, re, shutil

IN = "profiling/gpu_timing.txt"
OUT_DIR = "profiling"
DOT = os.path.join(OUT_DIR, "gpu_callgraph_from_timings.dot")
PNG = os.path.join(OUT_DIR, "gpu_callgraph_from_timings.png")

def parse_nsight_ascii(path):
    if not os.path.exists(path):
        print("Missing:", path)
        return {}

    totals = {}
    inside = False

    # Regex: capture (Time%), (TotalNS), (KernelName)
    line_re = re.compile(
        r"^\s*([\d\.]+)\s+([\d,]+)\s+\d+\s+.*?(\S+)\s*$"
    )

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()

            # Detect table start
            if "CUDA GPU Kernel Summary" in line:
                inside = True
                continue

            if not inside:
                continue

            # skip header lines
            if (
                "Time (%)" in line or
                "--------" in line or
                line.strip() == ""
            ):
                continue

            # stop if new block starts
            if line.startswith("Processing"):
                inside = False
                continue

            m = line_re.match(line)
            if not m:
                continue

            time_pct = float(m.group(1))
            total_ns = float(m.group(2).replace(",", ""))
            name = m.group(3)

            total_ms = total_ns / 1e6
            totals[name] = totals.get(name, 0.0) + total_ms

    return totals

def write_dot(totals):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    total = sum(totals.values())

    with open(DOT, "w") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=TB;\n")
        f.write('  node [shape=box, style=filled, fontname="Helvetica"];\n')

        def color(p):
            if p > 40: return "#d62728", 18
            if p > 10: return "#2ca02c", 12
            if p > 2: return "#17a2b8", 11
            return "#1f77b4", 10

        items = sorted(totals.items(), key=lambda x: x[1], reverse=True)

        for i, (name, ms) in enumerate(items):
            pct = (ms / total) * 100
            c, sz = color(pct)
            lbl = f"{name}\\n{pct:.2f}%\\n({ms:.3f} ms)"
            safe = name.replace('"','\\"')

            if i == 0:
                f.write(f'  "{safe}" [label="{lbl}", fillcolor="#d62728", fontcolor="#fff", fontsize=18];\n')
            else:
                f.write(f'  "{safe}" [label="{lbl}", fillcolor="{c}", fontsize={sz}];\n')

        # edges
        if items:
            root = items[0][0].replace('"','\\"')
            for name, ms in items[1:]:
                pct = (ms / total) * 100
                pen = max(1, pct/5)
                child = name.replace('"','\\"')
                f.write(f'  "{root}" -> "{child}" [label="{pct:.2f}%", penwidth={pen:.2f}];\n')

        f.write("}\n")

def render_png():
    if shutil.which("dot"):
        os.system(f'dot -Tpng "{DOT}" -o "{PNG}"')
        print("Wrote", PNG)
    else:
        print("Graphviz dot not found")

def main():
    timings = parse_nsight_ascii(IN)
    if not timings:
        print("No valid kernel timings parsed.")
        return
    write_dot(timings)
    render_png()

if __name__ == "__main__":
    main()

