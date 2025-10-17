#!/usr/bin/env python3
"""
Aggregate profiling/gpu_timing.txt into a simple callgraph DOT and PNG.

Expect lines like: "label <ms>". The script sums ms by label and writes
profiling/gpu_callgraph_from_timings.dot and .png.
"""
import os
import sys
import shutil

IN = 'profiling/gpu_timing.txt'
OUT_DIR = 'profiling'
DOT = os.path.join(OUT_DIR, 'gpu_callgraph_from_timings.dot')
PNG = os.path.join(OUT_DIR, 'gpu_callgraph_from_timings.png')

def parse_timings():
    if not os.path.exists(IN):
        print('No', IN, 'found')
        return {}
    totals = {}
    with open(IN) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            label, val = parts[0], parts[1]
            try:
                v = float(val)
            except:
                continue
            totals[label] = totals.get(label, 0.0) + v
    return totals

def write_dot(totals):
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    with open(DOT, 'w') as f:
        # Compute total time and percentage per node
        total = sum(totals.values())
        f.write('digraph G {\n')
        f.write('  rankdir=TB;\n')
        f.write('  node [shape=box, style=filled, fontname="Helvetica"];\n')
        # Assign node color classes similar to gprof image:
        # root (largest) -> red, big hotspots -> green, medium -> teal, small -> blue
        def color_class(pct):
            if pct > 40.0:
                return ('#d62728', 18)   # red, large font
            if pct > 10.0:
                return ('#2ca02c', 12)   # green
            if pct > 2.0:
                return ('#17a2b8', 11)   # teal/blue
            return ('#1f77b4', 10)       # dark blue small

        # sort nodes by total time descending for stable layout
        items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (k, v) in enumerate(items):
            pct = (v / total * 100.0) if total > 0 else 0.0
            label = f"{k}\\n{pct:.2f}%\\n({v:.3f} ms)"
            color, fontsize = color_class(pct)
            safe = k.replace('"', '\\"')
            # emphasize the top node as 'main' style red box with larger font
            if idx == 0:
                f.write(f'  "{safe}" [label="{label}", fillcolor="#d62728", fontcolor="#ffffff", fontsize=18, style=filled, shape=box];\n')
            else:
                f.write(f'  "{safe}" [label="{label}", fillcolor="{color}", fontsize={fontsize}, style=filled, shape=box];\n')

        # Create a simple synthetic edge layout: order nodes by size and link
        # biggest -> next ones as a star-like call graph to mimic the gprof style.
        if items:
            root = items[0][0].replace('"','\\"')
            for k, v in items[1:]:
                kk = k.replace('"','\\"')
                # edge width proportional to child's percent (visual thickness)
                child_pct = (v / total * 100.0) if total>0 else 0
                pen = max(1.0, child_pct / 5.0)
                f.write(f'  "{root}" -> "{kk}" [label="{child_pct:.2f}%", color="#4f94d4", penwidth={pen:.2f}];\n')

        f.write('}\n')
    print('Wrote', DOT)

def render():
    if shutil.which('dot') is None:
        print('dot not found; skipping PNG')
        return
    rc = os.system(f'dot -Tpng {DOT} -o {PNG}')
    if rc == 0:
        print('Wrote', PNG)
    else:
        print('dot returned', rc)

def main():
    totals = parse_timings()
    if not totals:
        # If no proper timing lines were parsed, check for CUDA error messages
        # in the profiling file and write a diagnostic DOT so the user gets
        # a visible artifact instead of nothing.
        if os.path.exists(IN):
            with open(IN) as f:
                text = f.read()
            # look for CUDA-related messages
            if any(pat in text for pat in ('CUDA error', 'CUDA initialization failed', 'cudaGetLastError', 'unknown error')):
                if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
                with open(DOT, 'w') as f:
                    f.write('digraph G {\n')
                    f.write('  node [shape=box, style=filled, fillcolor="#ffeeee", fontname="Helvetica"]\n')
                    # keep the label short (first 400 chars)
                    safe = text.replace('"', '\\"').replace('\n', '\\n')[:400]
                    f.write(f'  "CUDA error or no timings" [label="CUDA error or no timings\\n{safe}"];\n')
                    f.write('}\n')
                print('Wrote', DOT)
                # attempt to render
                render()
                return
        print('No timing data parsed')
        return
    write_dot(totals)
    render()

if __name__ == '__main__':
    main()
