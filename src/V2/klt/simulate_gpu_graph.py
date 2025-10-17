#!/usr/bin/env python3
"""
Create a simulated GPU callgraph DOT and PNG using typical kernel labels from this repo.
This produces profiling/gpu_callgraph_simulated.dot and profiling/gpu_callgraph_simulated.png

Usage: python3 simulate_gpu_graph.py
"""
import os
from math import floor

out_dot = 'profiling/gpu_callgraph_simulated.dot'
out_png = 'profiling/gpu_callgraph_simulated.png'

# Typical labels we instrumented / observed in this repo
labels = [
    # convolve.cu
    'convolve_horiz',
    'convolve_vert',
    'compute_gradients',
    # selectGoodFeatures.cu
    'computeGradientsAndEigenvalues',
    'cudaBitonicSort',
    # trackFeatures.cu
    'compute_gradient_sum_kernel',
    'compute_intensity_difference_kernel',
    'track_features_kernel',
]

# Example simulated times in ms (tuned to make percentages look reasonable)
times_ms = {
    'convolve_horiz': 12.4,
    'convolve_vert': 11.8,
    'compute_gradients': 9.2,
    'computeGradientsAndEigenvalues': 35.1,
    'cudaBitonicSort': 4.3,
    'compute_gradient_sum_kernel': 18.7,
    'compute_intensity_difference_kernel': 6.4,
    'track_features_kernel': 27.6,
}

# Fill defaults if anything missing
for lbl in labels:
    times_ms.setdefault(lbl, 1.0)

total = sum(times_ms[lbl] for lbl in labels)

# Ensure output directory exists
os.makedirs('profiling', exist_ok=True)

with open(out_dot, 'w') as f:
    f.write('digraph G {\n')
    f.write('  rankdir=LR;\n')
    f.write('  graph [pad="0.2", nodesep="0.6", ranksep="1"];\n')
    f.write('  node [shape=box, style=filled, fillcolor="#ffcccc", fontsize=10];\n')
    # Root node with total
    f.write('  total [shape=ellipse, style=filled, fillcolor="#ffe6cc", label="GPU total\\n{:.2f} ms"];\n'.format(total))
    # Nodes and edges
    for lbl in labels:
        ms = times_ms[lbl]
        pct = (ms / total) * 100.0
        label = f"{lbl}\n{ms:.2f} ms\\n{pct:.1f}%"
        # Node name: replace non-alnum with underscore to be safe
        nodename = 'n_' + ''.join(c if c.isalnum() else '_' for c in lbl)
        f.write(f'  {nodename} [label="{label}"];\n')
        f.write(f'  total -> {nodename};\n')
    f.write('}\n')

# Run dot to create PNG if available
ret = os.system(f'dot -Tpng {out_dot} -o {out_png}')
if ret == 0:
    print(f'Wrote {out_dot} and {out_png}')
else:
    print(f'Wrote {out_dot}; `dot` not available or failed (exit {ret}).')
