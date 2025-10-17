#!/usr/bin/env python3
"""
Simple, dependency-free converter from nvprof CSV/text to a Graphviz DOT file
and PNG. Does not require pandas or external Python packages.

Usage: run in the directory containing `nvprof_output.csv` (or `nvprof_output.txt`)
and a `profiling/` directory (created if missing). The script writes
`profiling/gpu_callgraph.dot` and `profiling/gpu_callgraph.png`.
"""
import csv
import os
import sys
import shutil

CSV = 'nvprof_output.csv'
TXT = 'nvprof_output.txt'
OUT_DIR = 'profiling'
DOT = os.path.join(OUT_DIR, 'gpu_callgraph.dot')
PNG = os.path.join(OUT_DIR, 'gpu_callgraph.png')


def ensure_out_dir():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


def parse_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        # skip possible header junk until a header row containing 'Name' is found
        headers = None
        for r in reader:
            if any('Name' in c for c in r):
                headers = [c.strip() for c in r]
                break
        if headers is None:
            return rows
        # Map columns
        name_i = None
        time_i = None
        calls_i = None
        for i, h in enumerate(headers):
            lh = h.lower()
            if 'name' in lh:
                name_i = i
            if 'time' in lh and time_i is None:
                time_i = i
            if 'calls' in lh:
                calls_i = i
        # read remaining rows
        for r in reader:
            if len(r) <= (name_i or 0):
                continue
            name = r[name_i].strip()
            if not name:
                continue
            time = r[time_i].strip() if time_i is not None and time_i < len(r) else ''
            calls = r[calls_i].strip() if calls_i is not None and calls_i < len(r) else ''
            rows.append((name, time, calls))
    return rows


def parse_txt(path):
    # Fallback parser for nvprof plain text: very simple extraction of lines
    # that look like kernel entries (contain :: or kernel names). This is
    # best-effort and may need manual inspection.
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # ignore warnings
            if line.startswith('========'):
                continue
            # attempt to parse lines like: <time>%  <calls>  <Name>
            parts = line.split()
            if len(parts) >= 3 and parts[0].replace('.', '', 1).isdigit():
                time = parts[0]
                calls = parts[1]
                name = ' '.join(parts[2:])
                rows.append((name, time, calls))
    return rows


def write_dot(rows):
    ensure_out_dir()
    with open(DOT, 'w') as f:
        f.write('digraph G {\n')
        f.write('  node [shape=box, style=filled, fillcolor="#f0f8ff"];\n')
        for name, time, calls in rows:
            # sanitize label
            label = f"{name}\\nTime: {time}\\nCalls: {calls}"
            safe_name = name.replace('"', '\\"')
            f.write(f'  "{safe_name}" [label="{label}"];\n')
        f.write('}\n')
    print(f'Wrote {DOT} ({len(rows)} entries)')


def render_png():
    # Use dot from Graphviz, which is already used for CPU graphs in this repo.
    if shutil.which('dot') is None:
        print('Graphviz `dot` not found in PATH; skipping PNG generation.', file=sys.stderr)
        return
    cmd = f'dot -Tpng {DOT} -o {PNG}'
    print('Running:', cmd)
    rc = os.system(cmd)
    if rc == 0:
        print('Wrote', PNG)
    else:
        print('dot failed (exit code', rc, ')', file=sys.stderr)


def main():
    rows = []
    if os.path.exists(CSV):
        rows = parse_csv(CSV)
    elif os.path.exists(TXT):
        rows = parse_txt(TXT)
    else:
        print('No nvprof_output.csv or nvprof_output.txt found in current dir', file=sys.stderr)
        sys.exit(1)

    if not rows:
        print('No rows parsed; check the nvprof output format.', file=sys.stderr)
        sys.exit(1)

    write_dot(rows)
    render_png()


if __name__ == '__main__':
    main()
