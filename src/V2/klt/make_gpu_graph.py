import pandas as pd

df = pd.read_csv('nvprof_output.csv', skiprows=3)
df = df[['Name', 'Time(%)', 'Calls']]
with open('gpu_callgraph.dot', 'w') as f:
    f.write('digraph G {\n')
    for _, r in df.iterrows():
        f.write(f'"{r["Name"]}" [label="{r["Name"]}\\nTime: {r["Time(%)"]}%\\nCalls: {r["Calls"]}"];\n')
    f.write('}\n')
