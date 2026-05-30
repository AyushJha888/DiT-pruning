import json
import sys

with open(r'c:\Users\ayush\Downloads\DiT pruning\dit_thesis\notebooks\01_baseline_benchmark.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'c:\Users\ayush\Downloads\DiT pruning\dit_thesis\nb_output_utf8.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"--- Cell {i+1} ({cell['cell_type']}) ---\n")
        source = "".join(cell.get('source', []))
        out.write(source)
        out.write("\n\n")
