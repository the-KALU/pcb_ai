import matplotlib.pyplot as plt
import re
import json
import os
import glob

def validate_kicad_file(file_path="generated_layout_mlp.kicad_pcb"):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        first_nonblank_line = next((line.strip() for line in lines if line.strip()), '')
        if not first_nonblank_line.startswith("(kicad_pcb"):
            return False, "❌ File does not start with kicad_pcb", 0


        module_count = content.count("(module ")
        edge_cuts = content.count("(layer Edge.Cuts)")

        if module_count == 0:
            return False, "❌ No modules found.", 0

        if edge_cuts < 4:
            return False, "❌ Board outline incomplete.", module_count

        if not content.strip().endswith(")"):
            return False, "❌ File does not end properly.", module_count

        return True, f"✅ Valid KiCad file. Found {module_count} modules and board outline.", module_count

    except Exception as e:
        return False, f"❌ Validation error: {str(e)}", 0

def find_best_graph_file(graph_folder="extracted_graphs", num_components=10):
    candidates = glob.glob(os.path.join(graph_folder, "*.json"))
    closest_match = None
    closest_diff = float('inf')

    for graph_path in candidates:
        try:
            with open(graph_path, 'r') as f:
                data = json.load(f)
            n_nodes = len(data["nodes"])
            diff = abs(n_nodes - num_components)
            if diff < closest_diff:
                closest_diff = diff
                closest_match = graph_path
        except Exception:
            continue

    return closest_match

def preview_kicad_layout(file_path="generated_layout_mlp.kicad_pcb", graph_file=None):
    try:
        with open(file_path, "r") as f:
            content = f.read()

        positions = re.findall(r'\(at ([\d\.\-]+) ([\d\.\-]+)\)', content)
        labels = re.findall(r'\(fp_text reference "([^"]+)"', content)

        if not positions:
            print("❌ No component positions found.")
            return

        if len(labels) != len(positions):
            print(f"⚠️ Mismatch: {len(labels)} labels vs {len(positions)} positions.")

        xs, ys = zip(*[(float(x), float(y)) for x, y in positions])
        id_to_pos = {label: (float(x), float(y)) for label, (x, y) in zip(labels, positions)}

        # Board outline
        edges = re.findall(r'\(gr_line \(start ([\d\.\-]+) ([\d\.\-]+)\) \(end ([\d\.\-]+) ([\d\.\-]+)', content)
        x_edges, y_edges = [], []
        for edge in edges:
            x1, y1, x2, y2 = map(float, edge)
            x_edges += [x1, x2]
            y_edges += [y1, y2]
        min_x, max_x = min(x_edges, default=0), max(x_edges, default=100)
        min_y, max_y = min(y_edges, default=0), max(y_edges, default=100)

        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(xs, ys, c='blue', s=120, marker='s', edgecolors='black')
        for label, (x, y) in zip(labels, positions):
            plt.text(float(x)+1, float(y)+1, label, fontsize=8, ha='left', va='bottom')

        # Outline
        plt.plot([min_x, max_x], [min_y, min_y], color='red')
        plt.plot([max_x, max_x], [min_y, max_y], color='red')
        plt.plot([max_x, min_x], [max_y, max_y], color='red')
        plt.plot([min_x, min_x], [max_y, min_y], color='red')

        # Nets
        if graph_file and os.path.exists(graph_file):
            with open(graph_file, 'r') as gf:
                graph = json.load(gf)

            node_id_to_idx = {node['id']: idx for idx, node in enumerate(graph['nodes'])}
            idx_to_label = {idx: f"C{idx}" for idx in range(len(graph['nodes']))}

            for edge in graph['edges']:
                src = edge['source']
                tgt = edge['target']
                src_idx = node_id_to_idx.get(src)
                tgt_idx = node_id_to_idx.get(tgt)
                if src_idx is not None and tgt_idx is not None:
                    src_label = idx_to_label.get(src_idx)
                    tgt_label = idx_to_label.get(tgt_idx)
                    if src_label in id_to_pos and tgt_label in id_to_pos:
                        x1, y1 = id_to_pos[src_label]
                        x2, y2 = id_to_pos[tgt_label]
                        plt.plot([x1, x2], [y1, y2], 'g--', linewidth=1)

        plt.title("PCB Layout Preview (Components + Nets)")
        plt.xlim(min_x - 10, max_x + 10)
        plt.ylim(min_y - 10, max_y + 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        plt.show()

    except Exception as e:
        print(f"❌ Preview error: {str(e)}")

if __name__ == "__main__":
    valid, message, module_count = validate_kicad_file()
    print(message)
    if valid:
        graph_path = find_best_graph_file("extracted_graphs", num_components=module_count)
        if graph_path:
            print(f"🎯 Auto-picked matching graph: {graph_path}")
            preview_kicad_layout(graph_file=graph_path)
        else:
            print("❌ No matching graph file found.")
