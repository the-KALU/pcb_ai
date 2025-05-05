import os
import math
import re
import argparse
import json

def parse_kicad_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Regex to find modules (footprints) and extract positions
    modules = re.findall(
        r'\(module\s+([^\s]+).*?\(at\s+([-\d\.]+)\s+([-\d\.]+)(?:\s+([-\d\.]+))?',
        content,
        re.DOTALL,
    )

    if not modules:
        print(f"⚠️ No modules found in {file_path}. Skipping.")
        return []

    # Parse all x, y first to find bounding box
    positions = [(float(x), float(y)) for (_, x, y, _) in modules]
    if not positions:
        print(f"⚠️ No component positions found in {file_path}. Skipping.")
        return []
    xs, ys = zip(*positions)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    nodes = []
    for idx, (footprint, x, y, rot) in enumerate(modules):
        x, y = float(x), float(y)
        rot = float(rot) if rot else 0.0

        # Normalize into [0, 1] layout space
        norm_x = (x - min_x) / range_x
        norm_y = (y - min_y) / range_y

        # Create feature vector (fixed at 180 length)
        base_features = [0] * 7 + [norm_x, norm_y, rot / 360.0]  # 10 features
        padded_features = base_features + [0.0] * (180 - len(base_features))  # Total = 180

        nodes.append(
            {
                "id": f"n{idx}",
                "type": "component",
                "features": padded_features,
                "target": [norm_x, norm_y],
                "net": footprint,
            }
        )

    return nodes


def compute_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def classify_net(net_name):
    name = net_name.lower()
    if any(p in name for p in ["vcc", "power", "vin", "3v3", "5v"]):
        return 1
    elif any(g in name for g in ["gnd", "ground"]):
        return 2
    return 0


def classify_wire(distance):
    if distance < 0.2:
        return 0
    elif distance < 0.5:
        return 1
    return 2


def generate_edges(nodes):
    edges = []
    for i, src in enumerate(nodes):
        for j, tgt in enumerate(nodes):
            if i >= j:
                continue

            d = compute_distance(src["target"], tgt["target"])
            norm_d = min(d, 1.0)
            same_net = 1 if src["net"] == tgt["net"] else 0
            net_class = (
                classify_net(src["net"])
                if classify_net(src["net"]) == classify_net(tgt["net"])
                else 0
            )
            wire_class = classify_wire(norm_d)

            edges.append(
                {
                    "source": i,
                    "target": j,
                    "features": [norm_d, same_net, net_class, wire_class],
                }
            )
    return edges


def convert_kicad_to_graph(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".kicad_pcb")]

    for idx, filename in enumerate(files):
        file_path = os.path.join(input_folder, filename)
        nodes = parse_kicad_file(file_path)

        if not nodes or len(nodes) < 2:
            print(f"⚠️ Skipping {filename} — insufficient components.")
            continue

        edges = generate_edges(nodes)

        # Create graph data structure
        graph_data = {
            "nodes": nodes,
            "edges": edges,
        }

        output_path = os.path.join(output_folder, f"graph_{idx}.json")
        with open(output_path, "w") as outfile:
            json.dump(graph_data, outfile, indent=4)

        print(f"✅ {filename} → {output_path} ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KiCad PCB files to graph data.")
    parser.add_argument("--input", default="real_pcbs", help="Input folder with .kicad_pcb files")
    parser.add_argument("--output", default="extracted_graphs", help="Output folder for graph files")
    args = parser.parse_args()

    convert_kicad_to_graph(args.input, args.output)
