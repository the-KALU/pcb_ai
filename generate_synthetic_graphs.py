import os
import json
import random
import uuid
import math
from datetime import datetime

NUM_GRAPHS = 50 #You can change the value from 50 to whatever number of graphs you want to generate
OUTPUT_DIR = "synthetic_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_FEATURE_DIM = 180
COMPONENT_TYPES = ["resistor", "capacitor", "diode", "IC", "transistor", "LED", "inductor"]
COMPONENT_TYPE_MAP = {
    ctype: [1 if i == idx else 0 for i in range(len(COMPONENT_TYPES))]
    for idx, ctype in enumerate(COMPONENT_TYPES)
}
NET_OPTIONS = ["VCC", "GND", "SIGNAL1", "SIGNAL2", "3V3", "5V", "VIN", "GND2", "IO1", "IO2"]

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

def generate_graph(graph_id, min_nodes=6, max_nodes=20):
    num_nodes = random.randint(min_nodes, max_nodes)
    nodes = []

    # Generate positions in a bounded 100x100 space
    positions = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]
    xs, ys = zip(*positions)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x or 1.0
    range_y = max_y - min_y or 1.0

    for i, (x, y) in enumerate(positions):
        norm_x = (x - min_x) / range_x
        norm_y = (y - min_y) / range_y
        rotation = random.uniform(0, 360)
        component_type = random.choice(COMPONENT_TYPES)
        net = random.choice(NET_OPTIONS)

        base_features = [0] * 7 + [norm_x, norm_y, rotation / 360.0]
        padded_features = base_features + [0.0] * (FIXED_FEATURE_DIM - len(base_features))

        nodes.append({
            "id": f"n{i}",
            "type": "component",
            "features": padded_features,
            "target": [norm_x, norm_y],
            "net": net,
        })

    # Generate complete edge list (undirected)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            src, tgt = nodes[i], nodes[j]
            d = compute_distance(src["target"], tgt["target"])
            norm_d = min(d, 1.0)
            same_net = int(src["net"] == tgt["net"])
            net_class = classify_net(src["net"]) if classify_net(src["net"]) == classify_net(tgt["net"]) else 0
            wire_class = classify_wire(norm_d)

            edges.append({
                "source": i,
                "target": j,
                "features": [norm_d, same_net, net_class, wire_class],
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "graph_id": f"pcb_graph_{graph_id}",
            "uuid": str(uuid.uuid4()),
            "timestamp": str(datetime.now()),
            "num_nodes": num_nodes,
            "num_edges": len(edges)
        }
    }

def generate_graph_dataset(num_graphs= NUM_GRAPHS):
    for i in range(num_graphs):
        graph = generate_graph(graph_id=i)
        with open(os.path.join(OUTPUT_DIR, f"graph_{i+20}.json"), "w") as f:
            json.dump(graph, f, indent=2)
    print(f"✅ Generated {num_graphs} KiCad-style synthetic graphs in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    generate_graph_dataset(num_graphs= NUM_GRAPHS)
