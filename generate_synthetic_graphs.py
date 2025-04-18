import os
import json
import random
import uuid
from datetime import datetime

# Define component types and their one-hot encodings
COMPONENT_TYPES = ["resistor", "capacitor", "diode", "IC", "transistor", "LED", "inductor"]
COMPONENT_TYPE_MAP = {
    ctype: [1 if i == idx else 0 for i in range(len(COMPONENT_TYPES))]
    for idx, ctype in enumerate(COMPONENT_TYPES)
}

# Define signal classes
SIGNAL_CLASSES = ["power", "signal", "clock"]

# Output directory
OUTPUT_DIR = "synthetic_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_graph(graph_id, min_nodes=4, max_nodes=20):
    num_nodes = random.randint(min_nodes, max_nodes)
    max_edges = num_nodes * (num_nodes - 1) // 3  # sparse
    num_edges = random.randint(num_nodes, max_edges)

    nodes = []
    for i in range(num_nodes):
        x = round(random.uniform(0, 100), 2)
        y = round(random.uniform(0, 100), 2)
        component_type = random.choice(COMPONENT_TYPES)
        nodes.append({
            "id": f"n{i}",
            "type": component_type,
            "features": COMPONENT_TYPE_MAP[component_type] + [x / 100, y / 100],
            "target": [round(x + random.uniform(-10, 10), 2) / 100, round(y + random.uniform(-10, 10), 2) / 100],
            "net": random.choice(["VCC", "GND", "SIGNAL1", "SIGNAL2"])
        })

    edges = set()
    while len(edges) < num_edges:
        a, b = random.sample(range(num_nodes), 2)
        if (a, b) not in edges and (b, a) not in edges:
            trace_width = round(random.uniform(0.15, 1.0), 2)
            signal_type_idx = random.randint(0, len(SIGNAL_CLASSES) - 1)
            edges.add((a, b, trace_width, signal_type_idx))

    edge_list = [
        {
            "source": f"n{a}",
            "target": f"n{b}",
            "features": [trace_width / 1.0, signal_type_idx]  # normalize width
        }
        for a, b, trace_width, signal_type_idx in edges
    ]

    return {
        "nodes": nodes,
        "edges": edge_list,
        "metadata": {
            "graph_id": f"pcb_graph_{graph_id}",
            "uuid": str(uuid.uuid4()),
            "timestamp": str(datetime.now()),
            "num_nodes": num_nodes,
            "num_edges": len(edge_list)
        }
    }

def generate_graph_dataset(num_graphs=30):
    for i in range(num_graphs):
        graph = generate_graph(graph_id=i)
        with open(os.path.join(OUTPUT_DIR, f"graph_{i}.json"), "w") as f:
            json.dump(graph, f, indent=2)

    print(f"✅ Generated {num_graphs} GNN-ready PCB graphs in '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    generate_graph_dataset(num_graphs=30)
