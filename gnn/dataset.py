import torch
from torch_geometric.data import Data
import json

def load_pcb_graph(json_path):
    with open(json_path) as f:
        graph = json.load(f)

    node_map = {node['id']: idx for idx, node in enumerate(graph['nodes'])}

    x = []
    for node in graph['nodes']:
        # Example: One-hot component type (resistor, cap, ic), pin count
        comp_type = node.get("type", "component")
        pin_count = node.get("pins", 2)
        feature = [
            1 if comp_type == "component" else 0,  # You can improve this
            pin_count / 20  # Normalize
        ]
        x.append(feature)

    edge_index = [[], []]
    for edge in graph['edges']:
        src = node_map[edge['source']]
        tgt = node_map[edge['target']]
        edge_index[0].append(src)
        edge_index[1].append(tgt)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )
    return data
