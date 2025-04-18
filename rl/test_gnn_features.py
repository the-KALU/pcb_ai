# test_gnn_features.py

import torch
import numpy as np
import json
import os
from rl.policy import GNNPolicyExtractor
from gymnasium import spaces

# Load a real graph file
GRAPH_PATH = os.path.join("synthetic_graphs", "graph_0.json")
with open(GRAPH_PATH, "r") as f:
    graph = json.load(f)

# Extract node features
node_features = np.array([n["features"] for n in graph["nodes"]], dtype=np.float32)
num_nodes, features_per_node = node_features.shape
flat_obs = node_features.flatten()

# Create a dummy observation space
observation_space = spaces.Box(low=0.0, high=1.0, shape=(flat_obs.shape[0],), dtype=np.float32)

# Instantiate the GNN feature extractor with the correct input_dim
extractor = GNNPolicyExtractor(observation_space, input_dim=features_per_node)

# 🔧 Lazy init the GCN layers if needed
if extractor.conv1 is None or extractor.conv2 is None:
    extractor.conv1 = extractor._build_conv(features_per_node, 64)
    extractor.conv2 = extractor._build_conv(64, 64)

# Build a fully-connected edge_index (excluding self-loops)
row, col = [], []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            row.append(i)
            col.append(j)
edge_index = torch.tensor([row, col], dtype=torch.long)

# Run forward pass
x = torch.tensor(node_features, dtype=torch.float32)
extractor.eval()
with torch.no_grad():
    x = extractor.relu(extractor.conv1(x, edge_index))
    x = extractor.relu(extractor.conv2(x, edge_index))
    pooled = x.mean(dim=0)  # Global mean pooling

print("\n✅ GNN output vector shape:", pooled.shape)
print("🔍 GNN embedding preview:", pooled[:10])
