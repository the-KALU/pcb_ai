import torch
import numpy as np
import json
from rl.policy import GNNPolicyExtractor

# === Load Graph ===
with open("synthetic_graphs/graph_0.json", "r") as f:
    graph = json.load(f)

nodes = graph["nodes"]
features_per_node = len(nodes[0]["features"])
num_nodes = len(nodes)
input_dim = features_per_node  # ✅ CORRECTED: pass per-node feature count!

flat_features = np.array([n["features"] for n in nodes]).flatten().astype(np.float32)
x = torch.tensor(flat_features, dtype=torch.float32).view(num_nodes, features_per_node)

# === Dummy edge_index ===
edge_index = torch.tensor([
    [i for i in range(num_nodes) for j in range(num_nodes) if i != j],
    [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
], dtype=torch.long)

# === Initialize with correct per-node input dim
extractor = GNNPolicyExtractor(observation_space=None, input_dim=input_dim)

# === Forward pass
with torch.no_grad():
    x = extractor.relu(extractor.conv1(x, edge_index))
    x = extractor.relu(extractor.conv2(x, edge_index))
    pooled = x.mean(dim=0)

print("✅ GNN output vector shape:", pooled.shape)
print("🔍 GNN embedding preview:", pooled[:10])

# === Save weights
torch.save(extractor.state_dict(), "gnn_weights.pth")
print("💾 Saved to gnn_weights.pth")
