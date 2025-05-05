import os
import json
import torch
import torch.nn.functional as F
from torch.nn import Linear, AdaptiveAvgPool1d
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

def load_graphs_from_json(folder):
    data_list = []
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename)) as f:
            graph = json.load(f)

        node_features = torch.tensor([n["features"] for n in graph["nodes"]], dtype=torch.float)
        node_targets = torch.tensor([n["target"] for n in graph["nodes"]], dtype=torch.float)

        edge_index = []
        for edge in graph["edges"]:
            edge_index.append([edge["source"], edge["target"]])
            edge_index.append([edge["target"], edge["source"]])  # undirected

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=node_features, edge_index=edge_index, y=node_targets)
        data_list.append(data)
    return data_list

# === Modified GCN Model Definition ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2): # Added out_channels
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = AdaptiveAvgPool1d(1) # Add pooling layer
        self.lin = Linear(hidden_channels, out_channels) # Linear layer to reduce to 2

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.unsqueeze(0).transpose(1, 2) # Prepare for pooling [batch_size, num_nodes, features] -> [batch_size, features, num_nodes]
        x = self.pool(x).squeeze(-1)       # Pool to [batch_size, features]
        x = self.lin(x)                     # Reduce to [batch_size, out_channels]
        return x

# === Training Script ===
graph_dir = "extracted_graphs"
graphs = load_graphs_from_json(graph_dir)

if len(graphs) == 0:
    raise ValueError("❌ No graphs found in extracted_graphs/. Did you run extract_graph_from_kicad.py?")

print(f"✅ Loaded {len(graphs)} graphs")

in_channels = graphs[0].x.size(-1)  # should be 180
model = GCN(in_channels=in_channels) # out_channels defaults to 2
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loader = DataLoader(graphs, batch_size=4, shuffle=True)

for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch) # shape: [batch_size, 2]

        # Calculate the mean of node targets for each graph in the batch
        graph_level_y = torch.stack([torch.mean(data.y.float(), dim=0) for data in batch.to_data_list()]) # Shape: [batch_size, 2]

        loss = F.mse_loss(out, graph_level_y) # Now the shapes should match
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "gnn_weights.pth")
print("✅ Training complete. Modified model saved to gnn_weights.pth")