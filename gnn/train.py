import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from gnn.model import ComponentPlacementGNN
from gnn.dataset import load_pcb_graph
import os

# === CONFIG ===
GRAPH_DIR = "./data"  # where pcb_graph.json files live
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 1

# === Load Graph Files ===
graph_files = [os.path.join(GRAPH_DIR, f) for f in os.listdir(GRAPH_DIR) if f.endswith(".json")]
dataset = [load_pcb_graph(f) for f in graph_files]
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# === Model Init ===
input_dim = dataset[0].x.shape[1]  # e.g. 2 features per node
model = ComponentPlacementGNN(input_dim=input_dim)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# === Dummy target generator (to simulate labels)
def generate_dummy_targets(data):
    return torch.rand(data.x.size(0), 2)  # e.g. [x, y] between 0 and 1

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        target = generate_dummy_targets(batch)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# === Save Model ===
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/gnn_model.pt")
print("✅ Training complete and model saved.")
