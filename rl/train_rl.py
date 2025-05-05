import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gymnasium import spaces
from rl.gnn_extractor import GNNPolicyExtractor
from rl.pcb_env import load_graphs_from_folder

# === Config ===
GRAPH_DIR = "extracted_graphs"
OUTPUT_PATH = "models/pretrained_gnn.pth"
FEATURE_DIM = 180     # Input dim (raw node features)
EXTRACTED_DIM = 64    # Output feature dim to match CustomGNNPolicy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

# === Load all node features ===
graphs = load_graphs_from_folder(GRAPH_DIR)
all_features = []

for graph in graphs.values():
    for node in graph["nodes"]:
        f = node["features"]
        if len(f) == FEATURE_DIM:
            all_features.append(f)

all_features = torch.tensor(all_features, dtype=torch.float32)
print(f"✨ Total nodes loaded: {len(all_features)} | Feature dim: {FEATURE_DIM}")

# === Dummy target: identity (self-supervised pretraining) ===
targets = all_features.clone()

dataset = TensorDataset(all_features, targets)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Instantiate GNN Extractor ===
obs_space = spaces.Box(low=0.0, high=1.0, shape=(FEATURE_DIM,), dtype=np.float32)
model = GNNPolicyExtractor(observation_space=obs_space, feature_dim=EXTRACTED_DIM).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(x_batch)  # forward() does GCN + ReLU
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📚 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# === Save pretrained GNN ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"✅ GNN saved to {OUTPUT_PATH}")
