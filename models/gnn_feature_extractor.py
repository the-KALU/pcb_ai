# 📂 models/gnn_feature_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNPolicyExtractor(nn.Module):
    def __init__(self, input_dim=180, hidden_dim=128, output_dim=64):
        super(GNNPolicyExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
