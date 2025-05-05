import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNNPolicyExtractor(nn.Module):
    def __init__(self, observation_space, gnn_weights_path, edge_index=None):
        super().__init__()
        in_channels = observation_space.shape[0] if len(observation_space.shape) == 1 else observation_space.shape[-1]
        hidden_channels = 64
        self.features_dim = hidden_channels
        self.edge_index = edge_index

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

        if gnn_weights_path:
            state_dict = torch.load(gnn_weights_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

    def forward(self, x):
        if self.edge_index is None:
            raise ValueError("edge_index is not set. Use set_edge_index before forward.")
        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        x = self.conv1(x, self.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, self.edge_index)
        x = torch.relu(x)
        x = x.unsqueeze(0).transpose(1, 2)
        x = self.pool(x).squeeze()
        return x