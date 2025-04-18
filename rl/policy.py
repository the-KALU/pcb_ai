# rl/policy.py
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym.spaces import Box


class GNNPolicyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, gnn_weights_path="gnn_weights.pth"):
        super().__init__(observation_space, features_dim=64)
        self.features_per_node = 9  # Set this to match gnn_train.py
        self.conv1 = GCNConv(self.features_per_node, 64)
        self.conv2 = GCNConv(64, 64)
        self.relu = nn.ReLU()

        # Load pretrained weights (trained on 9 features per node)
        state_dict = torch.load(gnn_weights_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, dtype=torch.float32)

        batch_size = observations.shape[0]
        total_features = observations.shape[1]

        num_nodes = total_features // self.features_per_node
        x = observations.view(batch_size, num_nodes, self.features_per_node).squeeze(0)
        edge_index = self._mock_edge_index(num_nodes)

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        return x.mean(dim=0, keepdim=True)  # Return global mean pooled embedding

    def _mock_edge_index(self, num_nodes):
        row, col = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    row.append(i)
                    col.append(j)
        return torch.tensor([row, col], dtype=torch.long)


class CustomGNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: Box, action_space: Box, lr_schedule, gnn_weights_path=None, freeze_gnn=False, **kwargs):
        input_dim = observation_space.shape[0]  # Flattened feature vector length
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNPolicyExtractor,
            features_extractor_kwargs=dict(
                input_dim=input_dim,
                gnn_weights_path=gnn_weights_path,
                freeze_gnn=freeze_gnn
            ),
            **kwargs,
        )
