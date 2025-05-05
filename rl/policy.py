import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNPolicy(torch.nn.Module):
    def __init__(self, observation_space, action_space, gnn_weights_path, edge_index, net_arch):
        super(GNNPolicy, self).__init__()
        self.fixed_obs_size = observation_space.shape[1] # 128
        in_channels = 128
        print(f"GNNPolicy initialized with in_channels for GCN: {in_channels}") # Debug
        hidden_channels = net_arch[0]
        out_channels_gnn = net_arch[-1]
        self.gnn = GCNNetwork(in_channels, hidden_channels, out_channels_gnn, gnn_weights_path, edge_index)
        self.fc1_actor = nn.Linear(out_channels_gnn, net_arch[1])
        self.fc2_actor = nn.Linear(net_arch[1], action_space.shape[0])

        self.fc1_critic = nn.Linear(out_channels_gnn, net_arch[1])
        self.fc2_critic = nn.Linear(net_arch[1], 1)

    def act(self, obs, edge_index):
        print(f"GNNPolicy act input obs shape: {obs.shape}") # Debug
        action_mean, value = self.forward(obs, edge_index)
        action = torch.clamp(torch.normal(action_mean, 0.1), 0.0, 1.0)
        return action, value

    def forward(self, obs, edge_index, num_nodes=None):
        num_nodes_graph = edge_index.max().item() + 1
        graph_obs = obs[:, :num_nodes_graph, :] # Shape: (1, num_nodes_graph, 128)
        x = graph_obs.squeeze(0) # Shape: (num_nodes_graph, 128)
        print(f"GNNPolicy forward input x shape to GNN: {x.shape}") # Debug
        features = self.gnn(x, edge_index, num_nodes=num_nodes_graph)
        actor_values = F.relu(self.fc1_actor(features))
        action_means = torch.softmax(self.fc2_actor(actor_values), dim=-1)
        critic_values = F.relu(self.fc1_critic(features))
        state_values = self.fc2_critic(critic_values).squeeze(-1)
        return action_means, state_values
    

    def _build_mlp(self, input_dim, output_dim, net_arch):
            layers = []
            prev_dim = input_dim
            for size in net_arch:
                layers.append(nn.Linear(prev_dim, size))
                layers.append(nn.Tanh())
                prev_dim = size
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)

class GCNNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_weights_path, edge_index):
        super(GCNNetwork, self).__init__()
        print(f"GCNNetwork initialized with in_channels: {in_channels}") # Debug
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, bias=True)
        self.lin = nn.Linear(out_channels, 2)
        # self.load_state_dict(torch.load(gnn_weights_path)) # Removed

    def forward(self, x, edge_index, num_nodes=None):
        print(f"GCNNetwork forward input shape (x): {x.shape}") # Debug
        x = self.conv1(x, edge_index)
        print(f"GCNNetwork forward after conv1 shape (x): {x.shape}") # Debug
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        print(f"GCNNetwork forward after conv2 shape (x): {x.shape}") # Debug
        x = F.relu(x)
        x = self.lin(x)
        return x
