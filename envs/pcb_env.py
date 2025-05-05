# 📂 envs/pcb_env.py
import gym
import numpy as np
import torch
from gym import spaces

class PCBPlacementEnv(gym.Env):
    def __init__(self, graph_dict, feature_extractor):
        super().__init__()
        self.graph = graph_dict
        self.feature_extractor = feature_extractor

        self.node_feats = torch.tensor([n["features"] for n in self.graph["nodes"]], dtype=torch.float32)
        self.edge_index = torch.tensor([[e["source"] for e in self.graph["edges"]] + [e["target"] for e in self.graph["edges"]],
                                        [e["target"] for e in self.graph["edges"]] + [e["source"] for e in self.graph["edges"]]], dtype=torch.long)

        self.num_nodes = len(self.graph["nodes"])
        self.current_idx = 0
        self.placements = torch.zeros((self.num_nodes, 2))  # x, y placement per node

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=self.node_feats.shape, dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.num_nodes, shape=self.edge_index.shape, dtype=np.int64),
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_idx = 0
        self.placements = torch.zeros((self.num_nodes, 2))
        return {"node_features": self.node_feats, "edge_index": self.edge_index}

    def step(self, action):
        self.placements[self.current_idx] = torch.tensor(action)
        self.current_idx += 1

        done = self.current_idx >= self.num_nodes
        reward = -self.compute_overlap_penalty() if done else 0
        obs = {"node_features": self.node_feats, "edge_index": self.edge_index}
        return obs, reward, done, {}

    def compute_overlap_penalty(self):
        penalty = 0
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if torch.dist(self.placements[i], self.placements[j]) < 0.05:
                    penalty += 1
        return penalty

    def render(self, mode="human"):
        print("Placements:", self.placements.tolist())
