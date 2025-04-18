import gymnasium as gym
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
from gymnasium import spaces


class PcbEnv(gym.Env):
    def __init__(self, graph_dir="synthetic_graphs", board_size=64, fixed_obs_size=None):
        super(PcbEnv, self).__init__()

        self.graph_dir = graph_dir
        self.graph_files = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith(".json")]
        assert self.graph_files, f"No graph files found in directory: {graph_dir}"

        self.board_size = board_size
        self.heatmap = np.zeros((board_size, board_size))
        self.fixed_obs_size = fixed_obs_size

        # Use one graph to infer observation shape
        sample_graph = self._load_graph(self.graph_files[0])
        sample_obs = self._get_flat_features(sample_graph)
        obs_dim = fixed_obs_size if fixed_obs_size else len(sample_obs)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.nodes = []
        self.edges = []
        self.component_positions = []

    def _load_graph(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _get_flat_features(self, graph):
        features = np.array([n["features"] for n in graph["nodes"]]).flatten().astype(np.float32)
        if self.fixed_obs_size and len(features) < self.fixed_obs_size:
            features = np.pad(features, (0, self.fixed_obs_size - len(features)), mode='constant')
        return features

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        graph_file = random.choice(self.graph_files)
        graph = self._load_graph(graph_file)

        self.nodes = graph["nodes"]
        self.edges = graph["edges"]
        self.num_components = len(self.nodes)
        self.component_positions = np.array([n["features"][-2:] for n in self.nodes])

        obs = self._get_flat_features(graph)
        self._update_heatmap()
        return obs, {}

    def step(self, action):
        moved_idx = np.random.randint(0, self.num_components)
        self.component_positions[moved_idx] = action.clip(0.0, 1.0)
        self.nodes[moved_idx]["features"][-2:] = self.component_positions[moved_idx].tolist()

        self._update_heatmap()
        reward, reward_details = self._calculate_reward()
        obs = self._get_flat_features({"nodes": self.nodes})

        self._last_reward_details = reward_details  # for render()

        return obs, reward, False, False, {}

    def _calculate_reward(self):
        reward = 0.0
        wire_penalty = 0.0
        overlap_penalty = 0.0

        for edge in self.edges:
            a = self._get_node_index(edge["source"])
            b = self._get_node_index(edge["target"])
            dist = np.linalg.norm(self.component_positions[a] - self.component_positions[b])
            wire_penalty += dist

        for i in range(self.num_components):
            for j in range(i + 1, self.num_components):
                if np.linalg.norm(self.component_positions[i] - self.component_positions[j]) < 0.05:
                    overlap_penalty += 1.0

        spread_bonus = -np.std(self.heatmap)

        reward = -wire_penalty - overlap_penalty + 0.1 * spread_bonus

        return reward, {
            "wire_penalty": wire_penalty,
            "overlap_penalty": overlap_penalty,
            "spread_bonus": spread_bonus,
            "total_reward": reward
        }

    def _update_heatmap(self):
        self.heatmap = np.zeros((self.board_size, self.board_size))
        for pos in self.component_positions:
            x = int(pos[0] * (self.board_size - 1))
            y = int(pos[1] * (self.board_size - 1))
            self.heatmap[y, x] += 1

    def _get_node_index(self, node_id):
        for i, node in enumerate(self.nodes):
            if node["id"] == node_id:
                return i
        raise ValueError(f"Node ID {node_id} not found in graph.")

    def render(self):
        print("📊 Reward breakdown:")
        if hasattr(self, "_last_reward_details"):
            for k, v in self._last_reward_details.items():
                print(f"  {k}: {v:.3f}")
        else:
            print("  No reward data available yet.")

        plt.imshow(self.heatmap, cmap='hot', interpolation='nearest')
        plt.title("📍 Component Placement Heatmap")
        plt.colorbar(label='Component Count')
        plt.pause(0.01)
        plt.clf()
