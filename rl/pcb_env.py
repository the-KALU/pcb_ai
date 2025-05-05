import gymnasium as gym
import numpy as np
import random
import json
import os
from gymnasium import spaces
import torch

BOARD_SIZE_MM = 100.0
MIN_SPACING_MM = 1.0
MAX_NODES = 320  # Define a maximum number of nodes
OUT_OF_BOUNDS_PENALTY = 1.0

class PcbEnv(gym.Env):
    def __init__(self, graphs=None, graph=None, fixed_obs_size=128, test_mode=False, test_graph_filename=None):
        super().__init__()
        self.graphs = graphs
        self.graph = graph
        self.fixed_obs_size = fixed_obs_size
        self.test_mode = test_mode
        if self.test_mode and test_graph_filename:
            self.graph = self._load_graph_by_filename(test_graph_filename)
        self.current_node_idx = 0
        self.placed_nodes = []
        self.component_positions = []
        # Observation space now has a fixed maximum number of nodes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_NODES, fixed_obs_size), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.max_steps = MAX_NODES * 2
        self.steps_taken = 0
        self.node_features_dim = fixed_obs_size
        self.board_width = BOARD_SIZE_MM
        self.board_height = BOARD_SIZE_MM

    def _load_graph_by_filename(self, filename):
        if self.graphs is None:
            raise ValueError("No graph set or graph list available.")
        for graph in self.graphs.values():
            if graph.get("metadata", {}).get("graph_id") == filename:
                return graph
        raise ValueError(f"Graph with filename '{filename}' not found.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.current_node_idx = 0
        self.placed_nodes = []
        self.component_positions = []
        print(f"Environment Reset - current_node_idx: {self.current_node_idx}, steps_taken: {self.steps_taken}")  # Debugging
        if self.test_mode:
            if self.graph is None:
                raise ValueError("Test mode enabled but no test graph loaded.")
        elif self.graph is not None:
            print(f"🎲 Training mode: using provided graph with {len(self.graph['nodes'])} nodes")
        elif self.graphs is not None:
            self.graph = random.choice(list(self.graphs.values()))
            print(f"🎲 Training mode: selected graph with {len(self.graph['nodes'])} nodes")
        else:
            raise ValueError("No graph or graphs provided to environment.")
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        if self.graph is None or "nodes" not in self.graph:
            raise ValueError("Graph not loaded or missing nodes.")

        num_nodes = len(self.graph["nodes"])
        all_node_features = np.zeros((MAX_NODES, self.fixed_obs_size), dtype=np.float32)  # Initialize with zeros

        for i, node in enumerate(self.graph["nodes"]):
            raw_features = node.get("features", [])
            padded_features = raw_features[:self.fixed_obs_size] + [0.0] * (self.fixed_obs_size - len(raw_features))
            all_node_features[i] = padded_features

        return all_node_features

    def step(self, action):
        print(f"PcbEnv - Step Start: {self.steps_taken}, current_node_idx: {self.current_node_idx}, action: {action}")   # Debug action
        self.steps_taken += 1
        x_norm, y_norm = action
        x_mm = x_norm * self.board_width
        y_mm = y_norm * self.board_height
        if self.current_node_idx < len(self.graph["nodes"]):
            self.component_positions.append((x_mm, y_mm))
            self.placed_nodes.append(self.current_node_idx)
            print(f"PcbEnv - Step: {self.steps_taken}, Placed node: {self.current_node_idx}")
        else:
            print(f"PcbEnv - Step: {self.steps_taken}, Already placed all nodes")

        print(f"PcbEnv - Step: {self.steps_taken}, current_node_idx: {self.current_node_idx}, num_nodes: {len(self.graph['nodes'])}, max_steps: {self.max_steps}")
        print(f"PcbEnv - Before Termination Check: steps_taken: {self.steps_taken}, max_steps: {self.max_steps}")    # Debug steps
        terminated = self.current_node_idx >= len(self.graph["nodes"])
        truncated = self.steps_taken >= self.max_steps
        print(f"PcbEnv - After Termination Check: terminated: {terminated}, truncated: {truncated}")    # Debug termination

        reward = self._compute_reward(x_mm, y_mm)  # Pass current position to reward
        self.current_node_idx += 1
        obs = self._get_obs()
        info = {
            "overlap_penalty": self._compute_overlap_penalty(),
            "spacing_penalty": self._compute_spacing_penalty(),
            "wirelength": self._compute_total_wirelength()
        }
        print(f"PcbEnv - Step End - truncated: {truncated}")
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, x, y):
        reward = 0.0
        grid_size = 10
        cell_width = self.board_width / grid_size
        cell_height = self.board_height / grid_size
        density_penalty_factor = 1  # Reduced slightly
        strong_edge_penalty_threshold = 1.0  # Distance from edge for strong penalty
        strong_edge_penalty = 50.0  # Very high penalty
        center_x = self.board_width / 2.0
        center_y = self.board_height / 2.0
        strong_centrality_reward_threshold = 1.0 # Distance from center for strong reward
        strong_centrality_reward = 500000  # Significant reward for being central
        weak_centrality_reward_factor = 0.005
        centrality_decay_factor = self.board_width / 3
        proximity_penalty_factor = 0.0005 # Reduced slightly
        min_spacing_penalty = MIN_SPACING_MM * 1.5

        def get_cell_index(px, py):
            row = int(py // cell_height)
            col = int(px // cell_width)
            return row, col

        current_row, current_col = get_cell_index(x, y)
        density = 0

        for px_placed, py_placed in self.component_positions:
            placed_row, placed_col = get_cell_index(px_placed, py_placed)
            if abs(placed_row - current_row) <= 1 and abs(placed_col - current_col) <= 1:
                density += 1

        reward -= density * density_penalty_factor

        # EXTREMELY STRONG PENALTY FOR BEING NEAR THE EDGE
        dist_to_edge = min(x, self.board_width - x, y, self.board_height - y)
        if dist_to_edge < strong_edge_penalty_threshold:
            reward -= strong_edge_penalty

        # STRONG REWARD FOR BEING NEAR THE CENTER
        distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if distance_to_center < strong_centrality_reward_threshold:
            reward += strong_centrality_reward
        else:
            # Weak centrality reward for being further out
            reward += weak_centrality_reward_factor * np.exp(-distance_to_center / centrality_decay_factor)

        reward -= self._compute_overlap_penalty()  # Reduced slightly

        return reward

    def _compute_overlap_penalty(self):
        penalty = 0.0
        radius = 1.0
        for i in range(len(self.component_positions)):
            xi, yi = self.component_positions[i]
            for j in range(i + 1, len(self.component_positions)):
                xj, yj = self.component_positions[j]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < radius * 2:
                    penalty += 5.0
        return penalty

    def _compute_spacing_penalty(self):
        penalty = 0.0
        for i in range(len(self.component_positions)):
            xi, yi = self.component_positions[i]
            for j in range(i + 1, len(self.component_positions)):
                xj, yj = self.component_positions[j]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < MIN_SPACING_MM:
                    penalty += (MIN_SPACING_MM - dist)
        return penalty

    def _compute_total_wirelength(self):
        total = 0.0
        if "edges" not in self.graph:
            return total
        for edge in self.graph["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            try:
                src_idx = int(src.replace("n", ""))
                tgt_idx = int(tgt.replace("n", ""))
                if src_idx < len(self.component_positions) and tgt_idx < len(self.component_positions):
                    x1, y1 = self.component_positions[src_idx]
                    x2, y2 = self.component_positions[tgt_idx]
                    total += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            except:
                continue
        return total

def load_graphs_from_folder(folder_path):
    graphs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            with open(path, "r") as f:
                graph = json.load(f)
            graph_id = graph.get("metadata", {}).get("graph_id", os.path.splitext(filename)[0])
            graphs[graph_id] = graph
    return graphs

if __name__ == '__main__':
    # Example usage:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_folder = os.path.join(current_dir, "extracted_graphs") # Assuming 'extracted_graphs' folder in the same directory

    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)
        # Create a dummy graph for testing
        dummy_graph = {
            "metadata": {"graph_id": "dummy_graph"},
            "nodes": [{"id": "n0", "features": [0.1, 0.2]}, {"id": "n1", "features": [0.3, 0.4]}, {"id": "n2", "features": [0.5, 0.6]}],
            "edges": [{"source": "n0", "target": "n1"}, {"source": "n1", "target": "n2"}]
        }
        with open(os.path.join(graphs_folder, "dummy_graph.json"), "w") as f:
            json.dump(dummy_graph, f)

    loaded_graphs = load_graphs_from_folder(graphs_folder)
    if loaded_graphs:
        env = PcbEnv(graphs=loaded_graphs, fixed_obs_size=128)
        obs, _ = env.reset()
        print("Initial Observation Shape:", obs.shape)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print("Next Observation Shape:", next_obs.shape)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", info)
        env.close()
    else:
        print("No graphs loaded. Please ensure the 'extracted_graphs' folder contains JSON graph files.")