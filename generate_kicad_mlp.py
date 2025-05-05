import os
import torch
import random
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.pcb_env import PcbEnv
from rl.mlp_policy import MLPPolicy # Import MLPPolicy
from rl.export_kicad import export_to_kicad

# === Paths ===
MODEL_PATH = "trained_mlp_policy.pth" # Path to your saved MLP policy weights
OUTPUT_FILE = "generated_layout_mlp.kicad_pcb"
GRAPH_DIR = "extracted_graphs"
BOARD_SIZE_MM = 100.0 # Define board size here as well

# === Load Graphs ===
def load_graphs(folder):
    graphs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath) as f:
                    graph = json.load(f)
                edge_index = []
                for edge in graph["edges"]:
                    edge_index.append([int(edge["source"]), int(edge["target"])])
                    edge_index.append([int(edge["target"]), int(edge["source"])])
                graph["edge_index"] = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                graph["num_nodes"] = len(graph["nodes"])
                graphs[filename[:-5]] = graph
            except json.JSONDecodeError as e:
                print(f"Error loading {filename}: {e}")
            except KeyError as e:
                print(f"Error loading {filename}: Missing key {e}")
    return graphs

graphs = load_graphs(GRAPH_DIR)
if not graphs:
    raise ValueError("No graphs loaded.")

# === Setup Env ===
selected_graph_name = random.choice(list(graphs.keys()))
selected_graph = graphs[selected_graph_name]
env = PcbEnv(graph=selected_graph, fixed_obs_size=128) # Use the same fixed_obs_size as training
obs, _ = env.reset()

# === Load Model ===
device = "cpu" # Or "cuda" if you have GPU
policy = MLPPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    hidden_dim=128
).to(device)
policy.load_state_dict(torch.load(MODEL_PATH))
policy.eval() # Set the policy to evaluation mode

# === Generate layout ===
component_positions_normalized = [] # Store normalized component positions
component_positions_mm = [] # Store component positions in millimeters

for _ in range(len(selected_graph['nodes'])):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension
    with torch.no_grad():
        action, _, _ = policy.act(obs_tensor)
    cpu_action = action.cpu().numpy()[0] # Get the normalized action [0, 1]

    component_positions_normalized.append(cpu_action)
    x_mm = cpu_action[0] * BOARD_SIZE_MM
    y_mm = cpu_action[1] * BOARD_SIZE_MM
    component_positions_mm.append((x_mm, y_mm)) # Store scaled positions

    obs, _, done, _, _ = env.step(cpu_action)
    if done:
        break

# === Export final layout ===
export_to_kicad(component_positions_mm, selected_graph, OUTPUT_FILE) # Pass the selected_graph
print(f"\n✅ Final layout exported to: {OUTPUT_FILE}")