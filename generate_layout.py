import os
import torch
from stable_baselines3 import PPO
from rl.pcb_env import PcbEnv
from rl.policy import CustomGNNPolicy
from rl.export_kicad import export_to_kicad

# === Paths ===
MODEL_PATH = "models/ppo_gnn_final"
OUTPUT_FILE = "output_layout.kicad_pcb"
GRAPH_DIR = "synthetic_graphs"

# === Setup Env ===
env = PcbEnv(graph_dir=GRAPH_DIR, fixed_obs_size=180)
obs, _ = env.reset()

# === Load Model ===
model = PPO.load(MODEL_PATH, custom_objects={"policy": CustomGNNPolicy}, device="cpu")

# === Generate layout ===
for _ in range(env.num_components):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)

# === Export final layout ===
export_to_kicad(env.component_positions, OUTPUT_FILE)
print(f"\n✅ Final layout exported to: {OUTPUT_FILE}")
