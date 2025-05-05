import os
import json
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.pcb_env import PcbEnv, load_graphs_from_folder
from rl.export_kicad import export_to_kicad
from rl.policy import CustomGNNPolicy

# === Config ===
GRAPH_DIR = "synthetic_graphs"
MODEL_PATH = "models/ppo_model.zip"
EXPORT_PATH = "exports/exported_layout.kicad_pcb"
FIXED_OBS_SIZE = 180
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# === Load test graphs ===
graphs = load_graphs_from_folder(GRAPH_DIR)
graph_ids = list(graphs.keys())
if not graph_ids:
    raise ValueError("❌ No graphs found in the folder.")

test_graph_id = random.choice(graph_ids)
print(f"🎯 Picked test graph: {test_graph_id}")
print(f"🔢 Graph has {len(graphs[test_graph_id]['nodes'])} nodes")

# === Build test environment ===
def make_test_env():
    return PcbEnv(
        graphs=graphs,
        fixed_obs_size=FIXED_OBS_SIZE,
        test_mode=True,
        test_graph_filename=test_graph_id,
    )

env = DummyVecEnv([make_test_env])

# === Load trained model ===
print("🔍 Loading trained PPO model...")
model = PPO.load(
    MODEL_PATH,
    env=env,
    device=DEVICE,
    custom_objects={"policy": CustomGNNPolicy},
)
print("✅ Model loaded.")

# === Run test rollout ===
obs = env.reset()
trajectory = []
max_steps = len(graphs[test_graph_id]["nodes"])

for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    positions = env.envs[0].component_positions.copy()
    if positions:
        x_mm, y_mm = positions[-1]
        print(f"📦 Step {step+1} - Placing component {step} at ({x_mm:.2f}, {y_mm:.2f})")
        trajectory.append(positions)
    else:
        print(f"⚠️ Step {step+1} - No position returned.")

    

    if done[0]:
        print("✅ Test finished early.")
        break

# === Export final layout ===
final_positions = trajectory[-1] if trajectory else None
graph_obj = graphs.get(test_graph_id)
expected_nodes = len(graph_obj["nodes"])

if final_positions and len(final_positions) == expected_nodes - 1:
    print(f"💾 Exporting layout to: {EXPORT_PATH}")
    export_to_kicad(final_positions, EXPORT_PATH)
    print("✅ Export complete.")
else:
    if graph_obj:
        
        print(f"❌ Expected {expected_nodes} positions but got {len(final_positions) if final_positions else 0}")
    else:
        print(f"⚠️ Could not find graph '{test_graph_id}' in loaded graphs.")
