import os
import torch
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.pcb_env import PcbEnv
from rl.policy import CustomGNNPolicy


# === 📁 Configurations ===
GRAPH_DIR = "synthetic_graphs"
LOG_DIR = "logs/ppo_run"
MODEL_DIR = "models"
GNN_WEIGHTS_PATH = "gnn_weights.pth"
TOTAL_TIMESTEPS = 50000

# === 📏 Infer max observation dimension ===
graph_files = [os.path.join(GRAPH_DIR, f) for f in os.listdir(GRAPH_DIR) if f.endswith(".json")]
max_obs_dim = 0
for file in graph_files:
    with open(file, 'r') as f:
        data = json.load(f)
        obs = sum(len(n["features"]) for n in data["nodes"])
        max_obs_dim = max(max_obs_dim, obs)
print(f"\U0001F4CF Max observation dimension across graphs: {max_obs_dim}")

# === ♻️ Environment wrapper ===
def make_env():
    return PcbEnv(graph_dir=GRAPH_DIR, fixed_obs_size=max_obs_dim)

env = DummyVecEnv([make_env])

# ✅ Sanity check
check_env(make_env(), warn=True)

# === ♻️ PPO Model with pretrained GNN ===
policy_kwargs = {
    "gnn_weights_path": GNN_WEIGHTS_PATH
}

model = PPO(
    policy=CustomGNNPolicy,
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
    device="cpu"
)

# === 💾 Checkpoint callback ===
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=MODEL_DIR, name_prefix="ppo_gnn")

# === 🚀 Train ===
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

# === ✅ Save final model ===
model.save(os.path.join(MODEL_DIR, "ppo_gnn_final"))
print("\n✅ PPO training complete and model saved.")