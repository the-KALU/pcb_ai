import torch
from rl.gnn_extractor import GNNPolicyExtractor
import gymnasium as gym

# === 🔧 Settings ===
WEIGHTS_PATH = "gnn_weights.pth"         # path to the trained GNN weights
OBS_DIM = 180                             # set this to your current fixed_obs_size
FEATURE_DIM = 64                          # this should match your extractor output

# === 🧪 Dummy Box observation space ===
obs_space = gym.spaces.Box(low=0, high=1, shape=(OBS_DIM,), dtype=float)

# === 🔍 Load extractor and weights ===
extractor = GNNPolicyExtractor(obs_space, gnn_weights_path=WEIGHTS_PATH, feature_dim=FEATURE_DIM)
print("\n✅ GNN weights loaded successfully.")

# === 🧪 Create dummy input and test forward pass ===
dummy_input = torch.rand(1, OBS_DIM)
out = extractor(dummy_input)

print(f"🔎 Output vector shape: {out.shape}")
print(f"🧠 Preview: {out.flatten()[:10]}")
