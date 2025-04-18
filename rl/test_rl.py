import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from rl.pcb_env import PcbEnv
from rl.policy import CustomGNNPolicy

MODEL_WEIGHTS = "models/ppo_run_20250418-022941/policy_weights.pt"
GRAPH_DIR = "synthetic_graphs"

# Make test env with same fixed_obs_size used in training
def make_env():
    return Monitor(PcbEnv(graph_dir=GRAPH_DIR, fixed_obs_size=180))

env = DummyVecEnv([make_env])

# Recreate the PPO model with same architecture (but untrained)
model = PPO(
    policy=CustomGNNPolicy,
    env=env,
    verbose=0,
)

# Load just the trained weights
model.policy.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))

# Test run
obs = env.reset()
total_reward = 0

for step in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    print(f"Step {step+1}: Reward = {reward[0]:.3f}")

print(f"\n🎯 Total Reward after 20 steps: {total_reward:.2f}")

# Optional render
env.envs[0].render()
