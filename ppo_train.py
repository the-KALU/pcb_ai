import os
import torch
import random
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.pcb_env import PcbEnv
from rl.mlp_policy import MLPPolicy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json
import torch.nn as nn

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
                graphs[filename[:-5]] = graph
            except json.JSONDecodeError:
                print(f"Warning: {filename} is not a valid JSON file and will be skipped.")
            except KeyError as e:
                print(f"Warning: {filename} is missing a required key ({e}). Skipping.")
    if not graphs:
        raise ValueError(f"No valid graphs loaded from {folder}.")
    return graphs

graph_dir = "extracted_graphs"
graphs = load_graphs(graph_dir)
if not graphs:
    raise ValueError("No graphs loaded.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

selected_graph_name = random.choice(list(graphs.keys()))
selected_graph = graphs[selected_graph_name]
print(f"Selected graph: {selected_graph_name} with {len(selected_graph['nodes'])} nodes")

def make_env():
    return PcbEnv(graph=selected_graph, fixed_obs_size=128) # Consistent fixed_obs_size

env = DummyVecEnv([make_env])

# Hyperparameters (you can tune these)
learning_rate = 0.0003
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
num_epochs = 3
batch_size = 64
trajectory_len = 1024  # Or your desired trajectory length

policy = MLPPolicy( # Changed to MLPPolicy
    observation_space=env.observation_space,
    action_space=env.action_space,
    hidden_dim=128 # You can adjust this
).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(len(rewards) - 1)):
        next_non_terminal = 1.0 - dones[t + 1]
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    return advantages

def train_ppo(env, policy, optimizer, num_epochs, batch_size, trajectory_len, gamma, gae_lambda, clip_epsilon, device):
    policy.train()
    total_steps = trajectory_len * num_epochs
    global_step = 0


        # Get board dimensions from the environment (assuming it's accessible)
    board_width = env.envs[0].board_width  # Access the underlying env
    board_height = env.envs[0].board_height
    center_x = board_width / 2
    center_y = board_height / 2
    CENTERING_REWARD_SCALE = 0.1  # Adjust as needed
    

    # --- TENSORBOARD SETUP ---
    writer = SummaryWriter(log_dir="ppo_training_logs")  # Create a SummaryWriter
    # --- END TENSORBOARD SETUP ---

    for epoch in range(num_epochs):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []
        episode_values = []
        dones = False
        episode_terminated = [False] * env.num_envs
        steps_in_episode = 0

        for step in range(trajectory_len):
            action, log_prob, value = policy.act(obs)
            cpu_actions = action.cpu().detach().numpy() # Detach before converting to NumPy
            next_obs, reward, terminated, truncated = env.step(cpu_actions)
            done = np.logical_or(terminated, truncated)


                        # --- CENTERING REWARD ---
            # Calculate distance to center and add to reward
            x = cpu_actions[:, 0] * board_width  # Scale back to board coords
            y = cpu_actions[:, 1] * board_height
            dist_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            center_reward = np.exp(-0.01 * dist_to_center) * CENTERING_REWARD_SCALE  # Exponential reward
            reward += center_reward
            # --- END CENTERING REWARD ---

            episode_obs.append(obs.cpu().numpy())
            episode_actions.append(cpu_actions)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob.cpu().detach().numpy()) # Detach here as well
            episode_values.append(value.squeeze().cpu().detach().numpy()) # Detach here too
            episode_terminated = done
            steps_in_episode += 1

            obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            global_step += 1
            if np.all(episode_terminated):
                break

        # Convert collected data to tensors only if we have some data
        if episode_obs:
            batch_obs = torch.tensor(np.array(episode_obs), dtype=torch.float32).to(device)
            batch_actions = torch.tensor(np.array(episode_actions), dtype=torch.float32).to(device)
            batch_rewards = torch.tensor(np.array(episode_rewards), dtype=torch.float32).to(device).unsqueeze(1)
            batch_log_probs = torch.tensor(np.array(episode_log_probs), dtype=torch.float32).to(device).unsqueeze(1)
            batch_values = torch.tensor(np.array(episode_values), dtype=torch.float32).to(device).unsqueeze(1)
            batch_dones = torch.tensor(np.array([False] * len(episode_rewards)), dtype=torch.float32).to(device).unsqueeze(1) # Assuming single env

            advantages = compute_gae(batch_rewards, batch_values, batch_dones, gamma, gae_lambda)
            returns = advantages + batch_values

            # Flatten the batch
            b_obs = batch_obs.reshape(-1, *env.observation_space.shape)
            b_actions = batch_actions.reshape(-1, env.action_space.shape[0])
            b_log_probs = batch_log_probs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = batch_values.reshape(-1)

            # Optimize policy
            b_inds = np.arange(len(b_obs))
            for _ in range(num_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, len(b_obs), batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]

                    _, _, new_values = policy(b_obs[mb_inds])
                    new_action_mean, new_action_std, _ = policy(b_obs[mb_inds])
                    distribution = torch.distributions.Normal(new_action_mean, new_action_std)
                    new_log_probs = distribution.log_prob(b_actions[mb_inds]).sum(dim=1)
                    entropy = distribution.entropy().mean()
                    logratio = new_log_probs - b_log_probs[mb_inds]
                    ratio = logratio.exp()

                    approx_kl = (b_log_probs[mb_inds] - new_log_probs).mean()

                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]

                    pg_loss1 = mb_advantages * ratio
                    pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                    critic_loss = F.mse_loss(new_values, mb_returns)

                    loss = pg_loss + 0.5 * critic_loss - 0.01 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()


                                        # --- TENSORBOARD LOGGING ---
                    writer.add_scalar("Loss/Policy", pg_loss.item(), global_step)
                    writer.add_scalar("Loss/Value", critic_loss.item(), global_step)
                    writer.add_scalar("Entropy", entropy.item(), global_step)
                    writer.add_scalar("KL Divergence", approx_kl.item(), global_step)
                    writer.add_scalar("Reward/Mean", np.mean(reward), global_step)  # Assuming 'reward' is available here
                    # --- END TENSORBOARD LOGGING ---

            print(f"Epoch: {epoch}, Policy Loss: {pg_loss.item()}, Value Loss: {critic_loss.item()}, Entropy: {entropy.item()}")
        else:
            print("Warning: No data collected in this episode. Skipping optimization.")

    # Save the trained policy
    save_path = "trained_mlp_policy.pth"
    torch.save(policy.state_dict(), save_path)
    print(f"PPO training complete with MLP Policy. Trained model saved to {save_path}")

train_ppo(env, policy, optimizer, num_epochs, batch_size, trajectory_len, gamma, gae_lambda, clip_epsilon, device)

print("PPO training script finished.")