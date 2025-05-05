import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MLPPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=64):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0] * observation_space.shape[1], hidden_dim) # Flattened observation
        self.fc2_actor = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_space.shape[0])
        self.fc_std = nn.Linear(hidden_dim, action_space.shape[0]) # Learnable standard deviation

        self.fc3_critic = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

        self.action_scale_internal = torch.tensor(1.0, dtype=torch.float32) # Internal scale for tanh output
        self.board_scale_policy = torch.tensor(0.5, dtype=torch.float32) # Scale tanh output to +/- 0.5, then shift
        self.action_bias_policy = torch.tensor(0.5, dtype=torch.float32) # Shift to be within 0 to 1

        self.clip_limit = 1.0 # Clip actions within the [0, 1] range

    def forward(self, obs):
        batch_size = obs.size(0)
        flattened_obs = obs.view(batch_size, -1) # Flatten the observation
        x = F.relu(self.fc1(flattened_obs))

        # Actor
        actor_x = F.relu(self.fc2_actor(x))
        # Scale the mean output using tanh to be within +/- 1, then scale and shift
        action_mean_raw = self.fc_mean(actor_x)
        action_mean = torch.tanh(action_mean_raw / self.action_scale_internal) * self.board_scale_policy + self.action_bias_policy

        action_std = F.softplus(self.fc_std(actor_x)) + 1e-5 # Ensure positive std

        # Critic
        critic_x = F.relu(self.fc3_critic(x))
        state_value = self.fc_value(critic_x).squeeze(1)

        return action_mean, action_std, state_value

    def act(self, obs):
        action_mean, action_std, value = self.forward(obs)
        distribution = Normal(action_mean, action_std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)

        clipped_action = torch.clamp(action, 0.0, self.clip_limit)

        return clipped_action, log_prob, value