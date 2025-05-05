# 📂 models/custom_policy.py
import torch
import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

class GNNFeatureWrapper(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, gnn_model):
        super().__init__(observation_space, features_dim=gnn_model.conv2.out_channels)
        self.gnn = gnn_model

    def forward(self, observations):
        node_feats = observations["node_features"]
        edge_index = observations["edge_index"]
        return self.gnn(node_feats, edge_index)

class CustomGNNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, gnn_model=None, **kwargs):
        self._custom_gnn = gnn_model
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=GNNFeatureWrapper,
            features_extractor_kwargs={"gnn_model": gnn_model},
            **kwargs,
        )

