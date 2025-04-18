# rl/reward_utils.py
import numpy as np

def compute_total_wire_length(component_positions, edge_list):
    """
    Computes the sum of Euclidean distances between connected components.
    """
    total_length = 0.0
    for edge in edge_list:
        a, b = edge
        dist = np.linalg.norm(component_positions[a] - component_positions[b])
        total_length += dist
    return total_length

def compute_overlap_penalty(component_positions, threshold=0.05):
    """
    Penalizes components that are closer than a given threshold.
    """
    penalty = 0.0
    num_components = len(component_positions)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            dist = np.linalg.norm(component_positions[i] - component_positions[j])
            if dist < threshold:
                penalty += 1.0
    return penalty

def compute_heatmap_spread_score(heatmap):
    """
    Penalizes highly concentrated areas; lower std = more uniform = better.
    """
    return -np.std(heatmap)
