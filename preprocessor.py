# utils/preprocessor.py
import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


def preprocess_tensor(tensor, replace_nan=True, replace_inf=True, normalize=True, clamp=True):
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if replace_nan and has_nan:
        tensor = torch.nan_to_num(tensor, nan=0.0)

    if replace_inf and has_inf:
        tensor = torch.nan_to_num(tensor, posinf=5.0, neginf=-5.0)

    if normalize and tensor.dim() > 1:
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True) + 1e-8
        tensor = (tensor - mean) / std

    if clamp:
        tensor = F.leaky_relu(tensor, negative_slope=0.01)

    if torch.isnan(tensor).any():
        logger.warning("NaN still exists after preprocessing")
        tensor = torch.nan_to_num(tensor, nan=0.0)

    return tensor


def validate_edge_indices(edge_index, num_nodes):
    if edge_index.size(1) == 0:
        return edge_index
    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    valid_mask &= (edge_index[0] >= 0) & (edge_index[1] >= 0)

    return edge_index[:, valid_mask]


def compute_class_weights(labels):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()

    class_weights = torch.zeros(len(labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_weights[mask] = weights[i]

    return class_weights