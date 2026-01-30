import torch
import torch.nn as nn
from .attention import StructuralAttention


class ContentDecoupler(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(ContentDecoupler, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.extractor(x)


class StructureDecoupler(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(StructureDecoupler, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.structure_conv = StructuralAttention(output_dim, output_dim, dropout)
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index=None):
        features = self.feature_extractor(x)
        if edge_index is not None and edge_index.size(1) > 0:
            max_node_idx = edge_index.max().item() if edge_index.size(1) > 0 else 0
            if max_node_idx < features.size(0):
                features = self.structure_conv(features, edge_index)
        return self.final_proj(features)
