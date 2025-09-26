# models/layers/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StructuralAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(StructuralAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.qk = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.output_proj = nn.Linear(out_dim, out_dim)
        self.layernorm = nn.LayerNorm(out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x, edge_index):
        residual = x

        qk = self.qk(x)
        q = qk
        k = qk
        v = self.value(x)

        if edge_index.size(1) > 0:
            src, dst = edge_index
            valid_edges = (src < x.size(0)) & (dst < x.size(0))
            src = src[valid_edges]
            dst = dst[valid_edges]

            if len(src) > 0:
                q_dst = q[dst]
                k_src = k[src]
                scores = torch.sum(q_dst * k_src, dim=-1) / math.sqrt(self.out_dim)
                scores = torch.clamp(scores, -5.0, 5.0)

                dst_nodes, dst_indices = torch.unique(dst, return_inverse=True)
                attn_weights = torch.zeros_like(scores)

                for i in range(len(dst_nodes)):
                    mask = (dst_indices == i)
                    if mask.any():
                        node_scores = scores[mask]
                        node_attn = F.softmax(node_scores, dim=0)
                        attn_weights[mask] = node_attn

                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

                out = torch.zeros_like(x)
                for i in range(len(src)):
                    out[dst[i]] += attn_weights[i] * v[src[i]]
            else:
                out = v
        else:
            out = v

        out = self.output_proj(out)
        out = self.layernorm(out + residual)
        return out


class TemporalAttention(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2):
        super(TemporalAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pooled = x.mean(dim=1)
        pooled = self.layernorm(pooled)
        pooled = self.dropout(pooled)
        return pooled