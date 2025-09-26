# models/layers/constraints.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalConstraint(nn.Module):
    def __init__(self, alpha=0.5):
        super(OrthogonalConstraint, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=False)

    def forward(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        inner_product = torch.sum(x_norm * y_norm, dim=1, keepdim=True)

        x_orthogonal = x_norm - self.alpha * inner_product * y_norm
        y_orthogonal = y_norm - self.alpha * inner_product * x_norm

        x_orthogonal = F.normalize(x_orthogonal, p=2, dim=1)
        y_orthogonal = F.normalize(y_orthogonal, p=2, dim=1)

        return x_orthogonal, y_orthogonal