import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectionNetwork(nn.Module):
    def __init__(self, D):
        super(ConnectionNetwork, self).__init__()
        self.mlp_cw = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )
        self.mlp_ccw = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

    def forward(self, descriptors):
        num_points = descriptors.size(0)
        S_cw = torch.zeros(num_points, num_points)
        S_ccw = torch.zeros(num_points, num_points)

        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    m_ij = torch.cat([descriptors[i], descriptors[j]], dim=-1)
                    S_cw[i, j] = self.mlp_cw(m_ij)
                    S_ccw[i, j] = self.mlp_ccw(m_ij)

        S = S_cw + S_ccw.T
        P = self.sinkhorn_algorithm(S)
        return P

    def sinkhorn_algorithm(self, S, num_iters=100):
        P = torch.exp(S)
        for _ in range(num_iters):
            P /= P.sum(dim=1, keepdim=True)
            P /= P.sum(dim=0, keepdim=True)
        return P
