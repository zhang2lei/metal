import torch
import torch.nn as nn

class RegressionNetwork(nn.Module):
    def __init__(self, num_bands, num_samples, d_k, d_v):
        super(RegressionNetwork, self).__init__()
        self.num_bands = num_bands
        self.num_samples = num_samples
        self.self_attention = nn.MultiheadAttention(embed_dim=num_samples + 1, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(num_samples + 1, d_k),
            nn.ReLU(),
            nn.Linear(d_k, d_v)
        )
        self.linear = nn.Linear(d_v, 4)  # For Hg, Cu, Pb, Cd concentrations

    def forward(self, R, lambda_):
        X = torch.cat([R, lambda_], dim=-1)
        X = X.transpose(0, 1)
        A, _ = self.self_attention(X, X, X)
        F = self.ffn(A)
        F = F.transpose(0, 1)
        concentrations = self.linear(F)
        return concentrations
