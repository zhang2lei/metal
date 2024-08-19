import torch
import torch.nn as nn


class AttentionGNN(nn.Module):
    def __init__(self, D):
        super(AttentionGNN, self).__init__()
        self.mlp_enc = nn.Sequential(
            nn.Linear(D + 2, D),
            nn.ReLU()
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=D, num_heads=8)
        self.mlp_match = nn.Linear(D, D)
        self.mlp_offset = nn.Linear(D, 2)

    def forward(self, positions, descriptors):
        d_prime = self.mlp_enc(torch.cat([descriptors, positions], dim=-1))
        Q = d_prime.transpose(0, 1)
        K = d_prime.transpose(0, 1)
        V = d_prime.transpose(0, 1)
        A, _ = self.self_attention(Q, K, V)
        m = self.mlp_match(A.transpose(0, 1))
        t = self.mlp_offset(A.transpose(0, 1))
        return m, t
