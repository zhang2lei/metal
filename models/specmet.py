import torch
import torch.nn as nn
from .regression_network import RegressionNetwork
from .attention_gnn import AttentionGNN
from .connection_network import ConnectionNetwork


class SpecMet(nn.Module):
    def __init__(self, num_bands, num_samples, d_k, d_v, D):
        super(SpecMet, self).__init__()
        self.regression_network = RegressionNetwork(num_bands, num_samples, d_k, d_v)
        self.attention_gnn = AttentionGNN(D)
        self.connection_network = ConnectionNetwork(D)

    def forward(self, R, lambda_):
        concentrations = self.regression_network(R, lambda_)
        positions, descriptors = self.attention_gnn(concentrations, R)
        permutation_matrix = self.connection_network(descriptors)
        return concentrations, positions, permutation_matrix
