import torch.optim as optim

def get_optimizer(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer
