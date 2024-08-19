import torch.nn as nn

def compute_loss(concentrations, positions, permutation_matrix, targets):
    criterion = nn.MSELoss()
    det_loss = criterion(concentrations, targets['concentrations'])
    match_loss = -torch.sum(targets['permutation_matrix'] * torch.log(permutation_matrix))
    corr_loss = 1 - torch.sum((concentrations - concentrations.mean(dim=0)) * (targets['concentrations'] - targets['concentrations'].mean(dim=0))) / (
        torch.sqrt(torch.sum((concentrations - concentrations.mean(dim=0)) ** 2)) * torch.sqrt(torch.sum((targets['concentrations'] - targets['concentrations'].mean(dim=0)) ** 2))
    )
    total_loss = det_loss + match_loss + corr_loss
    return total_loss
