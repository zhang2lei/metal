import torch
from torch.utils.data import DataLoader
from models.specmet import SpecMet
from data.dataset import SpecMetDataset
from train.loss import compute_loss
from train.optimizer import get_optimizer

def train_model(data_path, epochs, batch_size, lr):
    dataset = SpecMetDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SpecMet(num_bands=200, num_samples=1000, d_k=64, d_v=64, D=128)
    optimizer = get_optimizer(model, lr)
    model.train()

    for epoch in range(epochs):
        for i, (R, lambda_, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            concentrations, positions, permutation_matrix = model(R, lambda_)
            loss = compute_loss(concentrations, positions, permutation_matrix, targets)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'specmet_model.pth')

if __name__ == "__main__":
    train_model(data_path="path/to/data", epochs=100, batch_size=32, lr=1e-4)
