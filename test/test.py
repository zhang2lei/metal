import torch
from torch.utils.data import DataLoader
from models.specmet import SpecMet
from data.dataset import SpecMetDataset

def test_model(data_path, model_path):
    dataset = SpecMetDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SpecMet(num_bands=200, num_samples=1000, d_k=64, d_v=64, D=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for i, (R, lambda_, targets) in enumerate(dataloader):
        with torch.no_grad():
            concentrations, positions, permutation_matrix = model(R, lambda_)
            print(f"Sample {i+1}: Concentrations: {concentrations}, Positions: {positions}, Permutation Matrix: {permutation_matrix}")

if __name__ == "__main__":
    test_model(data_path="path/to/test/data", model_path="specmet_model.pth")
