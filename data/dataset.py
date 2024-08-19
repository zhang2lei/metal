import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class SpecMetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npz_path = os.path.join(self.data_dir, self.data_files[idx])
        data = np.load(npz_path)
        reflectance = data['reflectance']
        wavelength = data['wavelength']

        label_row = self.labels[self.labels['filename'] == self.data_files[idx]]
        concentrations = label_row[['Hg', 'Cu', 'Pb', 'Cd']].values.flatten().astype(np.float32)

        sample = {'reflectance': reflectance, 'wavelength': wavelength, 'concentrations': concentrations}

        if self.transform:
            sample = self.transform(sample)

        return sample


def default_collate(batch):
    reflectance = torch.tensor([item['reflectance'] for item in batch], dtype=torch.float32)
    wavelength = torch.tensor([item['wavelength'] for item in batch], dtype=torch.float32)
    concentrations = torch.tensor([item['concentrations'] for item in batch], dtype=torch.float32)

    return {'reflectance': reflectance, 'wavelength': wavelength, 'concentrations': concentrations}


class ToTensor:
    def __call__(self, sample):
        reflectance, wavelength, concentrations = sample['reflectance'], sample['wavelength'], sample['concentrations']

        return {
            'reflectance': torch.from_numpy(reflectance).float(),
            'wavelength': torch.from_numpy(wavelength).float(),
            'concentrations': torch.from_numpy(concentrations).float()
        }


class Normalize:
    def __call__(self, sample):
        reflectance, wavelength, concentrations = sample['reflectance'], sample['wavelength'], sample['concentrations']

        reflectance = (reflectance - np.mean(reflectance, axis=0)) / np.std(reflectance, axis=0)

        return {
            'reflectance': reflectance,
            'wavelength': wavelength,
            'concentrations': concentrations
        }


if __name__ == "__main__":
    # Example usage
    data_dir = 'path/to/data'
    transformed_dataset = SpecMetDataset(data_dir=data_dir, transform=transforms.Compose([Normalize(), ToTensor()]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['reflectance'].shape, sample['wavelength'].shape, sample['concentrations'])

        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, collate_fn=default_collate)

    for batch in dataloader:
        print(batch['reflectance'].shape, batch['wavelength'].shape, batch['concentrations'].shape)
        break
