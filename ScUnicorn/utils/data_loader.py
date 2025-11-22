"""
Data Loader Utility for ScUnicorn
Provides functions to load and prepare Hi-C data for training and evaluation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class HiCDataset(Dataset):
    """
    Custom PyTorch Dataset for Hi-C data.
    """
    def __init__(self, lr_dir, hr_dir):
        """
        Initialize the Hi-C Dataset.

        Parameters:
        - lr_dir (str): Directory containing low-resolution Hi-C data (.npy files).
        - hr_dir (str): Directory containing high-resolution Hi-C data (.npy files).
        """
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.npy')])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.npy')])

        assert len(self.lr_files) == len(self.hr_files), "Mismatch between LR and HR file counts."

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.lr_files)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (LR data, HR data) as PyTorch tensors.
        """
        lr_data = np.load(self.lr_files[idx])
        hr_data = np.load(self.hr_files[idx])

        # Convert to tensors
        lr_tensor = torch.tensor(lr_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        hr_tensor = torch.tensor(hr_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        return lr_tensor, hr_tensor

def get_dataloader(lr_dir, hr_dir, batch_size=8, shuffle=True):
    """
    Create a DataLoader for the Hi-C Dataset.

    Parameters:
    - lr_dir (str): Directory containing low-resolution Hi-C data (.npy files).
    - hr_dir (str): Directory containing high-resolution Hi-C data (.npy files).
    - batch_size (int): Batch size for the DataLoader.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    - DataLoader: PyTorch DataLoader for the Hi-C dataset.
    """
    dataset = HiCDataset(lr_dir, hr_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    # Test the data loader utility
    print("Testing data loader utility...")

    # Dummy directories (replace with actual paths to .npy files)
    lr_dir = "data/lr_samples/"
    hr_dir = "data/hr_samples/"

    # Ensure test directories exist
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)

    # Generate dummy .npy files for testing
    for i in range(2):
        np.save(os.path.join(lr_dir, f"lr_sample_{i+1}.npy"), np.random.rand(64, 64))
        np.save(os.path.join(hr_dir, f"hr_sample_{i+1}.npy"), np.random.rand(128, 128))

    # Load data using the DataLoader
    dataloader = get_dataloader(lr_dir, hr_dir, batch_size=1, shuffle=False)

    # Test DataLoader output
    for lr, hr in dataloader:
        print("LR Shape:", lr.shape)
        print("HR Shape:", hr.shape)

    print("Data loader utility test passed.")