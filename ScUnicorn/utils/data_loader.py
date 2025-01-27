"""
Data Loader Utility for ScUnicorn
Provides functions to load and prepare Hi-C data for training and evaluation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HiCDataset(Dataset):
    """
    Custom PyTorch Dataset for Hi-C data.
    """
    def __init__(self, lr_files, hr_files):
        """
        Initialize the Hi-C Dataset.

        Parameters:
        - lr_files (list of str): List of file paths to low-resolution Hi-C data (.npy files).
        - hr_files (list of str): List of file paths to high-resolution Hi-C data (.npy files).
        """
        self.lr_data = [np.load(file) for file in lr_files]
        self.hr_data = [np.load(file) for file in hr_files]
        assert len(self.lr_data) == len(self.hr_data), "Mismatch between LR and HR file counts."

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.lr_data)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - tuple: (LR data, HR data) as PyTorch tensors.
        """
        lr_tensor = torch.tensor(self.lr_data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        hr_tensor = torch.tensor(self.hr_data[idx], dtype=torch.float32).unsqueeze(0)  # Add channel dim
        return lr_tensor, hr_tensor

def get_dataloader(lr_files, hr_files, batch_size=8, shuffle=True):
    """
    Create a DataLoader for the Hi-C Dataset.

    Parameters:
    - lr_files (list of str): List of file paths to low-resolution Hi-C data.
    - hr_files (list of str): List of file paths to high-resolution Hi-C data.
    - batch_size (int): Batch size for the DataLoader.
    - shuffle (bool): Whether to shuffle the data.

    Returns:
    - DataLoader: PyTorch DataLoader for the Hi-C dataset.
    """
    dataset = HiCDataset(lr_files, hr_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    # Test the data loader utility
    print("Testing data loader utility...")

    # Dummy file paths (replace with actual paths to .npy files)
    lr_dummy_files = ["data/lr_sample_1.npy", "data/lr_sample_2.npy"]
    hr_dummy_files = ["data/hr_sample_1.npy", "data/hr_sample_2.npy"]

    # Generate dummy .npy files for testing
    for path in lr_dummy_files:
        np.save(path, np.random.rand(64, 64))
    for path in hr_dummy_files:
        np.save(path, np.random.rand(128, 128))

    # Load data using the DataLoader
    dataloader = get_dataloader(lr_dummy_files, hr_dummy_files, batch_size=1, shuffle=False)

    # Test DataLoader output
    for lr, hr in dataloader:
        print("LR Shape:", lr.shape)
        print("HR Shape:", hr.shape)

    print("Data loader utility test passed.")
