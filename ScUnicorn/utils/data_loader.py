"""
Data Loader Utility for ScUnicorn
Provides functions to load and prepare Hi-C data for training and evaluation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
"""
Data Loader Utility for ScUnicorn
Provides functions to load and prepare Hi-C and multi-modal data for training and evaluation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


# -------------------------------
# Original Hi-C Dataset
# -------------------------------
class HiCDataset(Dataset):
    """
    Custom PyTorch Dataset for Hi-C data.
    """
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith('.npy')])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith('.npy')])
        assert len(self.lr_files) == len(self.hr_files), "Mismatch between LR and HR file counts."

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_data = np.load(self.lr_files[idx])
        hr_data = np.load(self.hr_files[idx])

        lr_tensor = torch.tensor(lr_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        hr_tensor = torch.tensor(hr_data, dtype=torch.float32).unsqueeze(0)
        return lr_tensor, hr_tensor


def get_dataloader(lr_dir, hr_dir, batch_size=8, shuffle=True):
    """
    Returns standard Hi-C DataLoader.
    """
    dataset = HiCDataset(lr_dir, hr_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# -------------------------------
# NEW: Multi-Modal Dataset
# -------------------------------
class MultiModalDataset(Dataset):
    """
    Dataset for integrating scHi-C with ATAC-seq, ChIP-seq, and RNA-seq data.
    Each omic file is expected to be an .npz with key 'data' or a directory of .npy files.
    """
    def __init__(self, hic_dir, atac_path=None, chip_path=None, rna_path=None):
        self.hic_files = sorted([os.path.join(hic_dir, f) for f in os.listdir(hic_dir) if f.endswith('.npy')])

        # Load or initialize auxiliary data
        self.atac = np.load(atac_path)['data'] if atac_path else None
        self.chip = np.load(chip_path)['data'] if chip_path else None
        self.rna = np.load(rna_path)['data'] if rna_path else None

    def __len__(self):
        return len(self.hic_files)

    def __getitem__(self, idx):
        hic_data = np.load(self.hic_files[idx])
        hic_tensor = torch.tensor(hic_data, dtype=torch.float32).unsqueeze(0)

        # Each auxiliary omic is optional
        atac_tensor = torch.tensor(self.atac[idx], dtype=torch.float32) if self.atac is not None else torch.zeros(1)
        chip_tensor = torch.tensor(self.chip[idx], dtype=torch.float32) if self.chip is not None else torch.zeros(1)
        rna_tensor = torch.tensor(self.rna[idx], dtype=torch.float32) if self.rna is not None else torch.zeros(1)

        return hic_tensor, atac_tensor, chip_tensor, rna_tensor


def get_multimodal_dataloader(hic_dir, atac_path=None, chip_path=None, rna_path=None, batch_size=8, shuffle=True):
    """
    Returns DataLoader for multi-modal training.
    """
    dataset = MultiModalDataset(hic_dir, atac_path, chip_path, rna_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# -------------------------------
# Test Functionality
# -------------------------------
if __name__ == "__main__":
    print("Testing Hi-C and Multi-Modal Data Loaders...")

    # Create dummy Hi-C data
    lr_dir = "data/lr_samples/"
    hr_dir = "data/hr_samples/"
    os.makedirs(lr_dir, exist_ok=True)
    os.makedirs(hr_dir, exist_ok=True)

    for i in range(2):
        np.save(os.path.join(lr_dir, f"lr_sample_{i+1}.npy"), np.random.rand(64, 64))
        np.save(os.path.join(hr_dir, f"hr_sample_{i+1}.npy"), np.random.rand(128, 128))

    dataloader = get_dataloader(lr_dir, hr_dir, batch_size=1)
    for lr, hr in dataloader:
        print("LR Shape:", lr.shape, "HR Shape:", hr.shape)

    # Multi-modal test (fake data)
    np.savez("data/fake_atac.npz", data=np.random.rand(2, 256))
    np.savez("data/fake_chip.npz", data=np.random.rand(2, 256))
    np.savez("data/fake_rna.npz", data=np.random.rand(2, 512))

    multimodal_loader = get_multimodal_dataloader(
        lr_dir,
        atac_path="data/fake_atac.npz",
        chip_path="data/fake_chip.npz",
        rna_path="data/fake_rna.npz",
        batch_size=1
    )

    for hic, atac, chip, rna in multimodal_loader:
        print("Hi-C:", hic.shape, "ATAC:", atac.shape, "ChIP:", chip.shape, "RNA:", rna.shape)

    print("Data loader test passed.")

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
