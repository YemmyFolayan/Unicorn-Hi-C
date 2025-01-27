# scripts/test_pipeline.py
"""
End-to-End Test Script for ScUnicorn
This script tests the entire pipeline, from data loading to inference, ensuring all components work together.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.scunicorn import ScUnicorn
from utils.data_loader import HiCDataset
from utils.metrics import compute_mse, compute_ssim
from utils.visualization import compare_hic_maps

# Hyperparameters
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_pipeline():
    """
    Test the entire ScUnicorn pipeline using synthetic data.
    """
    print("Testing ScUnicorn pipeline...")

    # Generate synthetic LR and HR data
    num_samples = 10
    lr_data = [np.random.rand(64, 64) for _ in range(num_samples)]  # LR data
    hr_data = [np.random.rand(128, 128) for _ in range(num_samples)]  # HR data

    # Create dataset and dataloader
    dataset = HiCDataset(lr_files=lr_data, hr_files=hr_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the ScUnicorn model
    model = ScUnicorn(kernel_size=3, scale_factor=2, input_dim=128*64, output_dim=512).to(DEVICE)
    model.eval()

    # Evaluate the model on the synthetic dataset
    total_mse = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for lr_batch, hr_batch in dataloader:
            lr_batch = lr_batch.to(DEVICE)
            hr_batch = hr_batch.to(DEVICE)

            # Forward pass
            hr_restored = model(lr_batch)

            # Compute metrics
            mse = compute_mse(hr_batch, hr_restored)
            ssim = compute_ssim(hr_batch, hr_restored)
            total_mse += mse
            total_ssim += ssim
            num_batches += 1

            # Visualize a sample from the batch
            hr_np = hr_batch[0].cpu().numpy().squeeze()
            restored_np = hr_restored[0].cpu().numpy().squeeze()
            compare_hic_maps(hr_np, restored_np)

    # Print average metrics
    avg_mse = total_mse / num_batches
    avg_ssim = total_ssim / num_batches
    print(f"Pipeline Test Results:\nAverage MSE: {avg_mse:.6f}\nAverage SSIM: {avg_ssim:.6f}")

if __name__ == "__main__":
    test_pipeline()
