"""
Metrics Utility for ScUnicorn
Provides reusable functions to compute evaluation metrics such as MSE and SSIM.
"""
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_mse(hr_data, hr_restored):
    """
    Compute the Mean Squared Error (MSE) between high-resolution data and restored data.

    Parameters:
    - hr_data (torch.Tensor): Ground truth high-resolution data.
    - hr_restored (torch.Tensor): Restored high-resolution data from the model.

    Returns:
    - float: Mean Squared Error value.
    """
    mse = F.mse_loss(hr_restored, hr_data, reduction='mean').item()
    return mse

def compute_ssim(hr_data, hr_restored):
    """
    Compute the Structural Similarity Index Measure (SSIM) between high-resolution data and restored data.

    Parameters:
    - hr_data (torch.Tensor): Ground truth high-resolution data.
    - hr_restored (torch.Tensor): Restored high-resolution data from the model.

    Returns:
    - float: Average SSIM value across all samples in the batch.
    """
    batch_size = hr_data.size(0)
    ssim_values = []

    for i in range(batch_size):
        hr_np = hr_data[i].cpu().numpy().squeeze()
        restored_np = hr_restored[i].cpu().numpy().squeeze()
        ssim_value = ssim(hr_np, restored_np, data_range=restored_np.max() - restored_np.min())
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)

if __name__ == "__main__":
    # Test the metrics
    print("Testing metrics utility...")

    # Create dummy data for testing
    hr_data = torch.randn(4, 1, 128, 128)  # Batch of 4 HR samples
    hr_restored = hr_data + torch.randn_like(hr_data) * 0.1  # Add small noise to simulate restoration

    # Compute MSE and SSIM
    mse_value = compute_mse(hr_data, hr_restored)
    ssim_value = compute_ssim(hr_data, hr_restored)

    # Print results
    print(f"MSE: {mse_value:.6f}")
    print(f"SSIM: {ssim_value:.6f}")

    print("Metrics utility test passed.")
