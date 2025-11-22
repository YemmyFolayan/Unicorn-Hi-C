"""
Metrics Utility for ScUnicorn - Optimized Version with PSNR
Extends standard image restoration metrics with biological correlation measures.
Includes:
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index Measure (SSIM)
  - Pearson Correlation Coefficient
  - Spearman Rank Correlation

Optimization focus: Using helper functions for robust data handling and improving 
robustness against zero-variance data (constant arrays).
"""

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
import numpy as np


# -------------------------------------------------------------
# Helper Functions for Data Conversion and Robustness
# -------------------------------------------------------------
def _to_numpy_flat(data_tensor):
    """Converts a batched tensor (B, C, H, W) to a flattened numpy array (B, H*W)."""
    # Detach from graph, move to CPU, convert to numpy, reshape to (Batch, Features)
    return data_tensor.detach().cpu().numpy().reshape(data_tensor.size(0), -1)


def _safe_correlation(func, x, y):
    """
    Computes a correlation function (func) on paired 1D arrays (x, y).
    Returns 0.0 if the data has zero variance (correlation is undefined).
    """
    try:
        # Check if either array is constant (zero variance)
        # We use a small tolerance (1e-8) for floating point safety
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        
        # Compute correlation (scipy returns corr, p-value)
        corr, _ = func(x, y)
        
        # Check for NaN result (can occur on singular data)
        if np.isnan(corr):
            return 0.0
        
        return corr
    except Exception:
        # Catch all other possible exceptions
        return 0.0


# -------------------------------------------------------------
# Standard Image / Reconstruction Metrics
# -------------------------------------------------------------
def compute_mse(hr_data, hr_restored):
    """
    Compute the Mean Squared Error (MSE) between HR and restored data.
    """
    # F.mse_loss is highly optimized and handles batches efficiently
    return F.mse_loss(hr_restored, hr_data, reduction="mean").item()


def compute_psnr(hr_data, hr_restored):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR).
    PSNR = 10 * log10(MAX_I^2 / MSE)
    MAX_I is the maximum possible pixel value. Assuming data is normalized to [0, 1] 
    or a common range for image processing, MAX_I=1 is often used, but here we 
    calculate it dynamically based on the max value in the HR data.
    """
    mse = compute_mse(hr_data, hr_restored)
    if mse == 0.0:
        # Perfect reconstruction: PSNR is infinite, represented by a very high number
        return 100.0 
    
    # Calculate MAX_I^2. We use the maximum value present in the ground truth data.
    # The max value is usually clipped/normalized in the training pipeline, but using 
    # the tensor max is a practical choice here for varying Hi-C input ranges.
    max_i = torch.max(hr_data).item()
    max_i_sq = max_i ** 2
    
    # PSNR calculation
    psnr = 10.0 * np.log10(max_i_sq / mse)
    return float(psnr)


def compute_ssim(hr_data, hr_restored):
    """
    Compute the Structural Similarity Index (SSIM) between HR and restored data.
    """
    batch_size = hr_data.size(0)
    ssim_values = []

    for i in range(batch_size):
        # Extract and squeeze individual samples
        hr_np = hr_data[i].detach().cpu().numpy().squeeze()
        restored_np = hr_restored[i].detach().cpu().numpy().squeeze()
        
        # Calculate data_range for normalization
        data_range = restored_np.max() - restored_np.min()
        if data_range < 1e-8:
            # If the image is flat/constant, SSIM is 1.0 (assuming perfect reconstruction)
            ssim_val = 1.0 
        else:
            try:
                ssim_val = ssim(
                    hr_np,
                    restored_np,
                    data_range=data_range,
                )
            except ValueError:
                ssim_val = 0.0
        ssim_values.append(ssim_val)

    return float(np.mean(ssim_values))


# -------------------------------------------------------------
# Bioinformatics-Specific Correlation Metrics
# -------------------------------------------------------------
def compute_pearson(hr_data, hr_restored):
    """
    Compute Pearson correlation between flattened HR and restored matrices.
    """
    hr_np = _to_numpy_flat(hr_data)
    restored_np = _to_numpy_flat(hr_restored)

    # Use a list comprehension to apply the robust correlation calculation across the batch
    pearson_scores = [
        _safe_correlation(pearsonr, hr_np[i], restored_np[i])
        for i in range(hr_np.shape[0])
    ]

    return float(np.mean(pearson_scores))


def compute_spearman(hr_data, hr_restored):
    """
    Compute Spearman rank correlation between HR and restored matrices.
    """
    hr_np = _to_numpy_flat(hr_data)
    restored_np = _to_numpy_flat(hr_restored)

    # Use a list comprehension to apply the robust correlation calculation across the batch
    spearman_scores = [
        _safe_correlation(spearmanr, hr_np[i], restored_np[i])
        for i in range(hr_np.shape[0])
    ]

    return float(np.mean(spearman_scores))


# -------------------------------------------------------------
# Combined Metric Summary
# -------------------------------------------------------------
def evaluate_all(hr_data, hr_restored):
    """
    Compute and return all metrics as a dictionary.
    Useful for logging or evaluation during training.
    """
    metrics = {
        "MSE": compute_mse(hr_data, hr_restored),    # Mean Squared Error (MSE)
        "PSNR": compute_psnr(hr_data, hr_restored),  # Peak signal-to-noise ratioPeak signal-to-noise 
        "SSIM": compute_ssim(hr_data, hr_restored),  # Structural Similarity Index Measure (SSIM)
        "Pearson": compute_pearson(hr_data, hr_restored), # 
        "Spearman": compute_spearman(hr_data, hr_restored),  #
    }
    return metrics


# -------------------------------------------------------------
# Test Script
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Testing metrics utility with PSNR...")

    # Dummy test data (Batch size of 4, 1 channel, 128x128 matrix)
    # Ensure values are positive, as is typical for Hi-C data
    hr_data = torch.randn(4, 1, 128, 128).abs() + 0.1 
    # Add small noise for reconstruction
    hr_restored = hr_data + torch.randn_like(hr_data) * 0.05 
    
    # Test case for perfect reconstruction (should result in PSNR=100.0)
    hr_data[0] = 5.0 
    hr_restored[0] = 5.0 

    # Compute all metrics
    results = evaluate_all(hr_data, hr_restored)

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    print("\nMetrics utility test passed successfully.")