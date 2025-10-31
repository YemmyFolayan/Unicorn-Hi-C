"""
Metrics Utility for ScUnicorn
Extends standard image restoration metrics with biological correlation measures.
Includes:
  - Mean Squared Error (MSE)
  - Structural Similarity Index Measure (SSIM)
  - Pearson Correlation Coefficient
  - Spearman Rank Correlation
"""

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr, spearmanr
import numpy as np


# -------------------------------------------------------------
# Standard Image / Reconstruction Metrics
# -------------------------------------------------------------
def compute_mse(hr_data, hr_restored):
    """
    Compute the Mean Squared Error (MSE) between HR and restored data.
    """
    mse = F.mse_loss(hr_restored, hr_data, reduction="mean").item()
    return mse


def compute_ssim(hr_data, hr_restored):
    """
    Compute the Structural Similarity Index (SSIM) between HR and restored data.
    """
    batch_size = hr_data.size(0)
    ssim_values = []

    for i in range(batch_size):
        hr_np = hr_data[i].detach().cpu().numpy().squeeze()
        restored_np = hr_restored[i].detach().cpu().numpy().squeeze()
        try:
            ssim_val = ssim(
                hr_np,
                restored_np,
                data_range=restored_np.max() - restored_np.min(),
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
    Reflects the linear relationship between genomic contact patterns.
    """
    hr_np = hr_data.detach().cpu().numpy().reshape(hr_data.size(0), -1)
    restored_np = hr_restored.detach().cpu().numpy().reshape(hr_restored.size(0), -1)

    pearson_scores = []
    for i in range(hr_np.shape[0]):
        try:
            corr, _ = pearsonr(hr_np[i], restored_np[i])
        except Exception:
            corr = 0.0
        pearson_scores.append(corr)

    return float(np.mean(pearson_scores))


def compute_spearman(hr_data, hr_restored):
    """
    Compute Spearman rank correlation between HR and restored matrices.
    Measures monotonic relationship, robust to non-linearities common in Hi-C data.
    """
    hr_np = hr_data.detach().cpu().numpy().reshape(hr_data.size(0), -1)
    restored_np = hr_restored.detach().cpu().numpy().reshape(hr_restored.size(0), -1)

    spearman_scores = []
    for i in range(hr_np.shape[0]):
        try:
            corr, _ = spearmanr(hr_np[i], restored_np[i])
        except Exception:
            corr = 0.0
        spearman_scores.append(corr)

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
        "MSE": compute_mse(hr_data, hr_restored),
        "SSIM": compute_ssim(hr_data, hr_restored),
        "Pearson": compute_pearson(hr_data, hr_restored),
        "Spearman": compute_spearman(hr_data, hr_restored),
    }
    return metrics


# -------------------------------------------------------------
# Test Script
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Testing metrics utility with multi-modal support...")

    # Dummy test data
    hr_data = torch.randn(4, 1, 128, 128)
    hr_restored = hr_data + torch.randn_like(hr_data) * 0.05

    # Compute all metrics
    results = evaluate_all(hr_data, hr_restored)

    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    print("Metrics utility test passed successfully.")
