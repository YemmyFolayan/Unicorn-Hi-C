"""
Visualization Utility for ScUnicorn
Enhanced plotting functions for visualizing and comparing Hi-C maps,
difference heatmaps, and correlation matrices for biological interpretability.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# -------------------------------------------------------------
# Core Heatmap Visualization
# -------------------------------------------------------------
def plot_hic_map(hic_data, title="Hi-C Map", cmap="hot", save_path=None, dpi=300):
    """
    Plot a single Hi-C contact map as a heatmap.

    Parameters:
    - hic_data (np.ndarray): Hi-C contact matrix.
    - title (str): Plot title.
    - cmap (str): Colormap for visualization.
    - save_path (str): If provided, saves the figure.
    - dpi (int): Resolution for saved plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(hic_data, cmap=cmap, aspect="auto")
    plt.colorbar(label="Contact Intensity")
    plt.title(title)
    plt.xlabel("Genomic Position")
    plt.ylabel("Genomic Position")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"[INFO] Saved Hi-C map plot to {save_path}")

    plt.show()
    plt.close()


# -------------------------------------------------------------
# Side-by-Side Comparison (Ground Truth vs Restored)
# -------------------------------------------------------------
def compare_hic_maps(hr_data, restored_data, lr_data=None, save_path=None, cmap="hot", dpi=300):
    """
    Compare LR, Restored, and HR Hi-C maps side-by-side.

    Parameters:
    - hr_data (np.ndarray): Ground truth high-resolution Hi-C data.
    - restored_data (np.ndarray): Model-restored Hi-C data.
    - lr_data (np.ndarray): Optional low-resolution input Hi-C data.
    - save_path (str): If provided, saves comparison plot.
    - cmap (str): Colormap for heatmaps.
    - dpi (int): Resolution for saved plot.
    """
    panels = 3 if lr_data is not None else 2
    plt.figure(figsize=(6 * panels, 6))

    # LR Hi-C Map
    if lr_data is not None:
        plt.subplot(1, panels, 1)
        plt.imshow(lr_data, cmap=cmap, aspect="auto")
        plt.title("Low-Resolution (LR)")
        plt.colorbar(label="Intensity")

    # Restored Hi-C Map
    plt.subplot(1, panels, panels - 1 if lr_data is None else 2)
    plt.imshow(restored_data, cmap=cmap, aspect="auto")
    plt.title("Restored Hi-C (ScUnicorn)")
    plt.colorbar(label="Intensity")

    # Ground Truth Hi-C Map
    plt.subplot(1, panels, panels)
    plt.imshow(hr_data, cmap=cmap, aspect="auto")
    plt.title("Ground Truth (HR)")
    plt.colorbar(label="Intensity")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"[INFO] Comparison plot saved to {save_path}")

    plt.show()
    plt.close()


# -------------------------------------------------------------
# Difference Map Visualization
# -------------------------------------------------------------
def plot_difference_map(hr_data, restored_data, title="Difference Map", save_path=None, dpi=300):
    """
    Plot the difference between HR and restored Hi-C matrices as a heatmap.

    Parameters:
    - hr_data (np.ndarray): Ground truth high-resolution Hi-C matrix.
    - restored_data (np.ndarray): Restored Hi-C matrix from model.
    - title (str): Plot title.
    - save_path (str): If provided, saves plot.
    - dpi (int): Resolution for saved plot.
    """
    diff = hr_data - restored_data
    plt.figure(figsize=(6, 6))
    sns.heatmap(diff, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel("Genomic Position")
    plt.ylabel("Genomic Position")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"[INFO] Saved difference map to {save_path}")

    plt.show()
    plt.close()


# -------------------------------------------------------------
# Correlation Heatmap (Pearson & Spearman)
# -------------------------------------------------------------
def plot_correlation_maps(hr_data, restored_data, save_path=None, dpi=300):
    """
    Plot correlation matrices for HR vs Restored data using Pearson and Spearman metrics.

    Parameters:
    - hr_data (np.ndarray): Ground truth high-resolution Hi-C matrix.
    - restored_data (np.ndarray): Restored high-resolution Hi-C matrix.
    - save_path (str): If provided, saves figure.
    - dpi (int): Resolution for saved figure.
    """
    # Flatten matrices to compute correlations across rows
    pearson_matrix = np.zeros_like(hr_data)
    spearman_matrix = np.zeros_like(hr_data)

    for i in range(hr_data.shape[0]):
        for j in range(hr_data.shape[1]):
            try:
                pearson_matrix[i, j], _ = pearsonr(hr_data[i], restored_data[j])
                spearman_matrix[i, j], _ = spearmanr(hr_data[i], restored_data[j])
            except Exception:
                pearson_matrix[i, j], spearman_matrix[i, j] = 0.0, 0.0

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(pearson_matrix, cmap="viridis")
    plt.title("Pearson Correlation Matrix")

    plt.subplot(1, 2, 2)
    sns.heatmap(spearman_matrix, cmap="mako")
    plt.title("Spearman Correlation Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"[INFO] Correlation heatmaps saved to {save_path}")

    plt.show()
    plt.close()


# -------------------------------------------------------------
# Integrated Visualization Wrapper
# -------------------------------------------------------------
def visualize_all(hr_data, restored_data, lr_data=None, prefix="output/visuals", index=0):
    """
    Generate and save all relevant visualizations for one Hi-C sample.

    Parameters:
    - hr_data (np.ndarray): Ground truth Hi-C.
    - restored_data (np.ndarray): Model prediction.
    - lr_data (np.ndarray): Optional LR input.
    - prefix (str): Output directory for saving plots.
    - index (int): Index of the sample for naming files.
    """
    import os
    os.makedirs(prefix, exist_ok=True)

    plot_hic_map(restored_data, title=f"Restored Hi-C #{index}", save_path=f"{prefix}/restored_{index}.png")
    compare_hic_maps(hr_data, restored_data, lr_data, save_path=f"{prefix}/compare_{index}.png")
    plot_difference_map(hr_data, restored_data, save_path=f"{prefix}/difference_{index}.png")
    plot_correlation_maps(hr_data, restored_data, save_path=f"{prefix}/correlation_{index}.png")


# -------------------------------------------------------------
# Test Script
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Testing visualization utilities with multi-modal metrics...")

    # Dummy matrices
    hr = np.random.rand(64, 64)
    lr = np.random.rand(32, 32)
    restored = hr + np.random.normal(0, 0.05, hr.shape)

    # Individual and combined visualizations
    plot_hic_map(hr, title="Ground Truth Hi-C")
    compare_hic_maps(hr, restored, lr_data=None)
    plot_difference_map(hr, restored)
    plot_correlation_maps(hr, restored)
    visualize_all(hr, restored, prefix="output/test_visuals", index=1)

    print("Visualization utilities test passed successfully.")
