"""
Visualization Utility for ScUnicorn
Provides functions to visualize and compare Hi-C maps (HR vs. LR).
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_hic_map(hic_data, title="Hi-C Map", save_path=None):
    """
    Plot a Hi-C map as a heatmap.

    Parameters:
    - hic_data (numpy.ndarray): Hi-C data to plot (2D matrix).
    - title (str): Title of the plot.
    - save_path (str): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(hic_data, cmap="hot", aspect="auto")
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.xlabel("Genomic Position")
    plt.ylabel("Genomic Position")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

def compare_hic_maps(hr_data, restored_data, save_path=None):
    """
    Compare HR Hi-C data with restored Hi-C data side-by-side.

    Parameters:
    - hr_data (numpy.ndarray): Ground truth high-resolution Hi-C data (2D matrix).
    - restored_data (numpy.ndarray): Restored high-resolution Hi-C data (2D matrix).
    - save_path (str): If provided, saves the comparison plot to this path.
    """
    plt.figure(figsize=(12, 6))

    # Ground Truth
    plt.subplot(1, 2, 1)
    plt.imshow(hr_data, cmap="hot", aspect="auto")
    plt.colorbar(label="Intensity")
    plt.title("Ground Truth (HR)")
    plt.xlabel("Genomic Position")
    plt.ylabel("Genomic Position")

    # Restored Output
    plt.subplot(1, 2, 2)
    plt.imshow(restored_data, cmap="hot", aspect="auto")
    plt.colorbar(label="Intensity")
    plt.title("Restored Output")
    plt.xlabel("Genomic Position")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    # Test the visualization utilities
    print("Testing visualization utilities...")

    # Create dummy Hi-C data
    hr_data = np.random.rand(128, 128)
    restored_data = hr_data + np.random.normal(0, 0.1, hr_data.shape)  # Add noise for restored simulation

    # Plot individual Hi-C map
    plot_hic_map(hr_data, title="High-Resolution Hi-C Map")

    # Compare Hi-C maps
    compare_hic_maps(hr_data, restored_data)
