"""
Evaluation Script for the ScUnicorn Model
This script evaluates the ScUnicorn model on synthetic test data.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.scunicorn import ScUnicorn
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Hyperparameters
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data(num_samples=200, hr_size=(128, 128)):
    """
    Generate synthetic high-resolution test data.

    Parameters:
    - num_samples (int): Number of test samples to generate.
    - hr_size (tuple): Dimensions of the high-resolution data.

    Returns:
    - TensorDataset: Synthetic test dataset.
    """
    hr_data = torch.randn(num_samples, 1, *hr_size)  # HR Hi-C data
    return TensorDataset(hr_data)

def evaluate_scunicorn(checkpoint_path):
    """
    Evaluate the ScUnicorn model.

    Parameters:
    - checkpoint_path (str): Path to the model checkpoint.
    """
    # Initialize the ScUnicorn model
    model = ScUnicorn(kernel_size=3, scale_factor=2, input_dim=128*64, output_dim=512).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Generate synthetic test data
    dataset = generate_test_data()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define evaluation metrics
    mse_criterion = nn.MSELoss()
    total_mse = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for hr_data in dataloader:
            hr_data = hr_data[0].to(DEVICE)

            # Forward pass
            hr_restored = model(hr_data)

            # Compute MSE
            mse = mse_criterion(hr_restored, hr_data).item()
            total_mse += mse

            # Compute SSIM
            for i in range(hr_data.size(0)):
                hr_np = hr_data[i].cpu().numpy().squeeze()
                restored_np = hr_restored[i].cpu().numpy().squeeze()
                ssim_value = ssim(hr_np, restored_np, data_range=restored_np.max() - restored_np.min())
                total_ssim += ssim_value

            num_batches += hr_data.size(0)

    # Calculate average metrics
    avg_mse = total_mse / num_batches
    avg_ssim = total_ssim / num_batches

    print(f"Evaluation Results:\nAverage MSE: {avg_mse:.6f}\nAverage SSIM: {avg_ssim:.6f}")

if __name__ == "__main__":
    print("Evaluating ScUnicorn...")
    checkpoint = "checkpoints/scunicorn_epoch_10.pth"  # Example checkpoint path
    evaluate_scunicorn(checkpoint)
    print("Evaluation completed.")
