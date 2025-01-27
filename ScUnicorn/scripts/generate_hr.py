"""
Inference Script for ScUnicorn
This script generates high-resolution Hi-C data from low-resolution inputs using a trained ScUnicorn model.
"""
import torch
import numpy as np
from models.scunicorn import ScUnicorn

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lr_data(lr_file):
    """
    Load low-resolution Hi-C data from a file.

    Parameters:
    - lr_file (str): Path to the LR Hi-C data file (e.g., .npy format).

    Returns:
    - torch.Tensor: LR Hi-C data tensor.
    """
    lr_data = np.load(lr_file)
    lr_tensor = torch.tensor(lr_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    return lr_tensor

def save_hr_data(hr_tensor, output_file):
    """
    Save high-resolution Hi-C data to a file.

    Parameters:
    - hr_tensor (torch.Tensor): HR Hi-C data tensor.
    - output_file (str): Path to save the HR data (e.g., .npy format).
    """
    hr_data = hr_tensor.squeeze().cpu().numpy()  # Remove batch and channel dims
    np.save(output_file, hr_data)

def generate_hr(lr_file, checkpoint_path, output_file):
    """
    Generate high-resolution Hi-C data from low-resolution input.

    Parameters:
    - lr_file (str): Path to the LR Hi-C data file.
    - checkpoint_path (str): Path to the trained model checkpoint.
    - output_file (str): Path to save the generated HR Hi-C data.
    """
    # Load LR data
    lr_data = load_lr_data(lr_file).to(DEVICE)

    # Load the trained model
    model = ScUnicorn(kernel_size=3, scale_factor=2, input_dim=128*64, output_dim=512).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Generate HR data
    with torch.no_grad():
        hr_data = model(lr_data)

    # Save HR data
    save_hr_data(hr_data, output_file)
    print(f"High-resolution Hi-C data saved to {output_file}")

if __name__ == "__main__":
    print("Generating high-resolution Hi-C data...")
    lr_input_file = "data/test_lr.npy"  # Path to input LR Hi-C data
    model_checkpoint = "checkpoints/scunicorn_epoch_10.pth"  # Trained model checkpoint
    hr_output_file = "data/generated_hr.npy"  # Path to save the generated HR data

    generate_hr(lr_input_file, model_checkpoint, hr_output_file)
    print("High-resolution generation completed.")
