# scripts/generate_hr.py
"""
Inference Script for ScUnicorn
Generates high-resolution Hi-C maps from low-resolution inputs.
Supports models stored in .npz format.
"""
import os
import argparse
import torch
import numpy as np
from models.scunicorn import ScUnicorn

def load_lr_data(lr_file):
    """Load low-resolution Hi-C data from a file."""
    lr_data = np.load(lr_file)
    lr_tensor = torch.tensor(lr_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    return lr_tensor

def save_hr_data(hr_tensor, output_file):
    """Save high-resolution Hi-C data to a file."""
    hr_data = hr_tensor.squeeze().cpu().numpy()  # Remove batch and channel dims
    np.save(output_file, hr_data)

def load_npz_model(model, npz_path):
    """Load model weights from an .npz file."""
    data = np.load(npz_path)
    state_dict = {key: torch.tensor(data[key]) for key in data.files}
    model.load_state_dict(state_dict)
    print("Model weights loaded from .npz file.")

def generate_hr(model_path, data_path, output_path):
    """Generate high-resolution Hi-C data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LR data
    lr_data = load_lr_data(data_path).to(device)
    
    # Load trained model
    model = ScUnicorn().to(device)
    if model_path.endswith(".npz"):
        load_npz_model(model, model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # Generate HR data
    with torch.no_grad():
        hr_data = model(lr_data)
    
    # Save output
    save_hr_data(hr_data, output_path)
    print(f"High-resolution Hi-C data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR Hi-C data using ScUnicorn.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth or .npz).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input LR Hi-C data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated HR data.")
    
    args = parser.parse_args()
    generate_hr(args.model_path, args.data_path, args.output_path)
