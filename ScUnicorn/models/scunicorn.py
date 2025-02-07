# scripts/generate_hr.py
"""
Inference Script for Hi-C Super-Resolution
Generates high-resolution Hi-C maps from low-resolution inputs.
Uses a pre-trained ESRGAN model for super-resolution.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load pre-trained ESRGAN model
model_esrgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo', 'ESRGAN', pretrained=True)
model_esrgan.eval()

def load_lr_data(lr_file):
    """Load low-resolution data from a .png file."""
    if lr_file.endswith(".png"):
        lr_image = Image.open(lr_file).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        lr_tensor = transform(lr_image).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError("Unsupported file format. Use .png")
    return lr_tensor

def save_hr_data(hr_tensor, output_file):
    """Save high-resolution data to a .png file."""
    hr_image = hr_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    hr_image = ((hr_image + 1) / 2 * 255).astype(np.uint8)  # Rescale to 0-255
    hr_image = Image.fromarray(hr_image)
    hr_image.save(output_file)

def generate_hr(data_path, output_path):
    """Generate high-resolution data using ESRGAN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load LR data
    lr_data = load_lr_data(data_path).to(device)
    
    model = model_esrgan.to(device)
    model.eval()
    
    # Generate HR data
    with torch.no_grad():
        hr_data = model(lr_data)
    
    # Save output
    save_hr_data(hr_data, output_path)
    print(f"High-resolution data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR images using ESRGAN.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input LR image (.png).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated HR image.")
    
    args = parser.parse_args()
    generate_hr(args.data_path, args.output_path)
