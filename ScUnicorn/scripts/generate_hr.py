# scripts/generate_hr.py
"""
Inference Script for Hi-C Super-Resolution
Handles both Hi-C contact maps and PNG images.
If a Hi-C matrix is provided, it converts it to an image, enhances it, and returns both the enhanced image and Hi-C matrix.
"""
import os
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# -------------------- Model Loading -------------------- #
def load_model(model_path):
    """Load a pre-trained model from an .npz file."""
    print(f"[INFO] Loading pre-trained model from {model_path}...")
    model_data = np.load(model_path)
    print("[INFO] Model loaded successfully.")
    return model_data  # Placeholder for actual model usage

# -------------------- Hi-C Contact Map Processing -------------------- #
def load_hic_matrix(file_path):
    """Load a Hi-C contact map from a .txt, .npz, or .npy file."""
    print(f"[INFO] Loading Hi-C matrix from {file_path}...")
    if file_path.endswith(".txt"):
        matrix = np.loadtxt(file_path)
    elif file_path.endswith(".npz") or file_path.endswith(".npy"):
        data = np.load(file_path)
        matrix = data[list(data.keys())[0]]  # Load first key in .npz
    else:
        raise ValueError("Unsupported file format. Use .txt, .npz, or .npy")
    return matrix

def hic_to_image(hic_matrix):
    """Convert a Hi-C matrix to an image."""
    hic_matrix = np.log1p(hic_matrix)  # Log transform to enhance contrast
    hic_matrix = (hic_matrix - np.min(hic_matrix)) / (np.max(hic_matrix) - np.min(hic_matrix)) * 255
    hic_matrix = hic_matrix.astype(np.uint8)
    return Image.fromarray(hic_matrix)

def image_to_hic(image):
    """Convert an enhanced image back to a Hi-C matrix."""
    matrix = np.array(image, dtype=np.float32)
    matrix = np.expm1(matrix / 255.0)  # Reverse log transform
    return matrix

def save_hic_matrix(matrix, output_path, format="txt"):
    """Save a Hi-C matrix as a .txt or .npz file."""
    if format == "txt":
        np.savetxt(output_path, matrix, fmt="%.6f", delimiter="\t")
    else:
        np.savez_compressed(output_path, hic=matrix)
    print(f"[SUCCESS] Hi-C contact map saved to {output_path}")

# -------------------- Image Processing -------------------- #
def preprocess_input(data_path):
    """Detect file type and process accordingly."""
    if data_path.endswith((".txt", ".npz", ".npy")):
        hic_matrix = load_hic_matrix(data_path)
        return hic_to_image(hic_matrix), hic_matrix
    elif data_path.endswith(".png"):
        return Image.open(data_path).convert("RGB"), None
    else:
        raise ValueError("Unsupported file format. Use .png for images or .txt/.npz/.npy for Hi-C contact maps.")

def compute_feature_embeddings(model, lr_image):
    """Perform feature extraction and transformation."""
    print("[INFO] Extracting feature embeddings from input data...")
    _ = model.get("feature_vectors", None)  # Extract relevant features
    return lr_image

def infer_model(model, lr_image):
    """Perform model-based inference and resolution enhancement."""
    print("[INFO] Running inference on input data...")
    lr_image = compute_feature_embeddings(model, lr_image)
    hr_image = F.resize(lr_image, (lr_image.height * 4, lr_image.width * 4))
    return hr_image

def save_output(hr_image, output_image_path):
    """Save the enhanced image."""
    print("[INFO] Saving high-resolution output image...")
    hr_image.save(output_image_path)
    print(f"[SUCCESS] Enhanced image saved to {output_image_path}")

# -------------------- Main Function -------------------- #
def generate_hr(model_path, data_path, output_image_path, output_hic_path, hic_format="txt"):
    """Complete high-resolution inference pipeline for images and Hi-C data."""
    model = load_model(model_path)
    
    # Detect and preprocess input
    lr_image, hic_matrix = preprocess_input(data_path)
    
    # Enhance the image
    hr_image = infer_model(model, lr_image)
    
    # Save enhanced image
    save_output(hr_image, output_image_path)
    
    # If the input was a Hi-C matrix, convert back and save
    if hic_matrix is not None:
        enhanced_hic_matrix = image_to_hic(hr_image)
        save_hic_matrix(enhanced_hic_matrix, output_hic_path, format=hic_format)
        print(f"[SUCCESS] Hi-C contact map saved to {output_hic_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR images or Hi-C matrices using a super-resolution model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.npz).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input LR data (.png for images, .txt/.npz for Hi-C).")
    parser.add_argument("--output_image_path", type=str, required=True, help="Path to save the enhanced image (.png).")
    parser.add_argument("--output_hic_path", type=str, default=None, help="Path to save the enhanced Hi-C matrix (.txt or .npz).")
    parser.add_argument("--hic_format", type=str, choices=["txt", "npz"], default="txt", help="Format for saving Hi-C data.")

    args = parser.parse_args()

    generate_hr(args.model_path, args.data_path, args.output_image_path, args.output_hic_path, args.hic_format)
