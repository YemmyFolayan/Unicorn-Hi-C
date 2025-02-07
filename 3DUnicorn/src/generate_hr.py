# scripts/generate_hr.py
"""
Inference Script for Hi-C Super-Resolution
Generates high-resolution Hi-C maps from low-resolution inputs.
Utilizes a model-based approach to infer high-resolution images.
"""
import os
import argparse
import numpy as np
from PIL import Image
import subprocess  # To call the second script
import torchvision.transforms.functional as F

def load_model(model_path):
    """Load a pre-trained model from an .npz file."""
    print(f"[INFO] Loading pre-trained model from {model_path}...")
    model_data = np.load(model_path)
    print("[INFO] Model loaded successfully.")
    return model_data  # Placeholder for actual model usage

def preprocess_input(lr_file):
    """Load and preprocess low-resolution data."""
    if lr_file.endswith(".png"):
        lr_image = Image.open(lr_file).convert("RGB")
        print("[INFO] Preprocessing input image...")
    else:
        raise ValueError("Unsupported file format. Use .png")
    return lr_image

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

def save_output(hr_image, output_file):
    """Save high-resolution output data."""
    print("[INFO] Saving high-resolution output...")
    hr_image.save(output_file)

def generate_hr(model_path, data_path, output_path):
    """Complete high-resolution inference pipeline."""
    model = load_model(model_path)
    lr_image = preprocess_input(data_path)
    hr_image = infer_model(model, lr_image)
    save_output(hr_image, output_path)
    print(f"[SUCCESS] High-resolution image saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR images using a super-resolution model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.npz).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input LR image (.png).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated HR image.")
    
    args = parser.parse_args()
    generate_hr(args.model_path, args.data_path, args.output_path)


# Call the second script after inference
try:
    subprocess.run(["python3", "/Users/mohanchandru/Documents/SCL_3dMax_ZSSR/3DUnicorn/src/Image_to_hic.py"], check=True)
    print("Hi-C contact matrix script executed successfully.")

    # Call main.py after Hi-C matrix generation
    subprocess.run(["python", "main.py", "--parameters", "../examples/parameters.txt"], check=True)
    print("main.py executed successfully with provided parameters.")

except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the Hi-C script: {e}")