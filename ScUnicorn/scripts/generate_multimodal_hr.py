#!/usr/bin/env python3
"""
scripts/generate_multimodal_hr_v2.py

Inference script for ScUnicorn with automatic dimension alignment
between Hi-C encoder outputs and fusion layers, incorporating ATAC-seq,
ChIP-seq, and RNA-seq features.
"""

# --- LIBRARY IMPORTS ---
import argparse  # Used to handle command-line arguments (like file paths)
import os        # Used for interacting with the operating system (like creating folders)
import sys       # Used for system-specific parameters and functions (though less used here)
import numpy as np  # Used for heavy-duty numerical and array operations (like matrices)
import pandas as pd # Used for reading and manipulating tabular data (like CSV/TSV files)
from PIL import Image  # Used for image processing, specifically for Hi-C matrix visualization
import torch     # The main PyTorch library for building neural networks
import torch.nn as nn  # Contains modules and classes for building network layers (like Conv2d, Linear)
import torchvision.transforms.functional as TF # Functions for transforming images into PyTorch tensors

# --- GLOBAL MODEL DIMENSIONS ---
# These constants define the required input size for the encoder networks.
ATAC_FEATURE_DIM = 5    # Expected number of features for ATAC-seq input
CHIP_FEATURE_DIM = 10   # Expected number of features for ChIP-seq input
RNA_FEATURE_DIM = 15    # Expected number of features for RNA-seq input

# Output dimensions for the feature encoders
ATAC_NET_OUTPUT_DIM = 256  # All ATAC features are compressed to this size
CHIP_NET_OUTPUT_DIM = 256  # All ChIP features are compressed to this size
RNA_NET_OUTPUT_DIM = 256  # All RNA features are compressed to this size

# Hi-C Encoder output size (fixed using AdaptiveMaxPool in ScUnicorn)
HIC_ENCODER_OUTPUT_DIM = 8192 # Hi-C image features are compressed to this large size

# Total dimension for Fusion Layer: HiC + ATAC + ChIP + RNA = 8192 + 256 + 256 + 256 = 8930
FUSION_INPUT_DIM = HIC_ENCODER_OUTPUT_DIM + ATAC_NET_OUTPUT_DIM + CHIP_NET_OUTPUT_DIM + RNA_NET_OUTPUT_DIM
OUTPUT_FEATURE_SIZE = 1024 # Intermediate feature size before the final high-resolution reconstruction


# --- MODEL DEFINITION ---

class ATACEncoder(nn.Module):
    """Encodes the ATAC-seq feature vector."""
    def __init__(self, in_dim, out_dim):
        super().__init__() # Initialize the base class for the neural network module
        self.fc = nn.Sequential( # Defines a simple sequence of layers (Fully Connected)
            nn.Linear(in_dim, in_dim * 2),  # First linear layer: multiplies the input dimension by 2
            nn.ReLU(),                      # ReLU activation: introduces non-linearity (output = max(0, input))
            nn.Linear(in_dim * 2, out_dim)  # Second linear layer: maps the expanded features to the fixed output size (256)
        )
    
    def forward(self, x):
        return self.fc(x) # Runs the input data (x) through the defined sequence of layers

class ChipEncoder(nn.Module):
    """Encodes the ChIP-seq feature vector."""
    def __init__(self, in_dim, out_dim):
        super().__init__() # Initialize the base class
        self.fc = nn.Sequential( # Defines the same simple Fully Connected sequence as ATACEncoder
            nn.Linear(in_dim, in_dim * 2),  # Expands input features
            nn.ReLU(),                      # Non-linearity
            nn.Linear(in_dim * 2, out_dim)  # Maps features to the fixed output size (256)
        )
    
    def forward(self, x):
        return self.fc(x) # Runs the input data (x) through the defined layers

class RNAEncoder(nn.Module):
    """Encodes the RNA-seq feature vector."""
    def __init__(self, in_dim, out_dim):
        super().__init__() # Initialize the base class
        self.fc = nn.Sequential( # Defines the same simple Fully Connected sequence as the others
            nn.Linear(in_dim, in_dim * 2),  # Expands input features
            nn.ReLU(),                      # Non-linearity
            nn.Linear(in_dim * 2, out_dim)  # Maps features to the fixed output size (256)
        )
    
    def forward(self, x):
        return self.fc(x) # Runs the input data (x) through the defined layers

class FusionNet(nn.Module):
    """Combines features and performs initial reconstruction."""
    def __init__(self, fusion_input_dim, output_size):
        super().__init__() # Initialize the base class
        self.fc_fuse = nn.Sequential(  # First sequence: Fuses the multimodal features
            nn.Linear(fusion_input_dim, fusion_input_dim // 2), # Reduces the large combined vector size by half
            nn.ReLU(),                                          # Non-linearity
            nn.Linear(fusion_input_dim // 2, output_size)       # Maps to the intermediate feature size (1024)
        )
        self.reconstruct = nn.Sequential( # Second sequence: Expands the feature vector to simulate HR output
            nn.Linear(output_size, output_size * 4),    # Expands features (1024 -> 4096)
            nn.ReLU(),                                  # Non-linearity
            nn.Linear(output_size * 4, output_size * 16), # Final expansion (4096 -> 16384), matching the desired HR matrix size
        )

    def forward(self, x):
        fused = self.fc_fuse(x) # Combine and compress the input features
        return self.reconstruct(fused) # Expand the compressed features into the final high-resolution representation

class ScUnicorn(nn.Module):
    """The main ScUnicorn model architecture, combining all components."""
    def __init__(self, atac_dim, chip_dim, rna_dim):
        super().__init__() # Initialize the base class
        # Hi-C Encoder: A Convolutional Neural Network (CNN) to process the 2D Hi-C image
        self.hic_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # First convolutional layer (1 input channel (grayscale), 32 output channels)
            nn.MaxPool2d(2),                            # Max pooling: halves the size, keeping the most important features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# Second convolutional layer
            nn.MaxPool2d(2),                            # Max pooling
            nn.Conv2d(64, 512, kernel_size=3, padding=1),# Third convolutional layer
            nn.AdaptiveMaxPool2d((4, 4)),               # Adaptive pooling: forces the output size of the map to 4x4, regardless of input size
            nn.Flatten()                                # Flattens the 4x4x512 feature map into a single 8192-number vector
        )
        
        # Multimodal feature encoders (using the simple classes defined above)
        self.atac_encoder = ATACEncoder(atac_dim, ATAC_NET_OUTPUT_DIM)
        self.chip_encoder = ChipEncoder(chip_dim, CHIP_NET_OUTPUT_DIM)
        self.rna_encoder = RNAEncoder(rna_dim, RNA_NET_OUTPUT_DIM)

        # Fusion network (the core combination and reconstruction part)
        self.fusion_net = FusionNet(FUSION_INPUT_DIM, OUTPUT_FEATURE_SIZE)


# -------------------------------------------------------------------------
# --- DATA UTILITY FUNCTIONS ---

def load_hic_matrix(path):
    """Loads Hi-C matrix from .txt, .npy, or .npz file."""
    if path.endswith(".txt"):  # Check if the file is a plain text matrix
        mat = np.loadtxt(path) # Load data using numpy's text loading function
    elif path.endswith(".npy"): # Check for NumPy's binary format
        mat = np.load(path) # Load data using numpy's binary loading function
    elif path.endswith(".npz"): # Check for NumPy's compressed format
        data = np.load(path)
        mat = data[list(data.keys())[0]] # Extract the first array from the compressed archive
    else:
        raise ValueError("Unsupported Hi-C file format. Use .txt, .npy, or .npz") # Stop if file type is unknown
    return mat.astype(np.float32) # Convert matrix data to 32-bit floats for the model


def hic_matrix_to_image(mat):
    """Converts Hi-C matrix to a normalized PIL Image, ready for the CNN."""
    mat = np.nan_to_num(mat) # Replace any 'Not a Number' values with 0 (robustness)
    mat = np.log1p(mat)      # Apply log(1 + x) transformation (compresses large values, common for Hi-C)
    mn, mx = mat.min(), mat.max() # Find the minimum and maximum values for scaling
    # Normalize to 0-255 range (standard for 8-bit image data)
    scaled = ((mat - mn) / (mx - mn + 1e-8) * 255.0).astype(np.uint8) # Scale, convert to integer
    return Image.fromarray(scaled), (mn, mx) # Return the image and the original min/max for later reversal


def image_to_hic_matrix(img, scale_params):
    """Restores the HR image back to a numeric Hi-C matrix by reversing transformations."""
    arr = np.array(img).astype(np.float32) # Convert the output image back to a numerical array
    mn, mx = scale_params # Get the original min/max used during the initial scaling
    # Denormalize from 0-255 range back to the log-transformed scale
    restored = arr / 255.0 * (mx - mn) + mn
    # Reverse the log1p transformation (undoes log(1 + x))
    return np.expm1(restored) # returns (e^x - 1)


def save_hic(matrix, out_path):
    """Saves the final high-resolution Hi-C matrix to a text file."""
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Save the matrix to a tab-separated text file, formatted to 6 decimal places
    np.savetxt(out_path, matrix, fmt="%.6f", delimiter="\t")
    print(f"[INFO] Saved Hi-C matrix to {out_path}") # Confirmation message


def _process_feature_vector(feature_array, target_dim, device):
    """Pads or tiles a feature vector to match the target dimension (e.g., 2 features -> 5 features)."""
    current_dim = len(feature_array) # Get the current length of the feature vector
    if current_dim > target_dim: # Check if the feature array is too long
        # Truncate if too long (unlikely for this data)
        processed_features = feature_array[:target_dim] # Cut off extra features
    else:
        # Pad/tile if too short (necessary for our defined dims)
        # Tile the features to fill the required dimension (repeats the array until it's long enough)
        processed_features = np.tile(feature_array, (target_dim + current_dim - 1) // current_dim)[:target_dim]

    # Convert the processed feature array to a PyTorch tensor, set data type, move to device (CPU/GPU), and add batch dimension (unsqueeze(0))
    return torch.tensor(processed_features.astype(np.float32), device=device).unsqueeze(0)


def load_atac_features(path, device):
    """
    Loads ATAC-seq features (UMAP coordinates) and processes them to the correct dimension.
    """
    try:
        # File is tab-separated, skip the first row (e.g., '#cell_id...')
        df = pd.read_csv(path, sep='\t', skiprows=[1], header=0)
        
        # Extract UMAP1 and UMAP2 from the first data row (iloc[0])
        umap1 = df['UMAP1'].iloc[0]
        umap2 = df['UMAP2'].iloc[0]
        feature_array = np.array([umap1, umap2]) # Create a numpy array with the 2 features
        
        print(f"[INFO] Extracted ATAC features (UMAP1, UMAP2): {umap1:.2f}, {umap2:.2f}")
        return _process_feature_vector(feature_array, ATAC_FEATURE_DIM, device) # Resize and return as a tensor
    
    except Exception as e:
        # If loading fails, print an error and return a tensor of zeros with the expected dimension
        print(f"[ERROR] Could not load or process ATAC file features: {e}. Returning zero tensor.")
        return torch.zeros(1, ATAC_FEATURE_DIM, dtype=torch.float32, device=device)

def load_chip_features(path, device):
    """
    Loads ChIP-seq features (histone modification/motif scores) and processes them.
    """
    try:
        df = pd.read_csv(path, sep='\t') # Load the tab-separated file
        
        # Define the names of the key feature columns to be extracted
        chip_cols = ['H3K27me3_encode', 'DNase_encode', 'H3K27ac_encode', 'H3K4me1_encode', 
                     'H3K4me3_encode', 'H3K9ac_encode', 'gata6_best_motif_score', 'sox17_best_motif_score']
        
        # Select key columns from the first genomic region (row 0)
        # Fill NaN (missing values) with 0 for robustness
        feature_series = df.loc[0, chip_cols].fillna(0)
        feature_array = feature_series.values.astype(np.float32) # Convert the selected values to a float array
        
        print(f"[INFO] Extracted {len(feature_array)} ChIP features (Histone marks/Motifs) from first region.")
        return _process_feature_vector(feature_array, CHIP_FEATURE_DIM, device) # Resize and return as a tensor
    
    except Exception as e:
        # If loading fails, print an error and return a zero tensor
        print(f"[ERROR] Could not load or process ChIP file features: {e}. Returning zero tensor.")
        return torch.zeros(1, CHIP_FEATURE_DIM, dtype=torch.float32, device=device)

def load_rna_features(path, device):
    """
    Loads RNA-seq features (average FPKM) and processes them.
    """
    try:
        df = pd.read_csv(path, sep='\t') # Load the tab-separated file
        
        # Define the FPKM columns for the control samples
        rna_cols = ['shCon.FPKM.1', 'shCon.FPKM.2', 'shCon.FPKM.3']
        
        # Calculate the average FPKM of the first gene (row 0) across control samples
        avg_fpkm = df.loc[0, rna_cols].mean()
        feature_array = np.array([avg_fpkm]) # Create an array from the single average value
        
        print(f"[INFO] Extracted RNA feature (Avg shCon FPKM for first gene): {avg_fpkm:.2f}")
        return _process_feature_vector(feature_array, RNA_FEATURE_DIM, device) # Resize and return as a tensor
    
    except Exception as e:
        # If loading fails, print an error and return a zero tensor
        print(f"[ERROR] Could not load or process RNA file features: {e}. Returning zero tensor.")
        return torch.zeros(1, RNA_FEATURE_DIM, dtype=torch.float32, device=device)


def build_model(device):
    """Initializes the ScUnicorn model and prepares it for inference."""
    # Create an instance of the main model class
    model = ScUnicorn(atac_dim=ATAC_FEATURE_DIM, chip_dim=CHIP_FEATURE_DIM, rna_dim=RNA_FEATURE_DIM)
    model.to(device)  # Move the model's parameters (weights) to the CPU or GPU
    model.eval()      # Set the model to evaluation mode (turns off training-specific features like dropout)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Loads model weights, handling potential 'module.' prefix."""
    # This is a placeholder for the actual weight loading mechanism.
    # In a real scenario, this would load a state_dict from a file.
    print(f"[INFO] Skipping actual checkpoint loading from {checkpoint_path}. Using initial weights.")


def run_inference(model, hic_image, atac_tensor, chip_tensor, rna_tensor, device):
    """Runs the multimodal inference using all inputs to generate a high-resolution map."""
    # 1. Prepare Hi-C Input
    # Convert the PIL image to a PyTorch tensor, grayscale ('L'), add batch dimension, and move to device
    hic_tensor = TF.to_tensor(hic_image.convert("L")).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculation (saving memory/speed, necessary for inference)
        # 2. Pass through Encoders: Extract feature vectors from all four inputs
        hic_feat_flat = model.hic_encoder(hic_tensor)
        atac_feat_flat = model.atac_encoder(atac_tensor)
        chip_feat_flat = model.chip_encoder(chip_tensor)
        rna_feat_flat = model.rna_encoder(rna_tensor)

        # 3. Concatenate and Fusion: Stitch all feature vectors together
        # The dim=1 argument means concatenating along the feature dimension
        fused_input = torch.cat([hic_feat_flat, atac_feat_flat, chip_feat_flat, rna_feat_flat], dim=1)
        
        # Dynamic dimension check against the FusionNet's input layer (IMPORTANT for maintenance)
        target_dim = model.fusion_net.fc_fuse[0].in_features # Expected input size from FusionNet's first layer
        current_dim = fused_input.shape[1] # Actual size of the stitched vector

        if current_dim != target_dim:
            # If the calculated size (8930) doesn't match the model's expected size, raise a critical error
            print(f"[ERROR] Fusion input dimension mismatch! Expected {target_dim}, but got {current_dim}")
            raise RuntimeError("Fusion dimension mismatch.")

        # Run the combined vector through the Fusion and Reconstruction layers
        fused_out = model.fusion_net.fc_fuse(fused_input) # Intermediate feature vector (1024)
        out_tensor = model.fusion_net.reconstruct(fused_out) # Final expanded feature vector (HR output)

        # 4. Process Output: Reshape the 1D output vector back into a 2D matrix
        out_tensor = out_tensor.view(-1) # Flatten the tensor into a single dimension
        n = int(np.sqrt(out_tensor.numel())) # Calculate the side length (n) of the square matrix (n x n)
        if n * n != out_tensor.numel():
            # If the size isn't a perfect square (e.g., 16384), truncate it to the largest square size
            print(f"[WARN] Output size {out_tensor.numel()} is not a perfect square. Truncating to {n*n}.")
            out_tensor = out_tensor[: n * n]

        out_tensor = out_tensor.view(1, 1, n, n) # Reshape back to a 2D matrix (1 batch, 1 channel, n rows, n cols)
        arr = out_tensor.cpu().squeeze().numpy() # Move to CPU, remove extra dimensions (1, 1, n, n -> n, n), and convert to numpy array

    # Normalize output image (Scale the output data to 0-255 range for image saving)
    arr = np.nan_to_num(arr) # Handle NaNs again
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    # Return the processed output as an image and the raw normalized array
    return Image.fromarray(arr.astype(np.uint8)), arr


def main(args):
    """Main function to parse arguments and execute the entire inference pipeline."""
    # Determine the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Hi-C Data
    hic_mat = load_hic_matrix(args.data_path) # Load the low-resolution Hi-C numerical matrix
    hic_image, scale_params = hic_matrix_to_image(hic_mat) # Convert matrix to normalized image for the model
    
    # 2. Load Omics Features (using the helper functions)
    atac_tensor = load_atac_features(args.atac_data_path, device)
    chip_tensor = load_chip_features(args.chip_data_path, device)
    rna_tensor = load_rna_features(args.rna_data_path, device)
    
    # 3. Build and Load Model
    model = build_model(device) # Initialize the ScUnicorn model
    load_checkpoint(model, args.model_path, device) # Load trained weights (currently a placeholder)

    # 4. Run Multimodal Inference
    hr_image, numeric_output = run_inference(model, hic_image, atac_tensor, chip_tensor, rna_tensor, device) # Run the prediction

    # 5. Save Results
    os.makedirs(os.path.dirname(args.output_image_path) or ".", exist_ok=True) # Ensure output directory exists
    hr_image.save(args.output_image_path) # Save the enhanced image output
    print(f"[INFO] Enhanced image saved to {args.output_image_path}")

    # Restore the output image back to a Hi-C matrix (reversing log and scaling)
    restored_hic = image_to_hic_matrix(hr_image, scale_params)
    save_hic(restored_hic, args.output_hic_path) # Save the final numerical HR matrix


if __name__ == "__main__":
    # This block executes when the script is run directly
    # Setup argument parser to read file paths from the command line
    parser = argparse.ArgumentParser(description="Generate HR Hi-C map using ScUnicorn (Multimodal).")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pytorch)")
    parser.add_argument("--data_path", required=True, help="Path to Hi-C input (.txt/.npy/.npz)")
    parser.add_argument("--atac_data_path", required=True, help="Path to ATAC-seq features/metadata (.txt)")
    parser.add_argument("--chip_data_path", required=True, help="Path to ChIP-seq features/metadata (.txt)")
    parser.add_argument("--rna_data_path", required=True, help="Path to RNA-seq features/metadata (.txt)")
    parser.add_argument("--output_image_path", required=True, help="Path to save enhanced image (.png)")
    parser.add_argument("--output_hic_path", required=True, help="Path to save enhanced Hi-C matrix (.txt)")
    args = parser.parse_args() # Parse the arguments provided by the user
    main(args) # Call the main execution function