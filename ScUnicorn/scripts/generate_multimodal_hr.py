#!/usr/bin/env python3
"""
scripts/generate_multimodal_hr_v2.py

Inference script for ScUnicorn with automatic dimension alignment
between Hi-C encoder outputs and fusion layers, incorporating ATAC-seq features.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# --- MOCK MODEL DEFINITION (REPLACE WITH YOUR ACTUAL ScUnicorn CLASS) ---
# This mock is necessary to define the expected dimensions and structure
# for the multimodal inference logic to work.

ATAC_FEATURE_DIM = 5 # Dimension of the ATAC-seq feature vector for the cell/region.

class MockATACNet(nn.Module):
    """Mocks the ATAC-seq feature network."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Simple linear transformation of the input ATAC features
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.fc(x)

class MockFusionNet(nn.Module):
    """Mocks the Fusion and Reconstruction network."""
    def __init__(self, fusion_input_dim, output_size):
        super().__init__()
        # fc_fuse must be a Sequential layer for compatibility with the original script
        self.fc_fuse = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 2, output_size)
        )
        self.reconstruct = nn.Sequential(
            nn.Linear(output_size, output_size * 4),
            nn.ReLU(),
            nn.Linear(output_size * 4, output_size * 16), # Simulate upsampling/HR output
        )

    def forward(self, x):
        fused = self.fc_fuse(x)
        return self.reconstruct(fused)

class ScUnicorn(nn.Module):
    """Mocks the ScUnicorn model structure. FIX: Replaced MaxPool with AdaptiveMaxPool
    to ensure consistent output size regardless of input Hi-C matrix dimensions."""
    def __init__(self, atac_dim, chip_dim, rna_dim):
        super().__init__()
        # Mock Hi-C Encoder - will produce a consistent feature vector of size 512*4*4=8192
        self.hic_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            # Use AdaptiveMaxPool2d to ensure the output spatial dimensions are always 4x4
            nn.AdaptiveMaxPool2d((4, 4)), 
            nn.Flatten()
        )
        # Feature size is now consistently 512 * 4 * 4 = 8192
        HIC_ENCODER_OUTPUT_DIM = 8192 

        # ATAC Network is only used if ATAC dimension is specified
        ATAC_NET_OUTPUT_DIM = 256
        self.atac_net = MockATACNet(atac_dim, ATAC_NET_OUTPUT_DIM)

        # Total dimension for fusion network: HiC_Output + ATAC_Output
        FUSION_INPUT_DIM = HIC_ENCODER_OUTPUT_DIM + ATAC_NET_OUTPUT_DIM
        OUTPUT_FEATURE_SIZE = 1024 # Arbitrary size for the intermediate feature before reconstruction
        
        self.fusion_net = MockFusionNet(FUSION_INPUT_DIM, OUTPUT_FEATURE_SIZE)


# -------------------------------------------------------------------------


def load_hic_matrix(path):
    """Loads Hi-C matrix from .txt, .npy, or .npz file."""
    if path.endswith(".txt"):
        mat = np.loadtxt(path)
    elif path.endswith(".npy"):
        mat = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        mat = data[list(data.keys())[0]]
    else:
        raise ValueError("Unsupported Hi-C file format. Use .txt, .npy, or .npz")
    return mat.astype(np.float32)


def hic_matrix_to_image(mat):
    """Converts Hi-C matrix to a normalized PIL Image."""
    mat = np.nan_to_num(mat)
    mat = np.log1p(mat)
    mn, mx = mat.min(), mat.max()
    # Normalize to 0-255 range
    scaled = ((mat - mn) / (mx - mn + 1e-8) * 255.0).astype(np.uint8)
    return Image.fromarray(scaled), (mn, mx)


def image_to_hic_matrix(img, scale_params):
    """Restores the HR image back to a numeric Hi-C matrix."""
    arr = np.array(img).astype(np.float32)
    mn, mx = scale_params
    # Denormalize from 0-255 range
    restored = arr / 255.0 * (mx - mn) + mn
    # Reverse the log1p transformation
    return np.expm1(restored)


def save_hic(matrix, out_path):
    """Saves the final Hi-C matrix to a text file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savetxt(out_path, matrix, fmt="%.6f", delimiter="\t")
    print(f"[INFO] Saved Hi-C matrix to {out_path}")


def load_atac_features(path, device):
    """
    Loads ATAC-seq feature data (currently single-cell metadata) 
    and returns a placeholder feature vector.
    
    NOTE: In a real application, you would implement complex logic here
    to extract or aggregate features (e.g., TF motif enrichment, 
    average accessibility across a region, or a learned embedding) 
    corresponding to the cell type/region of the Hi-C input.
    """
    try:
        df = pd.read_csv(path, sep='\t')
        unique_clusters = df['cluster_name'].unique()
        print(f"[INFO] Found {len(unique_clusters)} unique cell clusters in the ATAC file: {list(unique_clusters)}")
        
        # We will assume a specific cluster, e.g., 'beta_1', corresponds 
        # to the input Hi-C matrix and generate a placeholder feature.
        # This is a dummy feature vector of size ATAC_FEATURE_DIM
        placeholder_features = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], 
                                            dtype=torch.float32, 
                                            device=device).unsqueeze(0)
        
        print(f"[INFO] Using placeholder ATAC feature vector of size {ATAC_FEATURE_DIM} for model input.")
        return placeholder_features
    
    except Exception as e:
        print(f"[ERROR] Could not load or process ATAC file: {e}")
        # Fallback to zero tensor if file fails to load
        return torch.zeros(1, ATAC_FEATURE_DIM, dtype=torch.float32, device=device)


def build_model(device):
    """Initializes the ScUnicorn model with ATAC-seq dimension."""
    # Set dimensions based on the mock structure defined above
    model = ScUnicorn(atac_dim=ATAC_FEATURE_DIM, chip_dim=0, rna_dim=0)
    model.to(device)
    # Set to evaluation mode
    model.eval() 
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Loads model weights, handling potential 'module.' prefix."""
    # In a real scenario, you'd load actual weights here.
    # Since this is a mock, we skip actual loading but print a message.
    print(f"[INFO] Mock: Skipping actual checkpoint loading from {checkpoint_path}. Using random weights.")


def run_inference(model, hic_image, atac_tensor, device):
    """Runs the multimodal inference using Hi-C image and ATAC features."""
    # 1. Prepare Hi-C Input
    # Convert PIL Image (H,W) to Tensor (1, 1, H, W)
    hic_tensor = TF.to_tensor(hic_image.convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        # 2. Pass through Encoders
        # Get Hi-C feature vector
        hic_feat_flat = model.hic_encoder(hic_tensor)

        # Get ATAC-seq feature vector
        atac_feat_flat = model.atac_net(atac_tensor)

        # 3. Concatenate and Fusion
        # Concatenate features from all modalities (Hi-C and ATAC-seq)
        fused_input = torch.cat([hic_feat_flat, atac_feat_flat], dim=1)
        
        # Get target dimension of fusion layer input from the mock model
        target_dim = model.fusion_net.fc_fuse[0].in_features
        current_dim = fused_input.shape[1]

        # Check for expected fusion input dimension (should match the mock's FUSION_INPUT_DIM)
        if current_dim != target_dim:
             # This block handles unexpected dimension mismatches at runtime
             # Should not happen if the MockScUnicorn is correctly defined based on the original model
            print(f"[ERROR] Fusion input dimension mismatch! Expected {target_dim}, but got {current_dim}")
            raise RuntimeError("Fusion dimension mismatch.")

        # Run through fusion and reconstruction
        fused_out = model.fusion_net.fc_fuse(fused_input)
        out_tensor = model.fusion_net.reconstruct(fused_out)

        # 4. Process Output
        # Flatten output
        out_tensor = out_tensor.view(-1)

        # Find nearest square size for visualization (Assumes output is a matrix)
        n = int(np.sqrt(out_tensor.numel()))
        if n * n != out_tensor.numel():
            print(f"[WARN] Output size {out_tensor.numel()} is not a perfect square. Truncating to {n*n}.")
            out_tensor = out_tensor[: n * n]

        # Reshape to (1,1,H,W)
        out_tensor = out_tensor.view(1, 1, n, n)
        arr = out_tensor.cpu().squeeze().numpy()

    # Normalize output image
    arr = np.nan_to_num(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    return Image.fromarray(arr.astype(np.uint8)), arr


def main(args):
    """Main function to run the multimodal super-resolution inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Hi-C Data
    hic_mat = load_hic_matrix(args.data_path)
    # Convert to image for model input
    hic_image, scale_params = hic_matrix_to_image(hic_mat)
    
    # 2. Load ATAC-seq Features
    # The loaded file is used here as the source for ATAC-seq features
    atac_tensor = load_atac_features(args.atac_data_path, device)
    
    # 3. Build and Load Model
    model = build_model(device)
    load_checkpoint(model, args.model_path, device)

    # 4. Run Multimodal Inference
    hr_image, numeric_output = run_inference(model, hic_image, atac_tensor, device)

    # 5. Save Results
    os.makedirs(os.path.dirname(args.output_image_path) or ".", exist_ok=True)
    hr_image.save(args.output_image_path)
    print(f"[INFO] Enhanced image saved to {args.output_image_path}")

    # Restore the output image back to a Hi-C matrix
    restored_hic = image_to_hic_matrix(hr_image, scale_params)
    save_hic(restored_hic, args.output_hic_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR Hi-C map using ScUnicorn (Multimodal).")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pytorch)")
    parser.add_argument("--data_path", required=True, help="Path to Hi-C input (.txt/.npy/.npz)")
    parser.add_argument("--atac_data_path", required=True, help="Path to ATAC-seq features/metadata (.txt)")
    parser.add_argument("--output_image_path", required=True, help="Path to save enhanced image (.png)")
    parser.add_argument("--output_hic_path", required=True, help="Path to save enhanced Hi-C matrix (.txt)")
    args = parser.parse_args()
    main(args)