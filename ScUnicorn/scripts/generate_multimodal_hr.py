#!/usr/bin/env python3
"""
scripts/generate_multimodal_hr.py

Inference script for ScUnicorn with automatic dimension alignment
between Hi-C encoder outputs and fusion layers.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.scunicorn import ScUnicorn


def load_hic_matrix(path):
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
    mat = np.nan_to_num(mat)
    mat = np.log1p(mat)
    mn, mx = mat.min(), mat.max()
    scaled = ((mat - mn) / (mx - mn + 1e-8) * 255.0).astype(np.uint8)
    return Image.fromarray(scaled), (mn, mx)


def image_to_hic_matrix(img, scale_params):
    arr = np.array(img).astype(np.float32)
    mn, mx = scale_params
    restored = arr / 255.0 * (mx - mn) + mn
    return np.expm1(restored)


def save_hic(matrix, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savetxt(out_path, matrix, fmt="%.6f", delimiter="\t")
    print(f"[INFO] Saved Hi-C matrix to {out_path}")


def build_model(device):
    model = ScUnicorn(atac_dim=None, chip_dim=None, rna_dim=None)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    if missing:
        print(f"[WARN] Missing keys ignored: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys ignored: {unexpected}")

    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")


def run_inference(model, hic_image, device):
    tensor = TF.to_tensor(hic_image.convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through Hi-C encoder only to get feature dimension
        hic_feat = model.hic_encoder(tensor)

        # Flatten
        hic_feat_flat = torch.flatten(hic_feat, 1)

        # Get target dimension of fusion layer input
        target_dim = model.fusion_net.fc_fuse[0].in_features
        current_dim = hic_feat_flat.shape[1]

        # Pad or truncate to match
        if current_dim < target_dim:
            pad = target_dim - current_dim
            hic_feat_flat = torch.cat([hic_feat_flat, torch.zeros(1, pad, device=device)], dim=1)
            print(f"[INFO] Padded Hi-C features from {current_dim} → {target_dim}")
        elif current_dim > target_dim:
            hic_feat_flat = hic_feat_flat[:, :target_dim]
            print(f"[INFO] Truncated Hi-C features from {current_dim} → {target_dim}")

        # Run through fusion and reconstruction
        fused_out = model.fusion_net.fc_fuse(hic_feat_flat)
        out_tensor = model.fusion_net.reconstruct(fused_out)

        # Flatten output
        out_tensor = out_tensor.view(-1)

        # Find nearest square size for visualization
        n = int(np.sqrt(out_tensor.numel()))
        if n * n != out_tensor.numel():
            print(f"[WARN] Output size {out_tensor.numel()} is not a perfect square. Truncating to {n*n}.")
            out_tensor = out_tensor[: n * n]

        # Reshape to (1,1,H,W)
        out_tensor = out_tensor.view(1, 1, n, n)
        arr = out_tensor.cpu().squeeze().numpy()


    # Normalize output
    arr = np.nan_to_num(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    return Image.fromarray(arr.astype(np.uint8)), arr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hic_mat = load_hic_matrix(args.data_path)
    hic_image, scale_params = hic_matrix_to_image(hic_mat)

    model = build_model(device)
    load_checkpoint(model, args.model_path, device)

    hr_image, numeric_output = run_inference(model, hic_image, device)

    os.makedirs(os.path.dirname(args.output_image_path) or ".", exist_ok=True)
    hr_image.save(args.output_image_path)
    print(f"[INFO] Enhanced image saved to {args.output_image_path}")

    restored_hic = image_to_hic_matrix(hr_image, scale_params)
    save_hic(restored_hic, args.output_hic_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HR Hi-C map using ScUnicorn.")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.pytorch)")
    parser.add_argument("--data_path", required=True, help="Path to Hi-C input (.txt/.npy/.npz)")
    parser.add_argument("--output_image_path", required=True, help="Path to save enhanced image (.png)")
    parser.add_argument("--output_hic_path", required=True, help="Path to save enhanced Hi-C matrix (.txt)")
    args = parser.parse_args()
    main(args)
