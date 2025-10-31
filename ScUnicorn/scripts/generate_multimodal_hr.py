#!/usr/bin/env python3
"""
scripts/generate_multimodal_hr.py

Inference script for ScUnicorn with optional multi-modal inputs:
    - Hi-C input (.txt, .npy, .npz) or image (.png)
    - Optional ATAC, ChIP, RNA inputs (each as .npz with key 'data' or a plain .npy array)
Outputs:
    - Enhanced image (.png)
    - Enhanced Hi-C matrix (.txt or .npz) when Hi-C input was provided

Usage example:
python scripts/generate_multimodal_hr.py \
    --model_checkpoint checkpoint/scunicorn_model.pytorch \
    --hic_input data/matrix_chr3_100kb.npy \
    --atac_input data/atac_sample.npz \
    --chip_input data/chip_sample.npz \
    --rna_input data/rna_sample.npz \
    --output_image output/enhanced_chr3.png \
    --output_hic output/enhanced_chr3.txt
"""
import argparse
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Import model - ensure models package is in PYTHONPATH or use relative import
from models.scunicorn import ScUnicorn


# --------------------
# Utilities: IO and transforms
# --------------------
def load_hic_matrix(path):
    """Load a Hi-C matrix from .txt, .npy or .npz. Returns numpy array."""
    if path.endswith(".txt"):
        mat = np.loadtxt(path)
    elif path.endswith(".npy"):
        mat = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        # pick first array-like entry if key unknown
        key = list(data.keys())[0]
        mat = data[key]
    else:
        raise ValueError("Unsupported Hi-C file format. Use .txt, .npy or .npz")
    return mat.astype(np.float32)


def hic_matrix_to_image(mat):
    """Convert Hi-C matrix to PIL image using log1p scaling and 0-255 normalization."""
    mat = np.array(mat, dtype=np.float32)
    # avoid negative or NaN
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.log1p(mat)
    mn, mx = mat.min(), mat.max()
    if mx <= mn:
        scaled = np.zeros_like(mat, dtype=np.uint8)
    else:
        scaled = ((mat - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return Image.fromarray(scaled), (mn, mx)


def image_to_hic_matrix(img, scale_params):
    """Convert a PIL image produced from hic_matrix_to_image back to a Hi-C matrix."""
    arr = np.array(img).astype(np.float32)
    mn, mx = scale_params
    # reverse normalization
    if mx <= mn:
        restored = np.zeros_like(arr, dtype=np.float32)
    else:
        restored = arr / 255.0 * (mx - mn) + mn
    restored = np.expm1(restored)  # reverse log1p
    return restored


def save_hic(matrix, out_path, fmt="txt"):
    """Save Hi-C matrix to txt or npz file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if fmt == "txt":
        np.savetxt(out_path, matrix, fmt="%.6f", delimiter="\t")
    else:
        np.savez_compressed(out_path, hic=matrix)
    print(f"[INFO] Saved Hi-C matrix to {out_path}")


def load_omics_vector(path):
    """Load omics vector data. Accepts .npz with key 'data' or any key, or .npy file."""
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Omics file not found: {path}")
    if path.endswith(".npz"):
        arr = np.load(path)
        # prefer 'data' key if exists
        key = "data" if "data" in arr.keys() else list(arr.keys())[0]
        vec = arr[key]
    elif path.endswith(".npy"):
        vec = np.load(path)
    else:
        raise ValueError("Unsupported omics format. Use .npz or .npy")
    return np.asarray(vec, dtype=np.float32)


# --------------------
# Model loading & inference
# --------------------
def build_model_from_inputs(atac_vec, chip_vec, rna_vec, device):
    """Construct ScUnicorn with appropriate modality dims inferred from provided arrays."""
    atac_dim = int(atac_vec.shape[1]) if (atac_vec is not None and atac_vec.ndim == 2) else (int(atac_vec.size) if atac_vec is not None and atac_vec.ndim == 1 else None)
    chip_dim = int(chip_vec.shape[1]) if (chip_vec is not None and chip_vec.ndim == 2) else (int(chip_vec.size) if chip_vec is not None and chip_vec.ndim == 1 else None)
    rna_dim = int(rna_vec.shape[1]) if (rna_vec is not None and rna_vec.ndim == 2) else (int(rna_vec.size) if rna_vec is not None and rna_vec.ndim == 1 else None)

    # If dims are 1D vectors, use their length
    atac_dim = atac_dim if atac_dim and atac_dim > 0 else None
    chip_dim = chip_dim if chip_dim and chip_dim > 0 else None
    rna_dim = rna_dim if rna_dim and rna_dim > 0 else None

    print(f"[INFO] Model modalities dims -> ATAC: {atac_dim}, ChIP: {chip_dim}, RNA: {rna_dim}")
    model = ScUnicorn(atac_dim=atac_dim, chip_dim=chip_dim, rna_dim=rna_dim)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint if available."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        # maybe directly saved state_dict
        state_dict = ckpt
    else:
        state_dict = ckpt
    # Handle possible DataParallel keys
    try:
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    except RuntimeError:
        # try to strip "module." prefixes
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        model.load_state_dict(new_state)
        print(f"[INFO] Loaded checkpoint (stripped module.) from {checkpoint_path}")


def run_inference(model, hic_image, atac_vec=None, chip_vec=None, rna_vec=None, device="cpu"):
    """
    Run the model on a single sample.
    hic_image: PIL image converted from the Hi-C matrix (mode 'L' or 'RGB')
    omics vectors: numpy arrays with shape (1, dim) or (dim,)
    Returns:
        hr_image (PIL Image)
        enhanced_hic_matrix (np.array)
    """
    # Convert hic_image to tensor (single channel)
    img = hic_image.convert("L")
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # shape: (1, 1, H, W)

    # Prepare omics tensors if present
    def to_torch_vec(vec):
        if vec is None:
            return None
        v = torch.tensor(vec, dtype=torch.float32)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        return v.to(device)

    atac_t = to_torch_vec(atac_vec)
    chip_t = to_torch_vec(chip_vec)
    rna_t = to_torch_vec(rna_vec)

    with torch.no_grad():
        # model expects hi-c, atac, chip, rna
        out = model(tensor, atac_t, chip_t, rna_t)
        # out shape expected (B, 1, H_out, W_out)
        if isinstance(out, tuple) or isinstance(out, list):
            out_tensor = out[0]
        else:
            out_tensor = out
        out_tensor = out_tensor.cpu().squeeze(0)  # (1, H, W) -> (1,H,W) or (H,W)
        # Ensure shape is (H,W)
        if out_tensor.ndim == 3 and out_tensor.shape[0] == 1:
            out_arr = out_tensor.squeeze(0).numpy()
        else:
            out_arr = out_tensor.numpy()

    # Map numeric output back to image 0-255. We did not use Tanh in this script so clamp and normalize.
    arr = out_arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = arr.min(), arr.max()
    if mx <= mn:
        img_out_arr = (arr * 0).astype(np.uint8)
    else:
        img_out_arr = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    hr_image = Image.fromarray(img_out_arr)

    # Return hr_image and numeric array (we will reconstruct Hi-C from hr_image using scale params passed externally)
    return hr_image, arr


# --------------------
# Main
# --------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    # Load Hi-C input
    is_image_input = args.hic_input.endswith(".png")
    if is_image_input:
        hic_image = Image.open(args.hic_input).convert("L")
        # For images, we do not have scale params; set dummy params
        scale_params = (0.0, 1.0)
        print(f"[INFO] Loaded Hi-C image {args.hic_input}")
    else:
        hic_mat = load_hic_matrix(args.hic_input)
        hic_image, scale_params = hic_matrix_to_image(hic_mat)
        print(f"[INFO] Loaded Hi-C matrix {args.hic_input} and converted to image. scale_params={scale_params}")

    # Load optional omics
    atac_vec = load_omics_vector(args.atac_input) if args.atac_input else None
    chip_vec = load_omics_vector(args.chip_input) if args.chip_input else None
    rna_vec = load_omics_vector(args.rna_input) if args.rna_input else None

    # If loaded omics are 2D arrays with multiple rows, take first sample by default
    def ensure_single_sample(arr):
        if arr is None:
            return None
        if arr.ndim == 2 and arr.shape[0] > 1:
            return arr[0:1]
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    atac_vec = ensure_single_sample(atac_vec)
    chip_vec = ensure_single_sample(chip_vec)
    rna_vec = ensure_single_sample(rna_vec)

    # Build model and load checkpoint
    model = build_model_from_inputs(atac_vec, chip_vec, rna_vec, device)
    load_checkpoint(model, args.model_checkpoint, device)

    # Run inference
    hr_image, out_numeric = run_inference(model, hic_image, atac_vec, chip_vec, rna_vec, device)

    # Save enhanced image
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    hr_image.save(args.output_image)
    print(f"[INFO] Enhanced image saved to {args.output_image}")

    # If Hi-C input was a matrix, convert hr_image back to Hi-C using original scale params and save
    if not is_image_input and args.output_hic:
        enhanced_hic = image_to_hic_matrix(hr_image, scale_params)
        save_hic(enhanced_hic, args.output_hic, fmt=args.hic_format)
    elif is_image_input and args.output_hic:
        # if input was an image and the user requested a Hi-C output, use the numeric output array and scale heuristically
        # here we apply a simple linear mapping using mean and range of original image if provided
        print("[WARN] Input was an image. Saved Hi-C will be derived from model output directly with heuristic scaling.")
        # naive inverse: scale out_numeric to a similar dynamic range as input image then expm1
        out_arr = out_numeric
        # normalize to 0-1
        if out_arr.max() > out_arr.min():
            norm = (out_arr - out_arr.min()) / (out_arr.max() - out_arr.min())
        else:
            norm = out_arr * 0.0
        # map back to 0..1 then apply expm1 of some small range as heuristic
        enhanced_hic = np.expm1(norm * 4.0)  # 4.0 is arbitrary; replace if you want domain specific scaling
        save_hic(enhanced_hic, args.output_hic, fmt=args.hic_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-modal enhanced Hi-C using ScUnicorn.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model checkpoint (.pytorch/.pt).")
    parser.add_argument("--hic_input", type=str, required=True, help="Path to Hi-C input (.txt, .npy, .npz) or image (.png).")
    parser.add_argument("--atac_input", type=str, default=None, help="Optional ATAC-seq vector (.npz or .npy).")
    parser.add_argument("--chip_input", type=str, default=None, help="Optional ChIP-seq vector (.npz or .npy).")
    parser.add_argument("--rna_input", type=str, default=None, help="Optional RNA-seq vector (.npz or .npy).")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save enhanced image (.png).")
    parser.add_argument("--output_hic", type=str, default=None, help="Path to save enhanced Hi-C matrix (.txt or .npz).")
    parser.add_argument("--hic_format", type=str, choices=["txt", "npz"], default="txt", help="Format to save Hi-C matrix.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    main(args)
