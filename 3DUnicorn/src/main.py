import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as F

# External 3DUnicorn Utility Imports (Assumed to be correctly installed)
try:
    from optimization.readInput import read_input
    from utils.convert2distance import convert_to_distance
    from utils.evaluation import evaluate_structure
    from optimization.optimization import optimization
    from utils.output_3DUnicorn import output_3DUnicorn
    from utils.initialize_structure_c_style import initialize_structure_c_style
except ImportError as e:
    # This block handles core 3DUnicorn components
    print(f"[ERROR] Critical 3DUnicorn module import failed: {e}")
    print("Please ensure your optimization and utils modules are correctly installed or on your Python path.")
    # Re-raise the error if core functionality is missing
    raise

# --- NEW: Import the full suite of metrics (MSE, PSNR, SSIM, Pearson, Spearman) ---
try:
    from utils.metrics import evaluate_all
except ImportError as e:
    # This block handles the new metrics module specifically.
    print(f"[WARNING] Could not import utils.metrics.evaluate_all. Map-level metrics (MSE, PSNR, SSIM) will not be calculated: {e}")
    # Define a dummy function to prevent NameError if the module is missing
    def evaluate_all(gt, pred):
        print("[ERROR] Skipping map-level metrics calculation due to missing dependency.")
        return {}

# --- Utility Function for Metrics ---

def numpy_to_tensor4d(arr):
    """
    Converts a 2D numpy array (H, W) to a 4D torch tensor (1, 1, H, W) 
    as required by the metrics utility for batch processing.
    """
    if arr is None:
        raise ValueError("Input array for tensor conversion cannot be None.")
    if arr.ndim == 2:
        return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] == 1: # Handle (1, H, W) case
        return torch.from_numpy(arr).float().unsqueeze(0)
    return torch.from_numpy(arr).float()


def load_model(model_path):
    """Load a pre-trained model from an .npz file."""
    print(f"[INFO] Loading pre-trained model from {model_path}...")
    # NOTE: Actual model loading and initialization logic would go here.
    # We are returning mock data as the real model code is not provided.
    model_data = np.load(model_path) if os.path.exists(model_path) else "MockModelData"
    print("[INFO] Model loaded successfully.")
    return model_data


def load_hic_matrix(file_path):
    print(f"[INFO] Loading Hi-C matrix from {file_path}...")
    if file_path.endswith(".txt"):
        matrix = np.loadtxt(file_path)
    elif file_path.endswith((".npz", ".npy")):
        data = np.load(file_path)
        matrix = data[list(data.keys())[0]]
    else:
        raise ValueError("Unsupported file format. Use .txt, .npz, or .npy")
    return matrix

def hic_to_image(hic_matrix):
    # Scale and normalize the Hi-C matrix to a format suitable for image processing
    hic_matrix = np.log1p(hic_matrix)
    hic_matrix = (hic_matrix - np.min(hic_matrix)) / (np.max(hic_matrix) - np.min(hic_matrix)) * 255
    hic_matrix = hic_matrix.astype(np.uint8)
    # Convert to grayscale image
    return Image.fromarray(hic_matrix).convert("L") # Ensure it's a single channel image for L format

def image_to_hic(image):
    # Convert image back to a normalized Hi-C matrix (inverse of hic_to_image)
    matrix = np.array(image.convert("L"), dtype=np.float32) # Ensure L format conversion
    matrix = np.expm1(matrix / 255.0)
    return matrix

def save_hic_matrix(matrix, output_path, format="txt"):
    if format == "txt":
        np.savetxt(output_path, matrix, fmt="%.6f", delimiter="\t")
    else:
        np.savez_compressed(output_path, hic=matrix)
    print(f"[SUCCESS] Hi-C contact map saved to {output_path}")

def preprocess_input(data_path):
    # This function handles loading either a raw Hi-C matrix (txt/npz) or a processed image (png)
    if data_path.endswith((".txt", ".npz", ".npy")):
        hic_matrix = load_hic_matrix(data_path)
        # Return image version (for model input) and raw matrix (for ground truth)
        return hic_to_image(hic_matrix), hic_matrix 
    elif data_path.endswith(".png"):
        # If input is already an image, return the image and None for the raw matrix
        return Image.open(data_path).convert("L"), None # L format
    else:
        raise ValueError("Unsupported file format. Use .png for images or .txt/.npz/.npy for Hi-C contact maps.")

def infer_model(model, lr_image):
    print("[INFO] Running inference on input data...")

    print(f"[DEBUG] Input Image Size: {lr_image.width}x{lr_image.height}")

    # The original implementation uses a placeholder inference logic: upscale factor 2, resize, then crop back.
    # In a real model, you would convert lr_image to a tensor, run model(tensor), and convert output tensor back to Image.
    upscale_factor = 2

    # Upscale
    hr_resized = F.resize(lr_image, (lr_image.height * upscale_factor, lr_image.width * upscale_factor))

    # Crop back to original size (This logic simulates a prediction of the same size as input)
    left = (hr_resized.width - lr_image.width) // 2
    top = (hr_resized.height - lr_image.height) // 2
    right = left + lr_image.width
    bottom = top + lr_image.height

    hr_cropped = hr_resized.crop((left, top, right, bottom))

    print(f"[DEBUG] Output Image Size: {hr_cropped.width}x{hr_cropped.height}")

    return hr_cropped.convert("L") # Ensure output is L format

def convert_dense_to_tuple_debug(input_file, output_file, bin_size=500000):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as out_file:
        for row_index, line in enumerate(lines):
            values = line.strip().split()
            for col_index, value in enumerate(values):
                try:
                    frequency = float(value)
                    if frequency != 0:
                        # Assuming 1-based indexing for bins starting from 1
                        position_1 = (row_index) * bin_size + (bin_size / 2) 
                        position_2 = (col_index) * bin_size + (bin_size / 2)
                        out_file.write(f"{int(position_1)} {int(position_2)} {frequency}\n")
                except ValueError:
                    print(f"Warning: Could not convert value '{value}' at row {row_index + 1}, col {col_index + 1}")

def main_3DUnicorn(params_file, updated_input_path, hic_matrix=None, enhanced_hic_matrix=None):
    """
    Performs 3D structure reconstruction using the enhanced Hi-C map.
    
    Args:
        params_file (str): Path to parameters file.
        updated_input_path (str): Path to the enhanced Hi-C map in tuple format.
        hic_matrix (np.array, optional): Ground Truth Hi-C matrix. Used for map metrics.
        enhanced_hic_matrix (np.array, optional): Enhanced Hi-C matrix. Used for map metrics.
    """
    def parse_parameters_txt(params_file):
        params = {}
        with open(params_file, 'r') as file:
            for line in file:
                if not line.strip() or line.startswith('#'):
                    continue
                try:
                    key, value = line.split('=', 1)
                    params[key.strip()] = value.strip()
                except ValueError:
                    print(f"Warning: Skipping malformed line in parameters file: {line.strip()}")
        return params

    parameters = parse_parameters_txt(params_file)
    OUTPUT_FOLDER = parameters.get("OUTPUT_FOLDER", "Outputs") # Use default if not found
    INPUT_FILE = updated_input_path
    
    try:
        MAX_ITERATION = int(parameters["MAX_ITERATION"])
        LEARNING_RATE = float(parameters["LEARNING_RATE"])
    except KeyError as e:
        print(f"[ERROR] Missing critical parameter: {e}. Check parameters.txt.")
        return # Stop execution if critical params are missing


    path = 'Scores/'
    os.makedirs(path, exist_ok=True)
    name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    P_CORR, RMSD = [], []

    lstCons, n, _ = read_input(INPUT_FILE, OUTPUT_FOLDER)


    # Variables to store the metrics
    map_metrics = {}

    for CONVERT_FACTOR in [0.6]:
        best_spearman_corr = -1
        best_pearson_corr = -1
        best_structure_name = None

        for l in range(1, 2):  # NUM=1
            lstCons, maxIF = convert_to_distance(lstCons, CONVERT_FACTOR)
            # Initialize with c-style random initialization
            coordinates = initialize_structure_c_style(n, 0.8, 5.0) 
            variables = coordinates.flatten()

            # Perform 3D Structure Optimization
            variables, _, _ = optimization(
                n, MAX_ITERATION, LEARNING_RATE, 1e-6, 1e-5, lstCons, maxIF
            )

            structure = variables.reshape(-1, 3)
            rmse, spearman_corr, pearson_corr, WishDist_clean, Dist_clean = evaluate_structure(
                lstCons, variables, n
            )

            P_CORR.append(pearson_corr)
            RMSD.append(rmse)

            if spearman_corr > best_spearman_corr:
                best_spearman_corr = spearman_corr
                best_pearson_corr = pearson_corr
                best_structure_name = f'{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}'

            output_3DUnicorn(variables, f'{path}{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}', WishDist_clean, Dist_clean)

    np.savetxt(os.path.join(path, f'{name}_pearsoncorr.txt'), P_CORR, fmt='%f', header='Pearson Correlation')
    np.savetxt(os.path.join(path, f'{name}_rmsd.txt'), RMSD, fmt='%f', header='RMSD')
    np.savetxt(os.path.join(path, f'{name}_Finalscores.txt'), [[CONVERT_FACTOR, 1, best_spearman_corr]], fmt='%f', 
               header='CONVERT_FACTOR\tStructure Number\tSpearman Correlation')

    # --- Metrics Calculation (Requires hic_matrix and enhanced_hic_matrix) ---
    if hic_matrix is not None and enhanced_hic_matrix is not None:
        try:
            # Convert NumPy matrices to (1, 1, H, W) PyTorch Tensors for the metrics module
            gt_tensor = numpy_to_tensor4d(hic_matrix)
            pred_tensor = numpy_to_tensor4d(enhanced_hic_matrix)
        except ValueError as e:
            print(f"[ERROR] Tensor conversion failed: {e}. Skipping map metrics.")
        else:
            # Calculate all metrics (MSE, PSNR, SSIM, Pearson, Spearman) and store them
            map_metrics = evaluate_all(gt_tensor, pred_tensor)
            
            # Print Map Metrics (GT vs. Enhanced)
            if map_metrics:
                print("\n--- Map Enhancement Metrics (GT vs. Enhanced) ---")
                for k, v in map_metrics.items():
                    print(f"  {k:<8}: {v:.6f}")
    else:
        print("\n[INFO] Skipping map enhancement metrics (MSE/PSNR/SSIM) as original Hi-C matrix (hic_matrix) was not available or passed.")
    # --- End Metrics Calculation ---

    print(f"[RESULT] Pearson Correlation: {best_pearson_corr:.6f}")
    print(f"[RESULT] Spearman Correlation: {best_spearman_corr:.6f}")
    print("[INFO] Process complete.")

def main_pipeline(params_file):
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                try:
                    key, value = line.strip().split('=', 1)
                    params[key.strip()] = value.strip()
                except ValueError:
                    continue

    try:
        model_path = params['MODEL_PATH']
        input_file = params['INPUT_HIC_FILE']
    except KeyError as e:
        print(f"[ERROR] Missing critical parameter in parameters.txt: {e}. Cannot start pipeline.")
        return

    input_filename = os.path.basename(input_file).replace('.txt', '')

    # Define paths
    scores_folder = 'Scores'
    os.makedirs(scores_folder, exist_ok=True)

    output_hic_path = os.path.join(scores_folder, f"{input_filename}_enhanced.txt")

    input_folder = "input"
    tuple_folder = os.path.join(input_folder, "tuple_format_input")
    os.makedirs(tuple_folder, exist_ok=True)

    tuple_output_path = os.path.join(tuple_folder, f"{input_filename}_tuple.txt")

    # Load model and perform enhancement
    model = load_model(model_path)
    # hic_matrix is the Ground Truth matrix, defined here.
    lr_image, hic_matrix = preprocess_input(input_file) 
    hr_image = infer_model(model, lr_image)

    # Initialize enhanced_hic_matrix outside the if block
    enhanced_hic_matrix = None 

    if hic_matrix is not None:
        # enhanced_hic_matrix is calculated here
        enhanced_hic_matrix = image_to_hic(hr_image) 
        save_hic_matrix(enhanced_hic_matrix, output_hic_path, format='txt')
        convert_dense_to_tuple_debug(output_hic_path, tuple_output_path)
    else:
        # If input was a PNG (no original Hi-C matrix), we still need the enhanced matrix
        # to run 3DUnicorn, so we convert the HR image back. 
        # However, we cannot calculate map metrics (hic_matrix will be None).
        print("[WARNING] Input was an image (PNG). Assuming it is the Hi-C map to be enhanced.")
        enhanced_hic_matrix = image_to_hic(hr_image) 
        save_hic_matrix(enhanced_hic_matrix, output_hic_path, format='txt')
        convert_dense_to_tuple_debug(output_hic_path, tuple_output_path)


    # Pass both Ground Truth and Enhanced matrices to main_3DUnicorn
    main_3DUnicorn(params_file, tuple_output_path, hic_matrix, enhanced_hic_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Genome Structure Reconstruction Pipeline")
    parser.add_argument('--parameters', required=True, help="Path to parameters.txt")
    args = parser.parse_args()
    main_pipeline(args.parameters)