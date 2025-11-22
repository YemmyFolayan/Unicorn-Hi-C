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
    if arr.ndim == 2:
        return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] == 1: # Handle (1, H, W) case
        return torch.from_numpy(arr).float().unsqueeze(0)
    return torch.from_numpy(arr).float()


# --- Multimodal Feature Loading Functions ---

def load_atac_features(file_path):
    """Loads ATAC-seq UMAP features from the file (GSE160472_ATAC_Seq.txt)."""
    print(f"[INFO] Loading ATAC features from {file_path}...")
    try:
        # Assuming the file structure starts with a header, followed by data
        df = pd.read_csv(file_path, sep='\t', header=0)
        feature_array = df[['UMAP1', 'UMAP2']].values.astype(np.float32)
        print(f"[INFO] Successfully loaded ATAC features (Shape: {feature_array.shape}).")
        return feature_array.mean(axis=0)
    except Exception as e:
        print(f"[ERROR] Failed to load ATAC features from {file_path}: {e}. Returning zeros.")
        return np.array([0.0, 0.0], dtype=np.float32)

def load_chip_features(file_path):
    """Loads specific numeric ChIP-seq features (e.g., histone marks from GSE269897_CHIP_Seq.txt)."""
    print(f"[INFO] Loading ChIP features from {file_path}...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Target the histone mark columns for feature extraction
        numeric_cols = ['H3K27me3_encode', 'DNase_encode', 'H3K27ac_encode', 
                        'H3K4me1_encode', 'H3K4me3_encode', 'H3K9ac_encode']
        
        feature_series = df[numeric_cols].mean()
        feature_array = feature_series.values.astype(np.float32)
        
        print(f"[INFO] Successfully loaded {feature_array.size} ChIP features.")
        return feature_array
    except Exception as e:
        print(f"[ERROR] Failed to load ChIP features from {file_path}: {e}. Returning zeros.")
        return np.zeros(6, dtype=np.float32)

def load_rna_features(file_path):
    """Loads RNA-seq features (e.g., average FPKM from GSE287905_RNA_Seq.txt)."""
    print(f"[INFO] Loading RNA features from {file_path}...")
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        rna_cols = ['shCon.FPKM.1', 'shCon.FPKM.2', 'shCon.FPKM.3']
        avg_fpkm = df[rna_cols].values.mean()
        
        print(f"[INFO] Successfully loaded RNA feature (Avg FPKM): {avg_fpkm:.4f}")
        return np.array([avg_fpkm], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to load RNA features from {file_path}: {e}. Returning zero.")
        return np.array([0.0], dtype=np.float32)

# --- Core Hi-C Processing Functions (From User's Original main.py) ---

def load_model(model_path):
    """Load a pre-trained model from an .npz file (Placeholder for actual model logic)."""
    print(f"[INFO] Loading pre-trained model from {model_path}...")
    try:
        model_data = np.load(model_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"[WARNING] Model file not found at {model_path}. Returning dummy data.")
        return {"placeholder": True}
    print("[INFO] Model loaded successfully.")
    return model_data

def load_hic_matrix(file_path):
    """Loads a pure numerical Hi-C matrix from various formats."""
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
    """Converts a Hi-C matrix (log-transformed and normalized) to a PIL Image."""
    hic_matrix = np.log1p(hic_matrix)
    hic_matrix = (hic_matrix - np.min(hic_matrix)) / (np.max(hic_matrix) - np.min(hic_matrix)) * 255
    hic_matrix = hic_matrix.astype(np.uint8)
    return Image.fromarray(hic_matrix).convert("RGB")

def image_to_hic(image):
    """Converts a PIL Image back to a Hi-C matrix (inverse normalization and exponential)."""
    matrix = np.array(image.convert('L'), dtype=np.float32)
    matrix = np.expm1(matrix / 255.0) 
    return matrix

def save_hic_matrix(matrix, output_path, format="txt"):
    """Saves the enhanced Hi-C matrix."""
    if format == "txt":
        np.savetxt(output_path, matrix, fmt="%.6f", delimiter="\t")
    else:
        np.savez_compressed(output_path, hic=matrix)
    print(f"[SUCCESS] Hi-C contact map saved to {output_path}")

def preprocess_input(data_path):
    """Handles both image and matrix Hi-C inputs."""
    if data_path.endswith((".txt", ".npz", ".npy")):
        hic_matrix = load_hic_matrix(data_path)
        return hic_to_image(hic_matrix), hic_matrix
    elif data_path.endswith(".png"):
        return Image.open(data_path).convert("RGB"), None
    else:
        raise ValueError("Unsupported file format. Use .png for images or .txt/.npz/.npy for Hi-C contact maps.")

def infer_model(model, lr_image, multimodal_features):
    """
    Placeholder for the core ScUnicorn/Enhancement inference.
    It now accepts multimodal features, which an actual model would use.
    """
    print("[INFO] Running inference on input data (Multimodal features received).")
    print(f"[DEBUG] Input Image Size: {lr_image.width}x{lr_image.height}")

    upscale_factor = 2
    
    # Simple placeholder: resize, convert, and crop to simulate HR output
    hr_resized = F.resize(lr_image, (lr_image.height * upscale_factor, lr_image.width * upscale_factor))

    # Crop back to original size (just for the placeholder example to match dimensions)
    left = (hr_resized.width - lr_image.width) // 2
    top = (hr_resized.height - lr_image.height) // 2
    right = left + lr_image.width
    bottom = top + lr_image.height

    hr_cropped = hr_resized.crop((left, top, right, bottom))

    print(f"[DEBUG] Output Image Size: {hr_cropped.width}x{hr_cropped.height}")

    return hr_cropped

def convert_dense_to_tuple_debug(input_file, output_file, bin_size=500000):
    """
    Converts a dense matrix file to a 3-column (pos1, pos2, frequency) tuple format.
    """
    print(f"[INFO] Converting enhanced dense matrix to 3-column tuple format: {output_file}")
    try:
        matrix = np.loadtxt(input_file)
    except Exception as e:
        print(f"[FATAL ERROR] Could not load enhanced Hi-C matrix for tuple conversion: {e}")
        raise
        
    n = matrix.shape[0]
    
    with open(output_file, 'w') as out_file:
        for i in range(n):
            for j in range(i, n):
                frequency = matrix[i, j]
                if frequency > 1e-6:
                    position_1 = (i + 1) * bin_size
                    position_2 = (j + 1) * bin_size
                    # Use space/tab separated values, consistent with many file formats
                    out_file.write(f"{position_1}\t{position_2}\t{frequency:.6f}\n")
    print("[SUCCESS] Conversion to 3-column tuple format complete.")

# --- 3DUnicorn Reconstruction Core (FIXED LOGIC) ---

def main_3DUnicorn(params_file, updated_input_path):
    """
    Runs the 3D structure reconstruction pipeline and returns the best correlation results.
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
                    continue
        return params

    parameters = parse_parameters_txt(params_file)
    OUTPUT_FOLDER = parameters.get("OUTPUT_FOLDER")
    INPUT_FILE = updated_input_path
    MAX_ITERATION = int(parameters["MAX_ITERATION"])
    LEARNING_RATE = float(parameters["LEARNING_RATE"])
    
    path = 'Scores/'
    os.makedirs(path, exist_ok=True)
    name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    P_CORR, RMSD = [], []

    print(f"\n[INFO] Starting 3D structure reconstruction using input: {INPUT_FILE}")

    # Read input (3 columns: pos1, pos2, IF) - lstCons is likely a list of tuples/lists
    lstCons, n, _ = read_input(INPUT_FILE, OUTPUT_FOLDER)
    
    # 1. Ensure lstCons is a standard Python list of lists 
    lstCons = [list(item) for item in lstCons]

    # Hardcoded values from the user's original script
    CONVERT_FACTOR = 0.6
    NUM_STRUCTURES = 1

    best_spearman_corr = -1
    best_pearson_corr = -1
    best_structure_name = None

    for l in range(1, NUM_STRUCTURES + 1):
        print(f"[INFO] Generating structure {l}/{NUM_STRUCTURES}...")
        
        # 2. Create a copy and convert it to a NumPy array for compatibility with convert_to_distance
        lstCons_to_convert = np.array(lstCons, dtype=np.float32)
        
        # Convert to distance (This creates the 4-column constraint list)
        lstCons_dist, maxIF = convert_to_distance(lstCons_to_convert, CONVERT_FACTOR)
        
        # Debugging check
        if len(lstCons_dist[0]) != 4:
             print(f"[FATAL DEBUG] CONVERSION FAILED. lstCons_dist item has {len(lstCons_dist[0])} columns, expected 4.")
             raise ValueError("Conversion to 4-column constraint list failed, aborting evaluation.")
        else:
             print(f"[DEBUG] Conversion successful: lstCons_dist item has {len(lstCons_dist[0])} columns.")
        
        
        coordinates = initialize_structure_c_style(n, 0.8, 5.0)
        variables = coordinates.flatten()

        # Optimization uses the 4-column constraint list
        variables, _, _ = optimization(
            n, MAX_ITERATION, LEARNING_RATE, 1e-6, 1e-5, lstCons_dist, maxIF
        )

        structure = variables.reshape(-1, 3)
        
        # Evaluation uses the 4-column constraint list
        rmse, spearman_corr, pearson_corr, WishDist_clean, Dist_clean = evaluate_structure(
            lstCons_dist, variables, n # Use the explicitly converted and checked list
        )

        P_CORR.append(pearson_corr)
        RMSD.append(rmse)
        
        # Update best correlations
        if spearman_corr > best_spearman_corr:
            best_spearman_corr = spearman_corr
            best_pearson_corr = pearson_corr
            best_structure_name = f'{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}'
            
        print(f"[DEBUG] Structure {l} Pearson: {pearson_corr:.6f}, Spearman: {spearman_corr:.6f}")


        # Output the 3D structure
        output_3DUnicorn(variables, f'{path}{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}', WishDist_clean, Dist_clean)
        
    # Save final results
    np.savetxt(os.path.join(path, f'{name}_pearsoncorr.txt'), P_CORR, fmt='%f', header='Pearson Correlation')
    np.savetxt(os.path.join(path, f'{name}_rmsd.txt'), RMSD, fmt='%f', header='RMSD')
    np.savetxt(os.path.join(path, f'{name}_Finalscores.txt'), [[CONVERT_FACTOR, NUM_STRUCTURES, best_spearman_corr]], fmt='%f', 
               header='CONVERT_FACTOR\tStructure Number\tSpearman Correlation')

    print("[INFO] 3D Reconstruction process complete. Returning best results.")
    
    # Return the best values found across all structures
    return best_pearson_corr, best_spearman_corr

# --- Main Pipeline Orchestrator ---

def main_pipeline(params_file):
    """Orchestrates the multimodal feature loading, Hi-C enhancement, and 3D reconstruction."""
    
    # 
    
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
        hic_input_file = params['INPUT_HIC_FILE'] 
        atac_data_path = params['ATAC_DATA_PATH']
        chip_data_path = params['CHIP_DATA_PATH']
        rna_data_path = params['RNA_DATA_PATH']
    except KeyError as e:
        print(f"[FATAL ERROR] Missing critical key in parameters file: {e}. Aborting pipeline.")
        return
        
    input_filename = os.path.basename(hic_input_file).replace('.txt', '').replace('.npz', '').replace('.npy', '')

    # Define paths
    scores_folder = 'Scores'
    os.makedirs(scores_folder, exist_ok=True)
    output_hic_path = os.path.join(scores_folder, f"{input_filename}_enhanced.txt")

    tuple_folder = os.path.join("input", "tuple_format_input")
    os.makedirs(tuple_folder, exist_ok=True)
    tuple_output_path = os.path.join(tuple_folder, f"{input_filename}_tuple.txt")

    # 1. Load Multimodal Features (ATAC, ChIP, RNA)
    atac_features = load_atac_features(atac_data_path)
    chip_features = load_chip_features(chip_data_path)
    rna_features = load_rna_features(rna_data_path)
    
    multimodal_features = {
        "atac": atac_features,
        "chip": chip_features,
        "rna": rna_features
    }
    
    # 2. Hi-C Enhancement Step
    model = load_model(model_path)
    # hic_matrix (NumPy) is the ground truth (GT)
    lr_image, hic_matrix = preprocess_input(hic_input_file) 
    
    # Pass multimodal features to the (placeholder) inference model
    hr_image = infer_model(model, lr_image, multimodal_features) 

    # Variables to store the metrics
    map_metrics = {}
    best_3d_pearson = 0.0
    best_3d_spearman = 0.0

    # 3. Prepare Enhanced Hi-C Input for 3DUnicorn
    if hic_matrix is not None:
        # enhanced_hic_matrix (NumPy) is the prediction (Pred)
        enhanced_hic_matrix = image_to_hic(hr_image)
        save_hic_matrix(enhanced_hic_matrix, output_hic_path, format='txt')

        # --- METRICS CALCULATION (Image Restoration Metrics) ---
        print("\n[INFO] Calculating Image Restoration and Map-Level Correlation Metrics...")
        
        # Convert NumPy matrices to (1, 1, H, W) PyTorch Tensors for the metrics module
        gt_tensor = numpy_to_tensor4d(hic_matrix)
        pred_tensor = numpy_to_tensor4d(enhanced_hic_matrix)
        
        # Calculate all metrics (MSE, PSNR, SSIM, Pearson, Spearman) and store them
        map_metrics = evaluate_all(gt_tensor, pred_tensor)
        
        if map_metrics:
            # Save metrics to a file 
            map_metrics_path = os.path.join(scores_folder, f"{input_filename}_map_metrics.txt")
            with open(map_metrics_path, 'w') as f:
                f.write("Metric\tValue\n")
                for k, v in map_metrics.items():
                    f.write(f"{k}\t{v:.6f}\n")
            print(f"[INFO] Map enhancement metrics saved to {map_metrics_path}")
        
        # Proceed with file conversion for 3DUnicorn
        convert_dense_to_tuple_debug(output_hic_path, tuple_output_path)

    # 4. Run 3D Structure Reconstruction (Captures the best correlations)
    best_3d_pearson, best_3d_spearman = main_3DUnicorn(params_file, tuple_output_path)


    # 5. Print Final Results in the requested order (3D Structure first, then Map)
    print("\n" + "="*60)
    print("                 FINAL PIPELINE RESULTS")
    print("="*60)

  
    # Print Map Metrics (GT vs. Enhanced)
    if map_metrics:
        print("\n--- Map Enhancement Metrics (GT vs. Enhanced) ---")
        for k, v in map_metrics.items():
            print(f"  {k:<8}: {v:.6f}")

          # Print 3D Metrics (Best results from all structures)
    print(f"3D Structure Pearson Correlation: {best_3d_pearson:.6f}")
    print(f"3D Structure Spearman Correlation: {best_3d_spearman:.6f}")
        
    print("---------------------------------------------------")
    
    print("\n[INFO] Pipeline execution finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Genome Structure Reconstruction Pipeline")
    parser.add_argument('--parameters', required=True, help="Path to parameters.txt file containing all input paths and hyperparameters.")
    args = parser.parse_args()
    
    print("-" * 50)
    print(f"Starting Multimodal 3DUnicorn Pipeline with parameters: {args.parameters}")
    print("-" * 50)
    
    try:
        main_pipeline(args.parameters)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline stopped: {e}")
        import traceback
        traceback.print_exc()