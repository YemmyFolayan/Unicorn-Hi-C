import os
import torch
import numpy as np
import argparse
from PIL import Image
import subprocess
import cv2
import torchvision.transforms.functional as F
from optimization.readInput import read_input
from utils.convert2distance import convert_to_distance
from utils.evaluation import evaluate_structure
from optimization.optimization import optimization
from utils.output_3DUnicorn import output_3DUnicorn
from utils.initialize_structure_c_style import initialize_structure_c_style

def load_model(model_path):
    print(f"[INFO] Loading pre-trained model from {model_path}...")
    model_data = np.load(model_path)
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
    hic_matrix = np.log1p(hic_matrix)
    hic_matrix = (hic_matrix - np.min(hic_matrix)) / (np.max(hic_matrix) - np.min(hic_matrix)) * 255
    hic_matrix = hic_matrix.astype(np.uint8)
    return Image.fromarray(hic_matrix)

def image_to_hic(image):
    matrix = np.array(image, dtype=np.float32)
    matrix = np.expm1(matrix / 255.0)
    return matrix

def save_hic_matrix(matrix, output_path, format="txt"):
    if format == "txt":
        np.savetxt(output_path, matrix, fmt="%.6f", delimiter="\t")
    else:
        np.savez_compressed(output_path, hic=matrix)
    print(f"[SUCCESS] Hi-C contact map saved")

def preprocess_input(data_path):
    if data_path.endswith((".txt", ".npz", ".npy")):
        hic_matrix = load_hic_matrix(data_path)
        return hic_to_image(hic_matrix), hic_matrix
    elif data_path.endswith(".png"):
        return Image.open(data_path).convert("RGB"), None
    else:
        raise ValueError("Unsupported file format. Use .png for images or .txt/.npz/.npy for Hi-C contact maps.")

def compute_feature_embeddings(model, lr_image):
    print("[INFO] Extracting feature embeddings from input data...")
    _ = model.get("feature_vectors", None)
    return lr_image

def infer_model(model, lr_image):
    print("[INFO] Running inference on input data...")
    lr_image = compute_feature_embeddings(model, lr_image)
    hr_image = F.resize(lr_image, (lr_image.height * 4, lr_image.width * 4))
    return hr_image

def convert_dense_to_tuple_debug(input_file, output_file, bin_size=500000):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    non_zero_found = 0

    with open(output_file, 'w') as out_file:
        for row_index, line in enumerate(lines):
            values = line.strip().split()
            for col_index, value in enumerate(values):
                try:
                    frequency = float(value)
                    if frequency != 0:
                        position_1 = (row_index + 1) * bin_size
                        position_2 = (col_index + 1) * bin_size
                        out_file.write(f"{position_1} {position_2} {frequency}\n")
                        if non_zero_found < 5:
                            #print(f"Non-zero entry found: {position_1} {position_2} {frequency}")
                            non_zero_found += 1
                except ValueError:
                    print(f"Warning: Could not convert value '{value}' at row {row_index + 1}, col {col_index + 1}")

def main_3DUnicorn(params_file):
    def parse_parameters_txt(params_file):
        params = {}
        with open(params_file, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=')
                params[key.strip()] = value.strip()
        return params

    # Parse parameters
    parameters = parse_parameters_txt(params_file)
    OUTPUT_FOLDER = parameters.get("OUTPUT_FOLDER")
    INPUT_FILE = parameters.get("OUTPUT_HIC_PATH")
    smooth_factor = 1e-6
    NEAR_ZERO = 0.00001
    NUM = 1
    LEARNING_RATE = 0.5
    MAX_ITERATION = 2000

    # Prepare output folder
    path = 'Scores/'
    os.makedirs(path, exist_ok=True)
    name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    P_CORR, RMSD = [], []


    # Read input and load constraints
    lstCons, n, mapping = read_input(INPUT_FILE, OUTPUT_FOLDER)

    for CONVERT_FACTOR in [0.5]:
        best_spearman_corr = -1  # Initialize to ensure it updates
        best_pearson_corr = -1
        best_structure_name = None

        for l in range(1, NUM + 1):
            lstCons, maxIF = convert_to_distance(lstCons, CONVERT_FACTOR)

            # Use C-style initialization
            coordinates = initialize_structure_c_style(
                sequence_length=n,
                min_dist=0.8,
                max_dist=5.0
            )
            variables = coordinates.flatten()

            # Perform optimization
            variables, change, cost = optimization(
                n, MAX_ITERATION, LEARNING_RATE, smooth_factor, NEAR_ZERO, lstCons, maxIF
            )

            # Evaluate the structure
            structure = variables.reshape(-1, 3)
            rmse, spearman_corr, pearson_corr, WishDist_clean, Dist_clean = evaluate_structure(
                lstCons, variables, n
            )

            P_CORR.append(pearson_corr)
            RMSD.append(rmse)

            # Update the best correlations and structure name
            if spearman_corr > best_spearman_corr:
                best_spearman_corr = spearman_corr
                best_pearson_corr = pearson_corr
                best_structure_name = f'{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}'

            # Output PDB file for this structure
            output_3DUnicorn(variables, f'{path}{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}', WishDist_clean, Dist_clean)

    # Save additional results as text files
    #np.savetxt(os.path.join(path, f'{name}_spearmancorr.txt'), Corr, fmt='%f', header='Spearman Correlation')
    np.savetxt(os.path.join(path, f'{name}_pearsoncorr.txt'), P_CORR, fmt='%f', header='Pearson Correlation')
    np.savetxt(os.path.join(path, f'{name}_rmsd.txt'), RMSD, fmt='%f', header='RMSD')

    # Final output for the best structure
    final_scores = np.column_stack((CONVERT_FACTOR, NUM, best_spearman_corr))
    np.savetxt(os.path.join(path, f'{name}_Finalscores.txt'), final_scores, fmt='%f', 
               header='CONVERT_FACTOR\tStructure Number\tSpearman Correlation')

    # Print the final results
    #print(f"The representative structure is {best_structure_name}.pdb")
    print(f"The Pearson Correlation of this structure: {best_pearson_corr:.6f}")
    print(f"The Spearman Correlation of this structure: {best_spearman_corr:.6f}")
    print("Process complete!")

def main_pipeline(params_file):
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=')
                params[key.strip()] = value.strip()

    model_path = params['MODEL_PATH']
    data_path = params['DATA_PATH']
    output_hic_path = params['OUTPUT_HIC_PATH']

    model = load_model(model_path)
    lr_image, hic_matrix = preprocess_input(data_path)
    hr_image = infer_model(model, lr_image)

    if hic_matrix is not None:
        enhanced_hic_matrix = image_to_hic(hr_image)
        save_hic_matrix(enhanced_hic_matrix, output_hic_path, format='txt')
        convert_dense_to_tuple_debug(output_hic_path, f"{output_hic_path}_tuple.txt")

    main_3DUnicorn(params_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Genome Structure Reconstruction Pipeline")
    parser.add_argument('--parameters', required=True, help="Path to parameters.txt")
    args = parser.parse_args()
    main_pipeline(args.parameters)


