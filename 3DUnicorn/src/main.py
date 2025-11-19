import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
from optimization.readInput import read_input
from utils.convert2distance import convert_to_distance
from utils.evaluation import evaluate_structure
from optimization.optimization import optimization
from utils.output_3DUnicorn import output_3DUnicorn
from utils.initialize_structure_c_style import initialize_structure_c_style

def load_model(model_path):
    """Load a pre-trained model from an .npz file."""
    print(f"[INFO] Loading pre-trained model from {model_path}...")
    model_data = np.load(model_path)
    print("[INFO] Model loaded successfully.")
    return model_data  # Placeholder for actual model usage


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
    print(f"[SUCCESS] Hi-C contact map saved to {output_path}")

def preprocess_input(data_path):
    if data_path.endswith((".txt", ".npz", ".npy")):
        hic_matrix = load_hic_matrix(data_path)
        return hic_to_image(hic_matrix), hic_matrix
    elif data_path.endswith(".png"):
        return Image.open(data_path).convert("RGB"), None
    else:
        raise ValueError("Unsupported file format. Use .png for images or .txt/.npz/.npy for Hi-C contact maps.")

def infer_model(model, lr_image):
    print("[INFO] Running inference on input data...")

    print(f"[DEBUG] Input Image Size: {lr_image.width}x{lr_image.height}")

    upscale_factor = 2  # You can change this factor as per your model's training

    # Upscale
    hr_resized = F.resize(lr_image, (lr_image.height * upscale_factor, lr_image.width * upscale_factor))

    # Crop back to original size
    left = (hr_resized.width - lr_image.width) // 2
    top = (hr_resized.height - lr_image.height) // 2
    right = left + lr_image.width
    bottom = top + lr_image.height

    hr_cropped = hr_resized.crop((left, top, right, bottom))

    print(f"[DEBUG] Output Image Size: {hr_cropped.width}x{hr_cropped.height}")

    return hr_cropped

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
                        position_1 = (row_index + 1) * bin_size
                        position_2 = (col_index + 1) * bin_size
                        out_file.write(f"{position_1} {position_2} {frequency}\n")
                except ValueError:
                    print(f"Warning: Could not convert value '{value}' at row {row_index + 1}, col {col_index + 1}")

def main_3DUnicorn(params_file, updated_input_path):
    def parse_parameters_txt(params_file):
        params = {}
        with open(params_file, 'r') as file:
            for line in file:
                if not line.strip() or line.startswith('#'):
                    continue
                key, value = line.split('=')
                params[key.strip()] = value.strip()
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

    lstCons, n, _ = read_input(INPUT_FILE, OUTPUT_FOLDER)

    for CONVERT_FACTOR in [0.6]:
        best_spearman_corr = -1
        best_pearson_corr = -1
        best_structure_name = None

        for l in range(1, 2):  # NUM=1
            lstCons, maxIF = convert_to_distance(lstCons, CONVERT_FACTOR)
            coordinates = initialize_structure_c_style(n, 0.8, 5.0)
            variables = coordinates.flatten()

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

    print(f"[RESULT] Pearson Correlation: {best_pearson_corr:.6f}")
    print(f"[RESULT] Spearman Correlation: {best_spearman_corr:.6f}")
    print("[INFO] Process complete.")

def main_pipeline(params_file):
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=')
                params[key.strip()] = value.strip()

    model_path = params['MODEL_PATH']
    input_file = params['INPUT_HIC_FILE']

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
    lr_image, hic_matrix = preprocess_input(input_file)
    hr_image = infer_model(model, lr_image)

    if hic_matrix is not None:
        enhanced_hic_matrix = image_to_hic(hr_image)
        save_hic_matrix(enhanced_hic_matrix, output_hic_path, format='txt')
        convert_dense_to_tuple_debug(output_hic_path, tuple_output_path)

    main_3DUnicorn(params_file, tuple_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Genome Structure Reconstruction Pipeline")
    parser.add_argument('--parameters', required=True, help="Path to parameters.txt")
    args = parser.parse_args()
    main_pipeline(args.parameters)
