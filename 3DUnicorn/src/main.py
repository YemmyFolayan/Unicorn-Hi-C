import os
import numpy as np
import argparse
from optimization.readInput import read_input
from utils.convert2distance import convert_to_distance
from utils.evaluation import evaluate_structure
from optimization.optimization import optimization
from utils.output_3DMax import output_3DMax
from utils.initialize_structure_c_style import initialize_structure_c_style

def main_3DMax(params_file):
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
    INPUT_FILE = parameters.get("INPUT_FILE")
    smooth_factor = 1e-6
    NEAR_ZERO = 0.00001
    NUM = 1
    LEARNING_RATE = 0.5
    MAX_ITERATION = 2000

    # Prepare output folder
    path = 'Scores/'
    os.makedirs(path, exist_ok=True)
    name = os.path.splitext(os.path.basename(INPUT_FILE))[0]

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

            # Update the best correlations and structure name
            if spearman_corr > best_spearman_corr:
                best_spearman_corr = spearman_corr
                best_pearson_corr = pearson_corr
                best_structure_name = f'{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}'

            # Output PDB file for this structure
            output_3DMax(variables, f'{path}{name}_CONVERT_FACTOR={CONVERT_FACTOR}_N={l}', WishDist_clean, Dist_clean)

    # Print the final results
    #print(f"The representative structure is {best_structure_name}.pdb")
    print(f"The Pearson Correlation of this structure: {best_pearson_corr:.6f}")
    print(f"The Spearman Correlation of this structure: {best_spearman_corr:.6f}")
    print("Process complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3DMax Algorithm")
    parser.add_argument('--parameters', required=True, help="Path to parameters.txt")
    args = parser.parse_args()
    main_3DMax(args.parameters)