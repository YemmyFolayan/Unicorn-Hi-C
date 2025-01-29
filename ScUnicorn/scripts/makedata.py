# scripts/makedata.py
"""
Data Processing Script for ScUnicorn
Assumes input data is already in .npy format (processed from raw text or images).
Focuses on patch extraction and organization.
"""
import os
import argparse
import numpy as np

def process_hic_patches(input_dir, output_folder, sub_mat_n, chromosomes):
    """
    Processes Hi-C data into patches for training.

    Parameters:
    - input_dir (str): Path to processed Hi-C data (.npy files).
    - output_folder (str): Path to store train-ready patches.
    - sub_mat_n (int): Patch size.
    - chromosomes (list): List of chromosomes to process.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "input_patches"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train_patches"), exist_ok=True)
    
    for chrom in chromosomes:
        file_path = os.path.join(input_dir, f"{chrom}.npy")
        if not os.path.exists(file_path):a
            print(f"Warning: {file_path} not found, skipping.")
            continue

        data = np.load(file_path)
        patches = []

        for i in range(0, data.shape[0], sub_mat_n):
            for j in range(0, data.shape[1], sub_mat_n):
                if i + sub_mat_n <= data.shape[0] and j + sub_mat_n <= data.shape[1]:
                    patches.append(data[i:i+sub_mat_n, j:j+sub_mat_n])

        patches = np.array(patches)
        save_path = os.path.join(output_folder, f"input_patches/data_{chrom}_{sub_mat_n}.npy")
        np.save(save_path, patches)
        print(f"Saved {len(patches)} patches for {chrom} in {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Hi-C data into trainable patches.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to preprocessed Hi-C data.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save processed patches.")
    parser.add_argument("--sub_mat_n", type=int, default=64, help="Patch size for processing.")
    parser.add_argument("--chromosomes", nargs='+', required=True, help="List of chromosomes to process.")
    
    args = parser.parse_args()
    process_hic_patches(args.input_dir, args.output_folder, args.sub_mat_n, args.chromosomes)
