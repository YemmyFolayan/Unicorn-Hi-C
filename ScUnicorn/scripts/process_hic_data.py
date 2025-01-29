# scripts/process_hic_data.py
"""
Data Preprocessing Script for ScUnicorn
Handles raw Hi-C interaction text files and image-based Hi-C maps.
"""
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

def process_hic_txt(file_path, output_path):
    """Reads a Hi-C interaction text file and converts it into a numpy array."""
    with open(file_path, 'r') as f:
        data = [list(map(int, line.strip().split())) for line in f]
    hic_matrix = np.array(data)
    np.save(output_path, hic_matrix)
    print(f"Saved processed Hi-C text data as {output_path}")

def process_hic_image(image_path, output_path):
    """Loads a Hi-C contact map from a PNG file and converts it into a grayscale numpy array."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return
    np.save(output_path, img)
    print(f"Saved processed Hi-C image as {output_path}")
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.title("Processed Hi-C Contact Map")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Hi-C data for ScUnicorn.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the raw Hi-C data (TXT or PNG file).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed data.")
    
    args = parser.parse_args()
    
    # Determine file type
    if args.file_path.endswith(".txt"):
        process_hic_txt(args.file_path, args.output_path)
    elif args.file_path.endswith(".png"):
        process_hic_image(args.file_path, args.output_path)
    else:
        print("Unsupported file format. Please provide a .txt or .png file.")
