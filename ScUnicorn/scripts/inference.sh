#!/bin/bash

# Bash Script for ScUnicorn Inference
# Runs generate_hr.py with support for both .pth and .npz model formats

# Define paths
MODEL_PATH="./../trained_model/scunicorn_epoch_10.npz"  # Change to .pth if needed
DATA_PATH="./../example_data/processed/input_patches/data_chr19_64.npy"
OUTPUT_PATH="./../ScUnicorn_prediction/scunicorn_hr.npy"

# Create output directory
mkdir -p $(dirname $OUTPUT_PATH)

# Run inference script
python3 generate_hr.py --model_path $MODEL_PATH \
                       --data_path $DATA_PATH \
                       --output_path $OUTPUT_PATH

# Completion message
echo "Inference completed. HR Hi-C data saved to $OUTPUT_PATH."
