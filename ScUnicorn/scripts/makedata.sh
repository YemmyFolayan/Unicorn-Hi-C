#!/bin/bash

# Bash Script for Data Preprocessing in ScUnicorn
# Runs makedata.py with example dataset

# Define directories
INPUT_DIR="./../example_data/raw/"
OUTPUT_DIR="./../example_data/processed/"
SUB_MAT_N=64
CHROMOSOMES="chr19"

# Create processed folder
mkdir -p $OUTPUT_DIR

# Run makedata.py
python3 makedata.py --input_dir $INPUT_DIR \
                    --output_folder $OUTPUT_DIR \
                    --sub_mat_n $SUB_MAT_N \
                    --chromosomes $CHROMOSOMES

# Copy necessary files for training
mkdir -p $OUTPUT_DIR/train_patches
cp $OUTPUT_DIR/input_patches/data_chr19_${SUB_MAT_N}.npy $OUTPUT_DIR/train_patches/

# Completion message
echo "Data preprocessing completed. Processed files are in: $OUTPUT_DIR"
