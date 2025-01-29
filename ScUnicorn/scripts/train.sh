#!/bin/bash

# Bash Script for Training ScUnicorn
# Runs train_scunicorn.py with predefined settings

# Define paths
DATA_TRAIN_PATH="./../example_data/processed/train_patches/"
CHECKPOINT_DIR="./../checkpoints/"
LOG_DIR="./../logs/"

# Create necessary directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Run training script
torchrun --nproc_per_node=1 --master_port=4321 train_scunicorn.py \
  --epoch 10 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --data_train_path $DATA_TRAIN_PATH \
  --resume_epoch 0 \
  --num_gpu 1 \
  --device_id 0 \
  --num_workers 1 \
  --loss mse \
  --early_stoppage_epochs 5 \
  --early_stoppage_start 100

# Completion message
echo "Training completed. Check logs in $LOG_DIR and model checkpoints in $CHECKPOINT_DIR."
