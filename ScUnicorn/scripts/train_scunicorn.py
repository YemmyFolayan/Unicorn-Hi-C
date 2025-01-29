# scripts/train_scunicorn.py
"""
Training Script for ScUnicorn
Trains the ScUnicorn model with CLI-based argument handling.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.scunicorn import ScUnicorn
from utils.data_loader import HiCDataset
from utils.logger import get_logger

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train the ScUnicorn model.")
    parser.add_argument("--epoch", type=int, required=True, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--data_train_path", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume training from.")
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs to use for training.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to use.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading.")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"], help="Loss function (mse or l1).")
    parser.add_argument("--early_stoppage_epochs", type=int, default=5, help="Number of epochs for early stopping.")
    parser.add_argument("--early_stoppage_start", type=int, default=100, help="Epoch to start applying early stopping.")
    return parser.parse_args()

def train_model(args):
    """Train ScUnicorn model."""
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() and args.num_gpu > 0 else "cpu")
    
    # Logger setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, "train_scunicorn")
    logger.info("Starting training...")

    # Load dataset
    dataset = HiCDataset([args.data_train_path], [args.data_train_path])  # Dummy pairing
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize model
    model = ScUnicorn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()

    # Resume training from checkpoint
    if args.resume_epoch > 0:
        checkpoint_path = f"checkpoints/scunicorn_epoch_{args.resume_epoch}.pth"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Resumed training from epoch {args.resume_epoch}")
    
    # Training loop
    for epoch in range(args.resume_epoch, args.epoch):
        model.train()
        epoch_loss = 0.0
        
        for lr_data, hr_data in dataloader:
            lr_data, hr_data = lr_data.to(device), hr_data.to(device)
            optimizer.zero_grad()
            output = model(lr_data)
            loss = criterion(output, hr_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        logger.info(f"Epoch [{epoch+1}/{args.epoch}], Loss: {epoch_loss/len(dataloader):.6f}")
        torch.save(model.state_dict(), f"checkpoints/scunicorn_epoch_{epoch+1}.pth")
    
    logger.info("Training completed.")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
