"""
Training Script for the ScUnicorn Model
This script trains the ScUnicorn model using simulated data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.scunicorn import ScUnicorn

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_synthetic_data(num_samples=1000, hr_size=(128, 128)):
    """
    Generate synthetic high-resolution data for training.

    Parameters:
    - num_samples (int): Number of samples to generate.
    - hr_size (tuple): Dimensions of the high-resolution data.

    Returns:
    - TensorDataset: Synthetic dataset of HR tensors and corresponding dummy targets.
    """
    hr_data = torch.randn(num_samples, 1, *hr_size)  # HR Hi-C data
    dummy_targets = torch.zeros(num_samples, 1, *hr_size)  # Placeholder targets
    return TensorDataset(hr_data, dummy_targets)

def train_scunicorn():
    """
    Train the ScUnicorn model.
    """
    # Initialize the ScUnicorn model
    model = ScUnicorn(kernel_size=3, scale_factor=2, input_dim=128*64, output_dim=512).to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Generate synthetic data
    dataset = generate_synthetic_data()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for hr_data, _ in dataloader:
            hr_data = hr_data.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            hr_restored = model(hr_data)

            # Compute loss
            loss = criterion(hr_restored, hr_data)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Log epoch progress
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.6f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/scunicorn_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    print("Starting training for ScUnicorn...")
    train_scunicorn()
    print("Training completed.")
