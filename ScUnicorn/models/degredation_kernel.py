"""
Dynamic Degradation Kernel Learning Module
This module defines the degradation kernel, responsible for transforming HR Hi-C maps into degraded LR versions.
Includes a random downsampling step for initial input data with configurable parameters.
"""
import torch
import torch.nn as nn
import random

class DegradationKernel(nn.Module):
    def __init__(self, kernel_size=3, scale_factor=2):
        """
        Initialize the Degradation Kernel.

        Parameters:
        - kernel_size (int): Size of the convolution kernel.
        - scale_factor (float): Factor by which to downsample the input.
        """
        super(DegradationKernel, self).__init__()
        self.kernel = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.scale_factor = scale_factor

    def random_downsample(self, hr_data):
        """
        Randomly downsample the HR data by the specified scale factor.

        Parameters:
        - hr_data (torch.Tensor): High-resolution input tensor of shape (batch, channel, height, width).

        Returns:
        - torch.Tensor: Randomly downsampled tensor.
        """
        _, _, height, width = hr_data.size()
        downsampled_height = int(height / self.scale_factor)
        downsampled_width = int(width / self.scale_factor)

        # Generate random indices for downsampling
        indices_h = sorted(random.sample(range(height), downsampled_height))
        indices_w = sorted(random.sample(range(width), downsampled_width))

        return hr_data[:, :, indices_h, :][:, :, :, indices_w]

    def forward(self, hr_data):
        """
        Forward pass for the Degradation Kernel.

        Parameters:
        - hr_data (torch.Tensor): High-resolution input tensor.

        Returns:
        - torch.Tensor: Degraded low-resolution tensor.
        """
        # Randomly downsample the input data
        downsampled_data = self.random_downsample(hr_data)
        
        # Apply the degradation kernel
        lr_data = self.kernel(downsampled_data)
        return lr_data

if __name__ == "__main__":
    # Test the DegradationKernel module
    print("Testing DegradationKernel module...")

    # Create example high-resolution input data
    hr_input = torch.randn(1, 1, 128, 128)  # Batch size = 1, Channels = 1, Height = 128, Width = 128

    # Initialize the Degradation Kernel
    degradation_kernel = DegradationKernel(kernel_size=3, scale_factor=2)

    # Generate low-resolution output
    lr_output = degradation_kernel(hr_input)

    # Print input and output shapes for verification
    print("HR Input Shape:", hr_input.shape)
    print("LR Output Shape:", lr_output.shape)

    # Ensure output dimensions match expected downsampling
    expected_height = int(hr_input.size(2) / 2)
    expected_width = int(hr_input.size(3) / 2)
    assert lr_output.size(2) == expected_height, "Height mismatch after downsampling."
    assert lr_output.size(3) == expected_width, "Width mismatch after downsampling."

    print("DegradationKernel test passed.")
