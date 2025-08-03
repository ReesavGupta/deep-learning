"""
PyTorch Fundamentals & Manual Custom ANN
Assignment: Tensor Creation and Operations

This script demonstrates basic PyTorch operations without using torch.nn or torch.nn.Module.
It covers tensor creation, matrix operations, and GPU utilization.

Author: Assignment Solution
Date: 2025-08-03
"""

import torch
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
import os

def check_device():
    """
    Check if CUDA is available and return the appropriate device.
    
    Returns:
        torch.device: The device to use for computations (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    
    return device

def tensor_operations():
    """
    Perform basic tensor operations as specified in the assignment.
    """
    print("\n" + "="*50)
    print("TENSOR CREATION AND OPERATIONS")
    print("="*50)
    
    # Get the device (CPU or GPU)
    device = check_device()
    
    # Create tensors A and B as specified
    print("\n1. Creating tensors A and B:")
    A = torch.randn(3, 2)
    B = torch.randn(2, 3)
    
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    
    # Matrix multiplication: C = A @ B
    print("\n2. Matrix multiplication (C = A @ B):")
    C = A @ B
    print(f"C: {C}")
    print(f"C shape: {C.shape}")
    
    # Element-wise addition: D = A + torch.ones_like(A)
    print("\n3. Element-wise addition (D = A + ones_like(A)):")
    D = A + torch.ones_like(A)
    print(f"D: {D}")
    print(f"D shape: {D.shape}")
    
    # Move tensors to GPU if available
    print("\n4. Moving tensors to device:")
    A_device = A.to(device)
    B_device = B.to(device)
    C_device = C.to(device)
    D_device = D.to(device)
    
    print(f"A is on device: {A_device.device}")
    print(f"B is on device: {B_device.device}")
    print(f"C is on device: {C_device.device}")
    print(f"D is on device: {D_device.device}")
    
    # Perform operations on GPU/device
    print("\n5. Operations on device:")
    C_device_computed = A_device @ B_device
    D_device_computed = A_device + torch.ones_like(A_device)
    
    print(f"C (computed on {device}): {C_device_computed}")
    print(f"D (computed on {device}): {D_device_computed}")
    
    return A_device, B_device, C_device, D_device

def additional_tensor_operations():
    """
    Demonstrate additional tensor operations for comprehensive understanding.
    """
    print("\n" + "="*50)
    print("ADDITIONAL TENSOR OPERATIONS")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create various types of tensors
    print("\n1. Different tensor creation methods:")
    
    # Zeros and ones
    zeros_tensor = torch.zeros(2, 3).to(device)
    ones_tensor = torch.ones(2, 3).to(device)
    identity_tensor = torch.eye(3).to(device)
    
    print(f"Zeros tensor: {zeros_tensor}")
    print(f"Ones tensor: {ones_tensor}")
    print(f"Identity tensor: {identity_tensor}")
    
    # Random tensors with different distributions
    uniform_tensor = torch.rand(2, 3).to(device)  # Uniform [0, 1)
    normal_tensor = torch.randn(2, 3).to(device)  # Normal distribution
    
    print(f"Uniform tensor: {uniform_tensor}")
    print(f"Normal tensor: {normal_tensor}")
    
    # Tensor operations
    print("\n2. Tensor operations:")
    
    # Element-wise operations
    sum_tensor = uniform_tensor + normal_tensor
    product_tensor = uniform_tensor * normal_tensor
    
    print(f"Element-wise sum: {sum_tensor}")
    print(f"Element-wise product: {product_tensor}")
    
    # Reduction operations
    print(f"Sum of all elements: {sum_tensor.sum()}")
    print(f"Mean of all elements: {sum_tensor.mean()}")
    print(f"Max element: {sum_tensor.max()}")
    print(f"Min element: {sum_tensor.min()}")
    
    # Reshaping
    reshaped = sum_tensor.view(-1)  # Flatten
    print(f"Flattened tensor: {reshaped}")
    print(f"Reshaped back to 3x2: {reshaped.view(3, 2)}")

def generate_sample_dataset():
    """
    Generate a sample classification dataset using sklearn and save to CSV.
    """
    print("\n" + "="*50)
    print("GENERATING SAMPLE DATASET")
    print("="*50)
    
    # Generate classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    csv_path = 'sample_dataset.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset generated and saved to: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns[:-1].tolist()}")
    print(f"Target classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    print(f"\nTensor shapes:")
    print(f"X_tensor: {X_tensor.shape} on {X_tensor.device}")
    print(f"y_tensor: {y_tensor.shape} on {y_tensor.device}")
    
    return X_tensor, y_tensor

def main():
    """
    Main function to run all tensor operations and demonstrations.
    """
    print("PyTorch Fundamentals & Manual Custom ANN")
    print("Assignment: Tensor Creation and Operations")
    print("Using only basic PyTorch operations (no torch.nn)")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Perform basic tensor operations as specified
    A, B, C, D = tensor_operations()
    
    # Additional tensor operations for learning
    additional_tensor_operations()
    
    # Generate sample dataset
    X_tensor, y_tensor = generate_sample_dataset()
    
    print("\n" + "="*50)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nSummary of operations performed:")
    print("✓ Created tensors A (3x2) and B (2x3)")
    print("✓ Computed matrix multiplication C = A @ B")
    print("✓ Computed element-wise addition D = A + ones_like(A)")
    print("✓ Moved tensors to GPU (if available)")
    print("✓ Generated sample dataset using sklearn")
    print("✓ Demonstrated additional tensor operations")

if __name__ == "__main__":
    main()
