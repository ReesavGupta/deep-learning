"""
Two-Layer Custom ANN Implementation (2-4-1 Architecture)
Assignment: Build a custom two-layer neural network with one hidden layer

Model Architecture:
- Input Layer: 2 neurons
- Hidden Layer: 4 neurons (ReLU activation)
- Output Layer: 1 neuron (Sigmoid activation)
- Loss: Binary Cross Entropy
- Optimizer: Manual weight update using gradients with .backward()

Author: Assignment Solution
Date: 2025-08-03
"""

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class TwoLayerANN:
    """
    Custom Two-Layer Artificial Neural Network implementation (2-4-1 architecture)
    using basic PyTorch operations with automatic differentiation
    """
    
    def __init__(self, learning_rate=0.01, device='cpu'):
        """
        Initialize the two-layer ANN with 2-4-1 architecture
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            device (str): Device to run computations on ('cpu' or 'cuda')
        """
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Initialize weights and biases as specified in the assignment
        print("Initializing 2-4-1 Neural Network...")
        
        # Layer 1: Input (2) -> Hidden (4)
        self.W1 = torch.randn(2, 4, requires_grad=True, device=self.device)
        self.b1 = torch.zeros(1, 4, requires_grad=True, device=self.device)
        
        # Layer 2: Hidden (4) -> Output (1)
        self.W2 = torch.randn(4, 1, requires_grad=True, device=self.device)
        self.b2 = torch.zeros(1, 1, requires_grad=True, device=self.device)
        
        print(f"Network Architecture: 2-4-1")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"W1 shape: {self.W1.shape}")
        print(f"b1 shape: {self.b1.shape}")
        print(f"W2 shape: {self.W2.shape}")
        print(f"b2 shape: {self.b2.shape}")
        
        # Display initial weights
        print(f"\nInitial weights:")
        print(f"W1:\n{self.W1.detach().cpu().numpy()}")
        print(f"b1: {self.b1.detach().cpu().numpy()}")
        print(f"W2:\n{self.W2.detach().cpu().numpy()}")
        print(f"b2: {self.b2.detach().cpu().numpy()}")
    
    def forward(self, X):
        """
        Forward pass through the network
        
        Forward Pass:
        Z1 = X @ W1 + b1
        A1 = torch.relu(Z1)
        Z2 = A1 @ W2 + b2
        Y_pred = torch.sigmoid(Z2)
        
        Args:
            X (torch.Tensor): Input features [batch_size, 2]
            
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        # Layer 1: Linear transformation + ReLU activation
        Z1 = X @ self.W1 + self.b1
        A1 = torch.relu(Z1)
        
        # Layer 2: Linear transformation + Sigmoid activation
        Z2 = A1 @ self.W2 + self.b2
        Y_pred = torch.sigmoid(Z2)
        
        return Y_pred
    
    def binary_cross_entropy_loss(self, y_pred, y_true):
        """
        Binary Cross Entropy Loss
        
        Args:
            y_pred (torch.Tensor): Predicted probabilities [batch_size, 1]
            y_true (torch.Tensor): True labels [batch_size, 1]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        
        # BCE Loss: -[y*log(y_pred) + (1-y)*log(1-y_pred)]
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss.mean()
    
    def train_step(self, X, y):
        """
        Single training step using automatic differentiation
        
        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): True labels
            
        Returns:
            float: Loss value
        """
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute loss
        loss = self.binary_cross_entropy_loss(y_pred, y)
        
        # Backward pass using automatic differentiation
        loss.backward()
        
        # Manual weight update using gradients
        with torch.no_grad():
            # Update weights and biases
            self.W1 -= self.learning_rate * self.W1.grad
            self.b1 -= self.learning_rate * self.b1.grad
            self.W2 -= self.learning_rate * self.W2.grad
            self.b2 -= self.learning_rate * self.b2.grad
            
            # Zero gradients after update
            self.W1.grad.zero_()
            self.b1.grad.zero_()
            self.W2.grad.zero_()
            self.b2.grad.zero_()
        
        return loss.item()
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Binary predictions (0 or 1)
        """
        with torch.no_grad():
            y_pred_prob = self.forward(X)
            y_pred = (y_pred_prob >= 0.5).float()
        return y_pred
    
    def accuracy(self, X, y):
        """
        Calculate accuracy
        
        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): True labels
            
        Returns:
            float: Accuracy percentage
        """
        y_pred = self.predict(X)
        correct = (y_pred == y).float().sum()
        accuracy = (correct / y.shape[0]) * 100
        return accuracy.item()

def load_dataset(csv_path='binary_data.csv'):
    """
    Load dataset from CSV file (same as Q2)
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        tuple: (X, y) features and labels
    """
    if os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        X = df[['f1', 'f2']].values
        y = df['label'].values
        print(f"Loaded dataset shape: {df.shape}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    else:
        print(f"CSV file {csv_path} not found. Generating new dataset...")
        X, y = generate_dataset()
    
    return X, y

def generate_dataset():
    """
    Generate binary classification dataset (same as Q2)
    """
    print("Generating binary classification dataset...")
    
    X, y = make_classification(
        n_samples=100, 
        n_features=2, 
        n_classes=2, 
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=1
    )
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(X, columns=['f1', 'f2'])
    df['label'] = y
    df.to_csv('binary_data.csv', index=False)
    
    print(f"Dataset saved to: binary_data.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for training (same as Q2)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    print(f"Training set: {X_train_tensor.shape[0]} samples")
    print(f"Test set: {X_test_tensor.shape[0]} samples")
    print(f"Device: {device}")
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def train_model(model, X_train, y_train, X_test, y_test, epochs=50):
    """
    Train the two-layer ANN
    """
    print(f"\nTraining Two-Layer ANN for {epochs} epochs...")
    print("=" * 60)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        # Training step
        loss = model.train_step(X_train, y_train)
        train_losses.append(loss)
        
        # Calculate accuracies
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Print progress
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}: Loss = {loss:.4f}, Train Acc = {train_acc:.1f}%, Test Acc = {test_acc:.1f}%")
    
    print("=" * 60)
    print("Training completed!")
    
    # Final results
    final_train_acc = train_accuracies[-1]
    final_test_acc = test_accuracies[-1]
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {final_train_acc:.1f}%")
    print(f"Test Accuracy: {final_test_acc:.1f}%")
    
    return train_losses, train_accuracies, test_accuracies

def visualize_results(X, y, model, train_losses):
    """
    Visualize the training results
    """
    try:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Dataset visualization
        X_np = X if isinstance(X, np.ndarray) else X.cpu().numpy()
        y_np = y if isinstance(y, np.ndarray) else y.cpu().numpy()
        
        scatter = ax1.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Feature 1 (f1)')
        ax1.set_ylabel('Feature 2 (f2)')
        ax1.set_title('Binary Classification Dataset')
        plt.colorbar(scatter, ax=ax1)
        
        # Plot 2: Training loss
        ax2.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Over Time (2-4-1 Network)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('two_layer_training_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'two_layer_training_results.png'")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

def main():
    """
    Main function to run the two-layer ANN training
    """
    print("Two-Layer Custom ANN Implementation (2-4-1 Architecture)")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset (same as Q2)
    X, y = load_dataset()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TwoLayerANN(learning_rate=0.1, device=device)
    
    # Train model
    train_losses, train_accs, test_accs = train_model(
        model, X_train, y_train, X_test, y_test, epochs=50
    )
    
    # Display sample output format as requested
    print(f"\n" + "=" * 50)
    print("SAMPLE OUTPUT FORMAT (as requested):")
    print("=" * 50)
    print(f"Epoch 1: Loss = {train_losses[0]:.2f}")
    if len(train_losses) >= 30:
        print(f"Epoch 30: Loss = {train_losses[29]:.2f}")
    print(f"Accuracy: {test_accs[-1]:.1f}%")
    
    # Show final model parameters
    print(f"\n" + "=" * 50)
    print("FINAL MODEL PARAMETERS:")
    print("=" * 50)
    print(f"W1 (Input -> Hidden):\n{model.W1.detach().cpu().numpy()}")
    print(f"b1: {model.b1.detach().cpu().numpy()}")
    print(f"W2 (Hidden -> Output):\n{model.W2.detach().cpu().numpy()}")
    print(f"b2: {model.b2.detach().cpu().numpy()}")
    
    # Visualize results
    visualize_results(X, y, model, train_losses)

    # Compare with single-layer performance
    print(f"\n" + "=" * 50)
    print("ARCHITECTURE COMPARISON:")
    print("=" * 50)
    print("Two-Layer Network (2-4-1):")
    print(f"  - Hidden layer with 4 neurons (ReLU)")
    print(f"  - Output layer with 1 neuron (Sigmoid)")
    print(f"  - Total parameters: {2*4 + 4 + 4*1 + 1} = 17")
    print(f"  - Final test accuracy: {test_accs[-1]:.1f}%")
    print(f"  - Final loss: {train_losses[-1]:.4f}")

if __name__ == "__main__":
    main()
