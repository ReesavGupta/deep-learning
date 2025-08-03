"""
Single-Layer Custom ANN Implementation
Assignment: Build a custom single-layer artificial neural network using only basic PyTorch operations

Model Architecture:
- Y = w^T * x + b
- Activation: Sigmoid
- Loss: Binary Cross Entropy
- Optimizer: Manual gradient descent

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

class SingleLayerANN:
    """
    Custom Single-Layer Artificial Neural Network implementation
    using only basic PyTorch operations (no torch.nn)
    """
    
    def __init__(self, input_size, learning_rate=0.01, device='cpu'):
        """
        Initialize the single-layer ANN
        
        Args:
            input_size (int): Number of input features
            learning_rate (float): Learning rate for gradient descent
            device (str): Device to run computations on ('cpu' or 'cuda')
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Initialize weights and bias
        # Using Xavier/Glorot initialization
        self.weights = torch.randn(input_size, 1, device=self.device, requires_grad=True) * np.sqrt(2.0 / input_size)
        self.bias = torch.zeros(1, device=self.device, requires_grad=True)
        
        print(f"Initialized Single-Layer ANN:")
        print(f"  Input size: {input_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Weights shape: {self.weights.shape}")
        print(f"  Bias shape: {self.bias.shape}")
    
    def sigmoid(self, z):
        """
        Sigmoid activation function
        
        Args:
            z (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Sigmoid output
        """
        # Clamp z to prevent overflow in exp
        z = torch.clamp(z, -500, 500)
        return 1.0 / (1.0 + torch.exp(-z))
    
    def forward(self, X):
        """
        Forward pass: Y = sigmoid(w^T * x + b)
        
        Args:
            X (torch.Tensor): Input features [batch_size, input_size]
            
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        # Linear transformation: z = X @ w + b
        z = X @ self.weights + self.bias
        
        # Apply sigmoid activation
        y_pred = self.sigmoid(z)
        
        return y_pred
    
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
        Single training step with manual gradient computation
        
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
        
        # Manual backward pass (compute gradients)
        batch_size = X.shape[0]
        
        # Gradient of loss w.r.t. predictions
        # d_loss/d_y_pred = -(y/y_pred - (1-y)/(1-y_pred))
        epsilon = 1e-15
        y_pred_clamped = torch.clamp(y_pred, epsilon, 1 - epsilon)
        d_loss_d_pred = -(y / y_pred_clamped - (1 - y) / (1 - y_pred_clamped)) / batch_size
        
        # Gradient of sigmoid: d_sigmoid/d_z = sigmoid * (1 - sigmoid)
        d_sigmoid_d_z = y_pred * (1 - y_pred)
        
        # Chain rule: d_loss/d_z = d_loss/d_pred * d_pred/d_z
        d_loss_d_z = d_loss_d_pred * d_sigmoid_d_z
        
        # Gradients w.r.t. weights and bias
        # d_loss/d_w = X^T @ d_loss/d_z
        d_loss_d_w = X.T @ d_loss_d_z
        
        # d_loss/d_b = sum(d_loss/d_z)
        d_loss_d_b = d_loss_d_z.sum(dim=0)
        
        # Manual parameter update (gradient descent)
        with torch.no_grad():
            self.weights -= self.learning_rate * d_loss_d_w
            self.bias -= self.learning_rate * d_loss_d_b
        
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

def generate_dataset():
    """
    Generate binary classification dataset
    """
    print("Generating binary classification dataset...")
    
    # Generate dataset using sklearn
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
    print(f"Feature columns: {df.columns[:-1].tolist()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return X, y

def load_dataset(csv_path='binary_data.csv'):
    """
    Load dataset from CSV file
    
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
    else:
        print(f"CSV file {csv_path} not found. Generating new dataset...")
        X, y = generate_dataset()
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for training
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: Prepared train/test sets as PyTorch tensors
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
    Train the single-layer ANN
    
    Args:
        model: SingleLayerANN instance
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs (int): Number of training epochs
    """
    print(f"\nTraining Single-Layer ANN for {epochs} epochs...")
    print("=" * 50)
    
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
    
    print("=" * 50)
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
    Visualize the dataset and training progress
    """
    try:
        import matplotlib.pyplot as plt

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
        ax2.set_title('Training Loss Over Time')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Visualization saved as 'training_results.png'")

    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

def main():
    """
    Main function to run the single-layer ANN training
    """
    print("Single-Layer Custom ANN Implementation")
    print("=" * 50)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load or generate dataset
    X, y = load_dataset()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SingleLayerANN(
        input_size=2,
        learning_rate=0.1,  # Higher learning rate for faster convergence
        device=device
    )

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
    print(f"Accuracy on test set = {test_accs[-1]:.1f}%")

    # Visualize results
    visualize_results(X, y, model, train_losses)

    # Show model parameters
    print(f"\n" + "=" * 50)
    print("FINAL MODEL PARAMETERS:")
    print("=" * 50)
    print(f"Weights: {model.weights.detach().cpu().numpy().flatten()}")
    print(f"Bias: {model.bias.detach().cpu().numpy().flatten()}")

if __name__ == "__main__":
    main()
