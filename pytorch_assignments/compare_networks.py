"""
Network Comparison: Single-Layer vs Two-Layer ANN
Compare the performance of single-layer (2-1) vs two-layer (2-4-1) neural networks

Author: Assignment Solution
Date: 2025-08-03
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def load_dataset(csv_path='binary_data.csv'):
    """Load the binary classification dataset"""
    if os.path.exists(csv_path):
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        X = df[['f1', 'f2']].values
        y = df['label'].values
        return X, y
    else:
        print(f"Dataset not found. Please run single_layer_ann.py first.")
        return None, None

def prepare_data(X, y):
    """Prepare data for training"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, device

class SingleLayerANN:
    """Single-layer neural network (2-1)"""
    
    def __init__(self, learning_rate=0.1, device='cpu'):
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.weights = torch.randn(2, 1, device=self.device, requires_grad=True)
        self.bias = torch.zeros(1, device=self.device, requires_grad=True)

        # Initialize weights with smaller values
        with torch.no_grad():
            self.weights *= 0.5
    
    def forward(self, X):
        z = X @ self.weights + self.bias
        return torch.sigmoid(z)
    
    def train_step(self, X, y):
        # Zero gradients first
        if self.weights.grad is not None:
            self.weights.grad.zero_()
        if self.bias.grad is not None:
            self.bias.grad.zero_()

        y_pred = self.forward(X)
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()

        loss.backward()

        with torch.no_grad():
            self.weights -= self.learning_rate * self.weights.grad
            self.bias -= self.learning_rate * self.bias.grad

        return loss.item()
    
    def accuracy(self, X, y):
        with torch.no_grad():
            y_pred = self.forward(X)
            y_pred_binary = (y_pred >= 0.5).float()
            correct = (y_pred_binary == y).float().sum()
            return (correct / y.shape[0] * 100).item()

class TwoLayerANN:
    """Two-layer neural network (2-4-1)"""
    
    def __init__(self, learning_rate=0.1, device='cpu'):
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        self.W1 = torch.randn(2, 4, requires_grad=True, device=self.device)
        self.b1 = torch.zeros(1, 4, requires_grad=True, device=self.device)
        self.W2 = torch.randn(4, 1, requires_grad=True, device=self.device)
        self.b2 = torch.zeros(1, 1, requires_grad=True, device=self.device)
    
    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = torch.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return torch.sigmoid(Z2)
    
    def train_step(self, X, y):
        # Zero gradients first
        if self.W1.grad is not None:
            self.W1.grad.zero_()
        if self.b1.grad is not None:
            self.b1.grad.zero_()
        if self.W2.grad is not None:
            self.W2.grad.zero_()
        if self.b2.grad is not None:
            self.b2.grad.zero_()

        y_pred = self.forward(X)
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()

        loss.backward()

        with torch.no_grad():
            self.W1 -= self.learning_rate * self.W1.grad
            self.b1 -= self.learning_rate * self.b1.grad
            self.W2 -= self.learning_rate * self.W2.grad
            self.b2 -= self.learning_rate * self.b2.grad

        return loss.item()
    
    def accuracy(self, X, y):
        with torch.no_grad():
            y_pred = self.forward(X)
            y_pred_binary = (y_pred >= 0.5).float()
            correct = (y_pred_binary == y).float().sum()
            return (correct / y.shape[0] * 100).item()

def train_and_compare(X_train, X_test, y_train, y_test, device, epochs=50):
    """Train both networks and compare performance"""
    
    print("Initializing networks...")
    single_layer = SingleLayerANN(learning_rate=0.1, device=device)
    two_layer = TwoLayerANN(learning_rate=0.1, device=device)
    
    # Storage for metrics
    single_losses = []
    single_train_accs = []
    single_test_accs = []
    
    two_losses = []
    two_train_accs = []
    two_test_accs = []
    
    print(f"\nTraining both networks for {epochs} epochs...")
    print("=" * 70)
    print(f"{'Epoch':<6} {'Single Loss':<12} {'Single Acc':<12} {'Two Loss':<12} {'Two Acc':<12}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        # Train single-layer network
        single_loss = single_layer.train_step(X_train, y_train)
        single_train_acc = single_layer.accuracy(X_train, y_train)
        single_test_acc = single_layer.accuracy(X_test, y_test)
        
        single_losses.append(single_loss)
        single_train_accs.append(single_train_acc)
        single_test_accs.append(single_test_acc)
        
        # Train two-layer network
        two_loss = two_layer.train_step(X_train, y_train)
        two_train_acc = two_layer.accuracy(X_train, y_train)
        two_test_acc = two_layer.accuracy(X_test, y_test)
        
        two_losses.append(two_loss)
        two_train_accs.append(two_train_acc)
        two_test_accs.append(two_test_acc)
        
        # Print progress
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{epoch:<6} {single_loss:<12.4f} {single_test_acc:<12.1f} {two_loss:<12.4f} {two_test_acc:<12.1f}")
    
    print("=" * 70)
    
    return {
        'single': {
            'losses': single_losses,
            'train_accs': single_train_accs,
            'test_accs': single_test_accs,
            'model': single_layer
        },
        'two': {
            'losses': two_losses,
            'train_accs': two_train_accs,
            'test_accs': two_test_accs,
            'model': two_layer
        }
    }

def visualize_comparison(results, X, y):
    """Create comparison visualizations"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original dataset
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Feature 1 (f1)')
    ax1.set_ylabel('Feature 2 (f2)')
    ax1.set_title('Binary Classification Dataset')
    plt.colorbar(scatter, ax=ax1)
    
    # Plot 2: Loss comparison
    epochs = range(1, len(results['single']['losses']) + 1)
    ax2.plot(epochs, results['single']['losses'], 'b-', label='Single Layer (2-1)', linewidth=2)
    ax2.plot(epochs, results['two']['losses'], 'r-', label='Two Layer (2-4-1)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test accuracy comparison
    ax3.plot(epochs, results['single']['test_accs'], 'b-', label='Single Layer (2-1)', linewidth=2)
    ax3.plot(epochs, results['two']['test_accs'], 'r-', label='Two Layer (2-4-1)', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Test Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final comparison bar chart
    categories = ['Final Loss', 'Test Accuracy']
    single_values = [results['single']['losses'][-1], results['single']['test_accs'][-1]]
    two_values = [results['two']['losses'][-1], results['two']['test_accs'][-1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Normalize values for better visualization
    single_norm = [single_values[0], single_values[1]/100]  # Normalize accuracy to 0-1
    two_norm = [two_values[0], two_values[1]/100]
    
    ax4.bar(x - width/2, single_norm, width, label='Single Layer (2-1)', alpha=0.8)
    ax4.bar(x + width/2, two_norm, width, label='Two Layer (2-4-1)', alpha=0.8)
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Normalized Values')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    
    # Add value labels on bars
    for i, (s, t) in enumerate(zip(single_norm, two_norm)):
        ax4.text(i - width/2, s + 0.01, f'{single_values[i]:.3f}' if i == 0 else f'{single_values[i]:.1f}%', 
                ha='center', va='bottom')
        ax4.text(i + width/2, t + 0.01, f'{two_values[i]:.3f}' if i == 0 else f'{two_values[i]:.1f}%', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('network_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison visualization saved as 'network_comparison.png'")

def main():
    """Main comparison function"""
    print("Neural Network Architecture Comparison")
    print("Single-Layer (2-1) vs Two-Layer (2-4-1)")
    print("=" * 60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    X, y = load_dataset()
    if X is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, device = prepare_data(X, y)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    print(f"Device: {device}")
    
    # Train and compare
    results = train_and_compare(X_train, X_test, y_train, y_test, device)
    
    # Print final comparison
    print(f"\n" + "=" * 60)
    print("FINAL COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"Single-Layer Network (2-1):")
    print(f"  Parameters: {2*1 + 1} = 3")
    print(f"  Final Loss: {results['single']['losses'][-1]:.4f}")
    print(f"  Test Accuracy: {results['single']['test_accs'][-1]:.1f}%")
    
    print(f"\nTwo-Layer Network (2-4-1):")
    print(f"  Parameters: {2*4 + 4 + 4*1 + 1} = 17")
    print(f"  Final Loss: {results['two']['losses'][-1]:.4f}")
    print(f"  Test Accuracy: {results['two']['test_accs'][-1]:.1f}%")
    
    # Calculate improvement
    acc_improvement = results['two']['test_accs'][-1] - results['single']['test_accs'][-1]
    loss_improvement = results['single']['losses'][-1] - results['two']['losses'][-1]
    
    print(f"\nImprovement with Two-Layer Network:")
    print(f"  Accuracy improvement: {acc_improvement:+.1f}%")
    print(f"  Loss improvement: {loss_improvement:+.4f}")
    print(f"  Parameter increase: {17-3} parameters ({((17-3)/3*100):.0f}% more)")
    
    # Visualize comparison
    visualize_comparison(results, X, y)

if __name__ == "__main__":
    main()
