import numpy as np
import matplotlib.pyplot as plt
import math

# Set random seed for reproducibility
np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x > 0, x, alpha * x)

class RandomNeuralNetwork:
    def __init__(self):
        # Randomly generate network structure
        self.n_inputs = np.random.randint(3, 7)  # [3, 6]
        self.n_hidden_layers = np.random.randint(1, 4)  # [1, 3]
        self.hidden_layer_sizes = [np.random.randint(2, 6) for _ in range(self.n_hidden_layers)]
        self.n_outputs = 1  # Single output
        
        # Generate random inputs
        self.inputs = np.random.uniform(-10, 10, self.n_inputs)
        
        # Generate random weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [self.n_inputs] + self.hidden_layer_sizes + [self.n_outputs]
        
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.uniform(-1, 1, (layer_sizes[i+1], layer_sizes[i]))
            bias_vector = np.random.uniform(-1, 1, layer_sizes[i+1])
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
    
    def forward_pass(self, activation_func):
        """Perform forward pass with given activation function"""
        current_input = self.inputs.copy()
        
        # Forward through all layers
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weight, current_input) + bias
            
            # Apply activation function (sigmoid for output layer)
            if i == len(self.weights) - 1:  # Output layer
                current_input = sigmoid(z)
            else:  # Hidden layers
                current_input = activation_func(z)
        
        return current_input[0] if len(current_input) == 1 else current_input
    
    def print_network_structure(self):
        """Print detailed network structure"""
        print("=" * 60)
        print("RANDOM NEURAL NETWORK STRUCTURE")
        print("=" * 60)
        print(f"Number of inputs: {self.n_inputs}")
        print(f"Number of hidden layers: {self.n_hidden_layers}")
        print(f"Hidden layer sizes: {self.hidden_layer_sizes}")
        print(f"Number of outputs: {self.n_outputs}")
        print()
        
        print(f"Input values: {[round(x, 2) for x in self.inputs]}")
        print()
        
        layer_names = ["Input→Hidden1"] + [f"Hidden{i}→Hidden{i+1}" for i in range(1, self.n_hidden_layers)] + [f"Hidden{self.n_hidden_layers}→Output"]
        
        for i, (weight, bias, name) in enumerate(zip(self.weights, self.biases, layer_names)):
            print(f"{name} Layer:")
            print(f"  Weights shape: {weight.shape}")
            print(f"  Weights: {np.round(weight, 2).tolist()}")
            print(f"  Biases: {np.round(bias, 2).tolist()}")
            print()

def main():
    # Create random neural network
    network = RandomNeuralNetwork()
    network.print_network_structure()
    
    # Define activation functions
    activation_functions = {
        'Sigmoid': sigmoid,
        'Tanh': tanh,
        'ReLU': relu,
        'Leaky ReLU': leaky_relu
    }
    
    # Perform forward pass with each activation function
    results = {}
    print("FORWARD PASS RESULTS:")
    print("-" * 40)
    
    for name, func in activation_functions.items():
        output = network.forward_pass(func)
        results[name] = output
        print(f"{name:12}: {output:.6f}")
    
    print()
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart
    names = list(results.keys())
    values = list(results.values())
    colors = ['blue', 'green', 'red', 'orange']
    
    bars = plt.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Neural Network Output Comparison\nDifferent Activation Functions', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Activation Function', fontsize=12)
    plt.ylabel('Final Output Value', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, max(values) * 1.2)
    
    # Add network info to plot
    info_text = f"Network: {network.n_inputs} inputs → {' → '.join(map(str, network.hidden_layer_sizes))} → 1 output"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("ANALYSIS:")
    print("-" * 40)
    max_output = max(results.values())
    min_output = min(results.values())
    max_func = max(results, key=results.get)
    min_func = min(results, key=results.get)
    
    print(f"Highest output: {max_func} ({max_output:.6f})")
    print(f"Lowest output:  {min_func} ({min_output:.6f})")
    print(f"Output range:   {max_output - min_output:.6f}")
    
    # Additional insights
    print(f"\nInsights:")
    print(f"- Sigmoid output is always in (0, 1)")
    print(f"- Tanh output is always in (-1, 1)")
    print(f"- ReLU can produce larger positive values")
    print(f"- Leaky ReLU handles negative inputs better than ReLU")

if __name__ == "__main__":
    main()