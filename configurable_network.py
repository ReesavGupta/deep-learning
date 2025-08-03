import random
import math

def sigmoid(z):
    """Calculate sigmoid activation function"""
    return 1 / (1 + math.exp(-z))

def relu(z):
    """Calculate ReLU activation function"""
    return max(0, z)

def main():
    # Get user configuration
    n = int(input("Enter number of inputs: "))
    h = int(input("Enter number of hidden neurons: "))
    activation = input("Enter activation (sigmoid/relu): ").lower()
    
    # Validate activation function
    if activation not in ['sigmoid', 'relu']:
        print("Invalid activation function. Using sigmoid.")
        activation = 'sigmoid'
    
    # Set activation function
    activation_func = sigmoid if activation == 'sigmoid' else relu
    
    # Set random seed for reproducible results (optional)
    random.seed(42)
    
    # Generate n random input values
    inputs = [random.uniform(-1, 1) for _ in range(n)]
    print(f"Inputs: {[round(x, 2) for x in inputs]}")
    
    # Hidden layer: h neurons with n weights each + bias
    hidden_weights = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(h)]
    hidden_biases = [random.uniform(-1, 1) for _ in range(h)]
    
    print(f"Hidden layer weights: {[[round(w, 2) for w in neuron] for neuron in hidden_weights]}")
    print(f"Hidden layer biases: {[round(b, 2) for b in hidden_biases]}")
    
    # Calculate hidden layer outputs
    hidden_outputs = []
    for i in range(h):
        z = sum(inputs[j] * hidden_weights[i][j] for j in range(n)) + hidden_biases[i]
        output = activation_func(z)
        hidden_outputs.append(output)
    
    print(f"Hidden outputs: {[round(h_out, 3) for h_out in hidden_outputs]}")
    
    # Output layer: 1 neuron with h weights + bias
    output_weights = [random.uniform(-1, 1) for _ in range(h)]
    output_bias = random.uniform(-1, 1)
    
    print(f"Output layer weights: {[round(w, 2) for w in output_weights]}")
    print(f"Output bias: {round(output_bias, 2)}")
    
    # Calculate final output (always use sigmoid for output layer)
    z_output = sum(hidden_outputs[i] * output_weights[i] for i in range(h)) + output_bias
    final_output = sigmoid(z_output)
    
    print(f"Activation function used: {activation}")
    print(f"Final Output: {round(final_output, 3)}")

if __name__ == "__main__":
    main()