import random
import math

def sigmoid(z):
    """Calculate sigmoid activation function"""
    return 1 / (1 + math.exp(-z))

def main():
    # Set random seed for reproducible results (optional)
    random.seed(42)
    
    # Generate 3 random input values
    inputs = [random.uniform(-1, 1) for _ in range(3)]
    print(f"Inputs: {[round(x, 1) for x in inputs]}")
    
    # First layer: 2 neurons with random weights and biases
    hidden_weights = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)]
    hidden_biases = [random.uniform(-1, 1) for _ in range(2)]
    
    print(f"Hidden layer weights: {[[round(w, 1) for w in neuron] for neuron in hidden_weights]}")
    print(f"Hidden layer biases: {[round(b, 1) for b in hidden_biases]}")
    
    # Calculate hidden layer outputs
    hidden_outputs = []
    for i in range(2):
        z = sum(inputs[j] * hidden_weights[i][j] for j in range(3)) + hidden_biases[i]
        output = sigmoid(z)
        hidden_outputs.append(output)
    
    print(f"Hidden outputs: {[round(h, 2) for h in hidden_outputs]}")
    
    # Second layer: 1 output neuron with random weights and bias
    output_weights = [random.uniform(-1, 1) for _ in range(2)]
    output_bias = random.uniform(-1, 1)
    
    print(f"Output layer weights: {[round(w, 1) for w in output_weights]}")
    print(f"Bias: {round(output_bias, 1)}")
    
    # Calculate final output
    z_output = sum(hidden_outputs[i] * output_weights[i] for i in range(2)) + output_bias
    final_output = sigmoid(z_output)
    
    print(f"Final Output: {round(final_output, 3)}")

if __name__ == "__main__":
    main()