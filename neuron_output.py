import math

def sigmoid(z):
    """Calculate sigmoid activation function"""
    return 1 / (1 + math.exp(-z))

def main():
    # Get input values from user
    x1, x2 = map(float, input("Enter x1, x2: ").split())
    w1, w2 = map(float, input("Enter w1, w2: ").split())
    b = float(input("Enter bias: "))
    
    # Calculate weighted sum
    z = x1 * w1 + x2 * w2 + b
    
    # Apply sigmoid activation function
    output = sigmoid(z)
    
    # Display result rounded to 3 decimal places
    print(f"Neuron output: {output:.3f}")

if __name__ == "__main__":
    main()