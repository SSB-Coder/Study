import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ==========================================
# 1. THE PERCEPTRON (Atomic Scale)
# ==========================================
class Perceptron:
    """
    The simplest neural network model.
    Learns a linear decision boundary for binary classification.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def _activation_func(self, x):
        # Step function: 1 if x>=0 else 0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure y is binary (0 or 1)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Forward pass: dot product of weights and input + bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_func(linear_output)

                # Update rule (Delta rule)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation_func(linear_output)


# ==========================================
# 2. FEEDFORWARD NEURAL NETWORK (MLP)
# ==========================================
class NeuralNetwork:
    """
    A simple 2-layer (1 hidden, 1 output) Feedforward Neural Network using pure NumPy.
    Uses Sigmoid activation and Mean Squared Error loss.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Initialize Weights randomly, Biases to zero
        # Shape: (input_size x hidden_size)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Shape: (hidden_size x output_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    # Activation Functions
    def sigmoid(self, z):
        # Using np.clip to prevent overflow warnings in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        """
        Forward propagation: Data flows from input to output.
        """
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        # Layer 2 (Output)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2

    def backward(self, X, y, output):
        """
        Backpropagation: Calculate gradients and update weights.
        """
        m = X.shape[0] # Number of training examples
        
        # Ensure y is shaped correctly (m x 1)
        y = y.reshape(-1, 1)

        # 1. Compute Loss Gradient at output layer
        # Derivative of MSE is (output - y), multiplied by activation derivative
        dZ2 = (output - y) * self.sigmoid_derivative(self.Z2)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # 2. Propagate error to hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # 3. Update Weights and Biases (Gradient Descent step)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Print Loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y.reshape(-1, 1) - output))
                print(f'Epoch {epoch}, MSE Loss: {loss:.4f}')


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    print("Generating Synthetic Data...")
    # Create a simple non-linear dataset
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                               n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("\nTraining Multilayer Perceptron (Feedforward NN)...")
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    nn.train(X_train, y_train, epochs=5000)

    # Test accuracy
    predictions = nn.forward(X_test)
    predictions = (predictions > 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
