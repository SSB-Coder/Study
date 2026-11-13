import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. CONVOLUTIONAL NEURAL NETWORK (CNN)
# Explicit layer building (No black-box models)
# ==========================================

class ExplicitCNN(nn.Module):
    """
    A CNN built from scratch using PyTorch primitives.
    Designed for 1-channel images (e.g., MNIST).
    """
    def __init__(self, num_classes=10):
        super(ExplicitCNN, self).__init__()
        
        # Convolutional Layer 1
        # Input: 1 channel, Output: 16 channels, Kernel: 3x3, Stride: 1, Padding: 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves dimensions
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves dimensions again
        
        # Fully Connected (Dense) Classifier
        # Assuming input image was 28x28. After two 2x2 pools, it's 7x7.
        # Flattened size: 32 channels * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature Extraction
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the tensor for the dense layers
        x = x.view(x.size(0), -1) 
        
        # Classification
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 2. RECURRENT NEURAL NETWORK (RNN & LSTM)
# Explicit mathematical implementation inside PyTorch
# ==========================================

class ExplicitRNN(nn.Module):
    """
    A vanilla RNN built by explicitly defining the weight matrices
    and the hidden state loop, rather than using nn.RNN.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ExplicitRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Input to Hidden Weights
        self.W_xh = nn.Linear(input_size, hidden_size, bias=True)
        # Hidden to Hidden Weights (The Memory)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        # Hidden to Output Weights
        self.W_ho = nn.Linear(hidden_size, output_size, bias=True)
        
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev=None):
        """
        x shape: (batch_size, sequence_length, input_size)
        """
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state to zero if not provided
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size).to(x.device)
            
        outputs = []
        h_current = h_prev
        
        # Process the sequence step by step (The limitation of RNNs)
        for t in range(seq_length):
            x_t = x[:, t, :] # Get input at time step t
            
            # The core RNN equation: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
            h_current = self.tanh(self.W_xh(x_t) + self.W_hh(h_current))
            
            # Compute output for this step
            o_t = self.W_ho(h_current)
            outputs.append(o_t.unsqueeze(1))
            
        # Concatenate outputs across the sequence dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs, h_current


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    print("--- Testing Explicit CNN ---")
    # Batch size 4, 1 channel, 28x28 image
    dummy_image_data = torch.randn(4, 1, 28, 28)
    cnn = ExplicitCNN(num_classes=10)
    out_cnn = cnn(dummy_image_data)
    print(f"CNN Output Shape: {out_cnn.shape} (Expected: 4, 10)")
    
    print("\n--- Testing Explicit RNN ---")
    # Batch size 2, Sequence Length 5, Input Size 10
    dummy_sequence_data = torch.randn(2, 5, 10)
    rnn = ExplicitRNN(input_size=10, hidden_size=20, output_size=5)
    out_rnn, hidden_state = rnn(dummy_sequence_data)
    print(f"RNN Output Shape: {out_rnn.shape} (Expected: 2, 5, 5)")
    print(f"RNN Final Hidden State Shape: {hidden_state.shape} (Expected: 2, 20)")
