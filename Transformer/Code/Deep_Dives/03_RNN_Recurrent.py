"""
03. Recurrent Neural Networks (RNN)
Deep Dive Implementation

This file demonstrates the explicit inner workings of an RNN cell.
Instead of using `nn.RNN`, we manually define the W_hx and W_hh matrices 
and manually loop over the sequence length to show how the hidden state 
(the short-term memory) is passed forward through time.
"""

import torch
import torch.nn as nn

class ExplicitRNNCell(nn.Module):
    """A single unrolled time step of an RNN."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # W_hx: Weights for the incoming input x_t
        self.weight_ih = nn.Linear(input_size, hidden_size)
        
        # W_hh: Weights for the previous memory state h_{t-1}
        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x_t, h_prev):
        # Equation: h_t = tanh(W_hx @ x_t + W_hh @ h_prev + bias)
        # Note: nn.Linear automatically handles the biases
        h_next = torch.tanh(self.weight_ih(x_t) + self.weight_hh(h_prev))
        return h_next

class ExplicitRNN(nn.Module):
    """The full RNN that unrolls the cell over the time sequence."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = ExplicitRNNCell(input_size, hidden_size)
        
        # W_yh: Weights to map the hidden state to a final prediction
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, X_sequence):
        # X_sequence shape: [Batch, Sequence_Length, Input_Size]
        batch_size = X_sequence.size(0)
        seq_length = X_sequence.size(1)
        
        # Initialize the first hidden state (h_0) as all zeros (an empty memory)
        h_t = torch.zeros(batch_size, self.hidden_size, device=X_sequence.device)
        
        # We will collect the outputs at every time step just in case we need them
        outputs = []
        
        # The Unrolled Loop: Process the sequence one word/step at a time
        for t in range(seq_length):
            x_t = X_sequence[:, t, :] # Get the data at time t for all batches
            
            # Pass input and previous memory into the cell to get new memory
            h_t = self.rnn_cell(x_t, h_t)
            
            outputs.append(h_t)
            
        # We usually only care about the final prediction after reading the whole sequence
        final_memory = h_t
        
        # Make a prediction based on the final memory state
        logits = self.fc_out(final_memory)
        return logits

if __name__ == "__main__":
    print("--- 03. Recurrent Neural Network ---")
    
    # Simulate a batch of 2 sentences, each with 5 words.
    # Each word is represented by an embedding vector of size 10.
    batch_size = 2
    sequence_len = 5
    input_dim = 10
    hidden_dim = 20
    output_classes = 3 # e.g. Positive, Negative, Neutral sentiment
    
    # Create the network
    rnn = ExplicitRNN(input_dim, hidden_dim, output_classes)
    
    # Generate dummy sequential data: [Batch, Sequence_Length, Features]
    sequence_data = torch.randn(batch_size, sequence_len, input_dim)
    
    # Forward Pass
    predictions = rnn(sequence_data)
    
    print(f"Input Sequence Shape: {sequence_data.shape} [Batch, Seq_Len, Embed_Dim]")
    print(f"Final Hidden State updated {sequence_len} times.")
    print(f"Output Predictions Shape: {predictions.shape} [Batch, Classes]")
    print("Success: Explicit RNN unrolled sequence processing complete.")
