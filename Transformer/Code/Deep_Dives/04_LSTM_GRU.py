"""
04. Long Short-Term Memory (LSTM) & GRU
Deep Dive Implementation

This file breaks open the "black box" of LSTMs and GRUs.
We explicitly define the Forget, Input, and Output gates for the LSTM,
and the Update and Reset gates for the GRU to match the hardcore math report.
"""

import torch
import torch.nn as nn

class ExplicitLSTMCell(nn.Module):
    """A single time step of an LSTM, showing the conveyor belt logic."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # In PyTorch, we can combine the inputs x_t and h_{t-1} into one vector.
        # So we need a Linear layer that takes (input_size + hidden_size)
        combined_size = input_size + hidden_size
        
        # 1. The Forget Bouncer (f_t)
        self.W_f = nn.Linear(combined_size, hidden_size)
        
        # 2. The Input Bouncer (i_t) and Candidate Value (C_tilde)
        self.W_i = nn.Linear(combined_size, hidden_size)
        self.W_C = nn.Linear(combined_size, hidden_size)
        
        # 3. The Output Bouncer (o_t)
        self.W_o = nn.Linear(combined_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state: [x_t, h_{t-1}]
        combined = torch.cat((x_t, h_prev), dim=1)
        
        # Step 1: Forget Gate - What old memory should we erase?
        f_t = torch.sigmoid(self.W_f(combined))
        
        # Step 2: Input Gate - What new memory should we keep?
        i_t = torch.sigmoid(self.W_i(combined))
        C_tilde = torch.tanh(self.W_C(combined)) # The proposed new candidate memory
        
        # Step 3: Update the Conveyor Belt (Cell State)
        # Multiply old memory by forget gate, add new memory scaled by input gate
        c_next = (f_t * c_prev) + (i_t * C_tilde)
        
        # Step 4: Output Gate - What part of the conveyor belt do we output?
        o_t = torch.sigmoid(self.W_o(combined))
        h_next = o_t * torch.tanh(c_next)
        
        return h_next, c_next


class ExplicitGRUCell(nn.Module):
    """A single time step of a GRU, showing the simplified gate logic."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size
        
        # 1. Update Gate (z_t) - Blends Forget and Input gates
        self.W_z = nn.Linear(combined_size, hidden_size)
        
        # 2. Reset Gate (r_t) - Decides what past to ignore entirely
        self.W_r = nn.Linear(combined_size, hidden_size)
        
        # 3. Candidate Hidden State computation
        # Note: the candidate takes input and the *reset* hidden state
        self.W_h = nn.Linear(combined_size, hidden_size)

    def forward(self, x_t, h_prev):
        combined = torch.cat((x_t, h_prev), dim=1)
        
        z_t = torch.sigmoid(self.W_z(combined))
        r_t = torch.sigmoid(self.W_r(combined))
        
        # Calculate the candidate state using the reset memory
        combined_reset = torch.cat((x_t, r_t * h_prev), dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        
        # Final Memory Update: Blend old memory and candidate memory based on z_t
        h_next = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_next


if __name__ == "__main__":
    print("--- 04. LSTM and GRU Mathematics ---")
    batch_size = 1
    input_dim = 10
    hidden_dim = 20
    
    # Initialize both cells
    lstm = ExplicitLSTMCell(input_dim, hidden_dim)
    gru = ExplicitGRUCell(input_dim, hidden_dim)
    
    # Dummy data for a single time step
    x_t = torch.randn(batch_size, input_dim)
    
    # LSTM requires both Hidden State and Cell State
    h_prev_lstm = torch.zeros(batch_size, hidden_dim)
    c_prev_lstm = torch.zeros(batch_size, hidden_dim)
    
    # GRU only requires Hidden State
    h_prev_gru = torch.zeros(batch_size, hidden_dim)
    
    # Run the cells
    h_next_lstm, c_next_lstm = lstm(x_t, h_prev_lstm, c_prev_lstm)
    h_next_gru = gru(x_t, h_prev_gru)
    
    print(f"LSTM Output Hidden State Shape: {h_next_lstm.shape}")
    print(f"LSTM Output Cell State Shape: {c_next_lstm.shape}")
    print(f"GRU Output Hidden State Shape: {h_next_gru.shape}")
    print("Success: Explicit LSTM and GRU gates calculated correctly.")
