"""
10. Liquid Neural Networks (LNN)
Deep Dive Implementation

This file explicitly demonstrates the core concept of a Liquid Time-Constant (LTC) neuron.
Unlike a standard RNN where weights are fixed, in an LNN, the synapses compute
their own connection strengths and time-decay rates dynamically on the fly based on the input.
(Note: Real LNNs use an actual ODE Solver library. This is an Euler-discretized representation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplicitLiquidLayer(nn.Module):
    """
    A simplified discrete-time implementation of a Liquid Time-Constant (LTC) layer.
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.hidden_features = hidden_features
        
        # Standard parameters
        self.weight = nn.Linear(in_features, hidden_features)
        
        # Liquid Parameters (The dynamic parts)
        # 1. Base time constant (tau) - The baseline speed of the neuron
        self.tau = nn.Parameter(torch.ones(hidden_features))
        
        # 2. Synapse modulation networks
        # These networks determine the 'Liquid' aspect: how much the input 
        # changes the neuron's time constant and connection strength.
        self.time_modulator = nn.Linear(in_features, hidden_features)
        self.weight_modulator = nn.Linear(in_features, hidden_features)
        
        # Synapse Sigmoid parameters (mu, gamma) as seen in the hardcore math report
        self.mu = nn.Parameter(torch.zeros(hidden_features))
        self.gamma = nn.Parameter(torch.ones(hidden_features))

    def forward(self, x_t, x_prev):
        """
        x_t: Input at current timestep [Batch, In_Features]
        x_prev: Hidden state from previous timestep [Batch, Hidden_Features]
        """
        # 1. Calculate the dynamic Synapse Activation S(x)
        # S(x) = 1 / (1 + exp(-gamma * (input - mu)))
        # We use a linear layer to map input to the hidden dimension first
        mapped_input = self.weight(x_t)
        synapse_activation = torch.sigmoid(self.gamma * (mapped_input - self.mu))
        
        # 2. Calculate the State-Dependent Time Constant
        # The time constant is NOT fixed. It is modified by the input!
        # tau_dynamic = 1 / ( (1/tau_base) + f(input) )
        f_input_time = torch.sigmoid(self.time_modulator(x_t))
        tau_dynamic = 1.0 / ( (1.0 / torch.abs(self.tau)) + f_input_time )
        
        # 3. Calculate the State-Dependent Weighting A(x, I)
        # The actual connection strength is calculated dynamically
        dynamic_weight = self.weight_modulator(x_t) * synapse_activation
        
        # 4. The Core ODE Update (Discretized via Euler method)
        # dx/dt = -(1/tau_dynamic) * x_prev + dynamic_weight
        # x_next = x_prev + dt * (dx/dt)
        dt = 1.0 # Assuming fixed timestep for this discrete simulation
        
        dx_dt = -(1.0 / tau_dynamic) * x_prev + dynamic_weight
        x_next = x_prev + (dt * dx_dt)
        
        # Apply Tanh to keep the state bounded
        x_next = torch.tanh(x_next)
        
        return x_next

class SimpleLNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.liquid_cell = ExplicitLiquidLayer(in_features, hidden_features)
        self.fc_out = nn.Linear(hidden_features, out_features)
        self.hidden_features = hidden_features

    def forward(self, sequence):
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        
        # Initial hidden state
        x_t = torch.zeros(batch_size, self.hidden_features, device=sequence.device)
        
        for t in range(seq_len):
            current_input = sequence[:, t, :]
            # The liquid cell updates its weights/time-constants on the fly every step
            x_t = self.liquid_cell(current_input, x_t)
            
        return self.fc_out(x_t)


if __name__ == "__main__":
    print("--- 10. Liquid Neural Network (LNN) ---")
    
    batch_size = 2
    seq_len = 10
    input_dim = 4 # e.g. Car sensors
    hidden_dim = 19 # LNNs use incredibly few neurons (e.g. 19 to drive a car)
    output_dim = 2 # Steering angle, Acceleration
    
    lnn = SimpleLNN(input_dim, hidden_dim, output_dim)
    
    dummy_sequence = torch.randn(batch_size, seq_len, input_dim)
    
    print("Executing Liquid continuous-time state updates...")
    predictions = lnn(dummy_sequence)
    
    print(f"Input Sequence: {dummy_sequence.shape} [Batch, Seq_Len, Features]")
    print(f"Output Predictions: {predictions.shape} [Batch, Actions]")
    print("Success: LNN dynamically adapted its time constants and processed the sequence.")
