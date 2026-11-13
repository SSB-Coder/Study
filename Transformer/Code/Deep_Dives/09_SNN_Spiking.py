"""
09. Spiking Neural Networks (SNN)
Deep Dive Implementation

This file explicitly demonstrates a Leaky Integrate-and-Fire (LIF) neuron.
Crucially, it implements the "Surrogate Gradient" trick, allowing backpropagation
to pass through the non-differentiable Dirac delta spike function.
"""

import torch
import torch.nn as nn

class SurrogateSpike(torch.autograd.Function):
    """
    The Surrogate Gradient Trick.
    Forward Pass: Returns a hard 1 or 0 (biological spike).
    Backward Pass: Pretends the spike was a smooth sigmoid curve so calculus works.
    """
    @staticmethod
    def forward(ctx, voltage_membrane, threshold):
        # Save voltage for the backward pass
        ctx.save_for_backward(voltage_membrane, threshold)
        
        # Hard Spike: If voltage >= threshold, return 1.0, else 0.0
        return (voltage_membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        voltage_membrane, threshold = ctx.saved_tensors
        
        # We approximate the Dirac Delta derivative with a Fast Sigmoid derivative
        # The steepness of this curve is controlled by alpha
        alpha = 2.0
        surrogate_gradient = (1 / (alpha * torch.abs(voltage_membrane - threshold) + 1.0) ** 2)
        
        # Chain rule
        return grad_output * surrogate_gradient, None

# Wrap the autograd function so it acts like a normal PyTorch layer
spike_activation = SurrogateSpike.apply

class ExplicitLIFLayer(nn.Module):
    """A layer of Leaky Integrate-and-Fire Neurons."""
    def __init__(self, in_features, out_features, tau=0.8, threshold=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Synaptic Weights
        self.weight = nn.Linear(in_features, out_features)
        
        # Biological Parameters
        self.tau = tau             # The leak rate (how fast the bucket drains)
        self.threshold = torch.tensor(threshold) # The brim of the bucket
        self.v_rest = 0.0          # Resting voltage
        
    def forward(self, input_spikes, voltage_prev):
        """
        input_spikes: binary tensor of spikes from previous layer [Batch, In_Features]
        voltage_prev: the internal voltage from the previous timestep [Batch, Out_Features]
        """
        # 1. Integrate: Add incoming current (spikes * weights)
        current = self.weight(input_spikes)
        
        # 2. Leak: The voltage naturally decays toward v_rest over time
        voltage_next = (self.tau * voltage_prev) + current
        
        # 3. Fire: Check if voltage crossed the threshold using our Surrogate function
        out_spikes = spike_activation(voltage_next, self.threshold)
        
        # 4. Hard Reset: If a neuron fired, its voltage instantly drops back to 0
        # We mathematically mask out the voltage of neurons that just fired
        voltage_next = voltage_next * (1.0 - out_spikes)
        
        return out_spikes, voltage_next


class SpikingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps=20):
        super().__init__()
        self.time_steps = time_steps
        self.lif1 = ExplicitLIFLayer(input_dim, hidden_dim)
        self.lif2 = ExplicitLIFLayer(hidden_dim, output_dim)

    def forward(self, x):
        # x is continuous data (e.g. pixels). 
        # In a real SNN, this is usually converted to a train of spikes first (Poisson encoding).
        # For simplicity, we just feed constant current over time.
        batch_size = x.size(0)
        
        # Initialize internal voltages to 0
        v1 = torch.zeros(batch_size, self.lif1.out_features, device=x.device)
        v2 = torch.zeros(batch_size, self.lif2.out_features, device=x.device)
        
        # Keep track of the total number of spikes the output layer fired
        output_spikes_sum = torch.zeros(batch_size, self.lif2.out_features, device=x.device)
        
        # The Asynchronous Time Loop
        for t in range(self.time_steps):
            # Layer 1 gets constant pixel intensity as current
            spikes1, v1 = self.lif1(x, v1)
            
            # Layer 2 only receives pure binary spikes
            spikes2, v2 = self.lif2(spikes1, v2)
            
            output_spikes_sum += spikes2
            
        # The final prediction is usually based on "Rate Coding": 
        # Which output neuron fired the MOST spikes over the time window?
        return output_spikes_sum

if __name__ == "__main__":
    print("--- 09. Spiking Neural Network (LIF & Surrogate Gradients) ---")
    
    batch_size = 2
    input_dim = 10
    hidden_dim = 15
    output_classes = 3
    time_window = 20 # Simulate 20 milliseconds of brain activity
    
    # Create the network
    snn = SpikingNetwork(input_dim, hidden_dim, output_classes, time_window)
    
    # Dummy continuous data (will act as constant input current)
    input_current = torch.rand(batch_size, input_dim)
    
    # Forward Pass
    print(f"Simulating network over {time_window} timesteps...")
    total_spikes_fired = snn(input_current)
    
    print(f"Input Data Shape: {input_current.shape}")
    print(f"Output (Total Spikes per Class): \n{total_spikes_fired.detach().numpy()}")
    print("Success: SNN ran with biological time integration and differentiable spikes.")
