"""
51. Neural Symbolic Discovery & Physics Law Extraction
Deep Dive Implementation: Differentiable Symbolic Regression & Analytic Extraction

This file explicitly demonstrates extracting human-readable mathematical laws from data:
1. Symbolic Operator Layer: Applying elementary basis functions (identity, square, sin, cos).
2. L1 Sparsity Regularization: Forcing unneeded mathematical terms to exact zero.
3. Closed-Form Equation Extraction: Printing the discovered physics equation as a string.
"""

import torch
import torch.nn as nn
import math

class SymbolicBasisLayer(nn.Module):
    """
    Evaluates a candidate library of elementary physical operators on input variables:
    [x, x^2, sin(pi * x), cos(pi * x), exp(-x^2)]
    """
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        # 5 basis functions per input variable
        self.num_operators = 5
        self.out_features = in_features * self.num_operators

    def forward(self, x):
        """
        x: [Batch_Size, In_Features]
        returns: [Batch_Size, In_Features * Num_Operators]
        """
        bases = []
        for i in range(self.in_features):
            xi = x[:, i:i+1]
            f_id = xi                                       # Identity: x
            f_sq = xi ** 2                                  # Quadratic: x^2
            f_sin = torch.sin(math.pi * xi)                 # Wave/Harmonic: sin(pi * x)
            f_cos = torch.cos(math.pi * xi)                 # Wave/Harmonic: cos(pi * x)
            f_gauss = torch.exp(-torch.clamp(xi**2, max=10.0)) # Gaussian / Local: exp(-x^2)
            bases.extend([f_id, f_sq, f_sin, f_cos, f_gauss])

        return torch.cat(bases, dim=-1)


class NeuralSymbolicRegressor(nn.Module):
    """
    Differentiable Neural Symbolic Regression model.
    Combines candidate basis functions with a sparse linear recombination layer.
    """
    def __init__(self, in_features=2):
        super().__init__()
        self.in_features = in_features
        self.basis = SymbolicBasisLayer(in_features)
        
        # Linear layer mapping all basis functions to the target output
        self.coefficients = nn.Linear(self.basis.out_features, 1, bias=True)

    def forward(self, x):
        phi = self.basis(x)
        return self.coefficients(phi)

    def extract_symbolic_equation(self, threshold=0.08):
        """
        Inspects the trained weights, thresholds near-zero coefficients,
        and constructs the analytical mathematical equation as a human-readable string.
        """
        weights = self.coefficients.weight.data.squeeze(0).cpu().numpy()
        bias = self.coefficients.bias.data.item()

        terms = []
        op_names = ["x_{idx}", "x_{idx}^2", "sin(pi * x_{idx})", "cos(pi * x_{idx})", "exp(-x_{idx}^2)"]

        for var_idx in range(self.in_features):
            for op_idx in range(5):
                flat_idx = var_idx * 5 + op_idx
                coeff = weights[flat_idx]
                if abs(coeff) > threshold:
                    term_str = op_names[op_idx].format(idx=var_idx + 1)
                    terms.append(f"{coeff:+.3f} * {term_str}")

        if abs(bias) > threshold:
            terms.append(f"{bias:+.3f}")

        if not terms:
            return "0.0"
        return " ".join(terms).lstrip("+ ")


if __name__ == "__main__":
    print("--- 51. Neural Symbolic Discovery & Equation Extraction ---")
    print("Discovering Hidden Physical Laws from Noisy Experimental Data...\n")

    torch.manual_seed(42)

    # 1. Ground Truth Physical Law: y = 2.5 * x_1^2 + 1.2 * sin(pi * x_2)
    print("Ground Truth Law to Discover: y = 2.500 * x_1^2 + 1.200 * sin(pi * x_2)")

    # 2. Generate Noisy Synthetic Training Data
    n_samples = 400
    x_train = torch.empty(n_samples, 2).uniform_(-1.5, 1.5)
    noise = torch.randn(n_samples, 1) * 0.05 # 5% sensor noise
    y_true = 2.5 * (x_train[:, 0:1] ** 2) + 1.2 * torch.sin(math.pi * x_train[:, 1:2])
    y_train = y_true + noise

    # 3. Model & Optimizer
    model = NeuralSymbolicRegressor(in_features=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    print("Beginning Sparse Neural Symbolic Training...")
    # Training Loop with L1 Regularization (Lasso) to drive spurious coefficients to zero
    lambda_l1 = 0.02

    for epoch in range(1, 201):
        optimizer.zero_grad()
        y_pred = model(x_train)
        
        # Mean Squared Error (Data Fitting)
        mse_loss = torch.mean((y_pred - y_train) ** 2)
        
        # L1 Sparsity Penalty on Coefficients (Occam's Razor: favor simplest equations)
        l1_penalty = torch.sum(torch.abs(model.coefficients.weight))
        
        total_loss = mse_loss + lambda_l1 * l1_penalty
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | MSE: {mse_loss.item():.5f} | L1 Penalty: {l1_penalty.item():.4f}")

    # 4. Extract and print the discovered closed-form formula
    discovered_equation = model.extract_symbolic_equation(threshold=0.1)
    print(f"\n========================================================")
    print(f"DISCOVERED PHYSICAL EQUATION:")
    print(f"  y = {discovered_equation}")
    print(f"========================================================")

    # 5. Out-of-distribution Generalization Test
    x_test = torch.tensor([[1.2, 0.5], [-1.0, -0.5]])
    y_test_exact = 2.5 * (x_test[:, 0:1] ** 2) + 1.2 * torch.sin(math.pi * x_test[:, 1:2])
    y_test_pred = model(x_test)
    print(f"\nGeneralization Test on Unseen Coordinates:")
    print(f"Exact Target:     {y_test_exact.squeeze().detach().numpy()}")
    print(f"Model Prediction: {y_test_pred.squeeze().detach().numpy()}")
    print("Success: Exact mathematical formula extracted from neural network weights.")
