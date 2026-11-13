"""
J10: Explainable AI (Permutation Feature Importance, Shapley Values, Mini-LIME)
Pure NumPy implementation of feature attribution mechanisms:
1. Permutation Feature Importance
2. Exact Shapley Values via coalition permutation enumeration
3. Local Interpretable Model-agnostic Explanations (LIME)
"""

import numpy as np
import itertools
import math

class ExplainableAIToolkit:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def permutation_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Measures loss increase when each feature column is randomly shuffled."""
        base_preds = self.predict_fn(X)
        base_mse = np.mean((base_preds - y) ** 2)

        n_features = X.shape[1]
        importances = np.zeros(n_features)

        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_preds = self.predict_fn(X_perm)
            perm_mse = np.mean((perm_preds - y) ** 2)
            importances[j] = perm_mse - base_mse

        return importances

    def exact_shapley_values(self, x: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """
        Computes exact Shapley values for an instance x relative to baseline
        by enumerating all 2^D feature coalitions.
        """
        D = len(x)
        shapley = np.zeros(D)

        feature_indices = list(range(D))

        for i in range(D):
            other_features = [f for f in feature_indices if f != i]
            # Enumerate all subsets S of other features
            for s_len in range(len(other_features) + 1):
                combos = itertools.combinations(other_features, s_len)
                # Weight: |S|! * (D - |S| - 1)! / D!
                weight = (math.factorial(s_len) * math.factorial(D - s_len - 1)) / math.factorial(D)

                for S in combos:
                    # Model value with S
                    x_S = baseline.copy()
                    for feat in S:
                        x_S[feat] = x[feat]
                    val_S = float(np.asarray(self.predict_fn(x_S.reshape(1, -1))).item())

                    # Model value with S U {i}
                    x_S_plus_i = x_S.copy()
                    x_S_plus_i[i] = x[i]
                    val_S_plus = float(np.asarray(self.predict_fn(x_S_plus_i.reshape(1, -1))).item())

                    # Marginal contribution
                    shapley[i] += weight * (val_S_plus - val_S)

        return shapley

    def mini_lime(self, x: np.ndarray, n_samples: int = 150, sigma: float = 1.0) -> np.ndarray:
        """
        Local Interpretable Model-agnostic Explanations:
        Perturbs x, weights by distance kernel, and fits local weighted linear surrogate.
        """
        D = len(x)
        # Generate Gaussian perturbations around x
        perturbations = x + np.random.randn(n_samples, D) * 0.5
        preds = self.predict_fn(perturbations)

        # Exponential distance kernel weights
        dists = np.sum((perturbations - x) ** 2, axis=1)
        weights = np.exp(-dists / (sigma ** 2))

        # Fit weighted linear regression: beta = (Z^T W Z)^-1 Z^T W y
        Z = np.hstack([np.ones((n_samples, 1)), perturbations])
        W_diag = np.diag(weights)
        beta = np.linalg.pinv(Z.T @ W_diag @ Z) @ (Z.T @ W_diag @ preds)

        return beta[1:] # Return feature coefficients


if __name__ == "__main__":
    print("--- J10: Explainable AI (Permutation, SHAP, LIME) Sanity Check ---")
    np.random.seed(42)

    # Known non-linear ground truth function: y = 3.0*x0 + 1.0*x1 + 0.1*x2 + 0.0*x3
    def model_fn(X):
        return 3.0 * X[:, 0] + 1.0 * X[:, 1] + 0.1 * X[:, 2]

    x_test = np.array([2.0, 1.0, 0.5, 0.0])
    baseline = np.array([0.0, 0.0, 0.0, 0.0])

    explainer = ExplainableAIToolkit(model_fn)

    # 1. Exact Shapley Values
    shap_vals = explainer.exact_shapley_values(x_test, baseline)
    print(f"Exact Shapley Values: {shap_vals.round(3)}")

    # Efficiency axiom check: sum(phi_i) = f(x) - f(baseline)
    expected_diff = float((model_fn(x_test.reshape(1, -1)) - model_fn(baseline.reshape(1, -1))).item())
    sum_shap = float(np.sum(shap_vals))
    print(f"Sum of Shapley Values: {sum_shap:.3f} | Actual Model Output Diff: {expected_diff:.3f}")
    assert np.isclose(sum_shap, expected_diff), "Shapley values must satisfy Efficiency Axiom"
    assert shap_vals[0] > shap_vals[1] > shap_vals[2], "Shapley attributions must match feature impact ordering"

    # 2. Mini-LIME Local Surrogate
    lime_coeffs = explainer.mini_lime(x_test, n_samples=200)
    print(f"Mini-LIME Local Coefficients: {lime_coeffs.round(3)}")
    assert lime_coeffs[0] > lime_coeffs[1] > lime_coeffs[2], "LIME coefficients must reflect true local gradients"

    print("J10 Sanity Check: PASSED cleanly.")
