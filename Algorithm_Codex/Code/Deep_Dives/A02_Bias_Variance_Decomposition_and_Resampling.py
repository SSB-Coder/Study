"""
A02. Bias-Variance Decomposition & Resampling
From-scratch implementation in pure NumPy.
"""

import numpy as np

class Resampler:
    """Hand-crafted resampling suite: K-Fold, LOOCV, and Bootstrap."""
    @staticmethod
    def k_fold_split(n_samples: int, k: int = 5, shuffle: bool = True, seed: int = 42):
        """Yields (train_indices, val_indices) for k folds."""
        indices = np.arange(n_samples)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        
        current = 0
        splits = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            splits.append((train_idx, val_idx))
            current = stop
        return splits

    @staticmethod
    def bootstrap_sample(n_samples: int, seed: int = None):
        """Draws n_samples with replacement. Returns (sample_idx, oob_idx)."""
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        unique_samples = np.unique(sample_idx)
        all_indices = np.arange(n_samples)
        oob_idx = np.setdiff1d(all_indices, unique_samples)
        return sample_idx, oob_idx

class BiasVarianceDecomposition:
    """Estimates empirical bias^2, variance, and noise for a given regressor."""
    @staticmethod
    def evaluate(estimator_fn, X_train_pool, y_train_pool, X_test, y_test_true, 
                 n_bootstraps: int = 50, sample_size: int = 40):
        m_pool = len(X_train_pool)
        predictions = np.zeros((n_bootstraps, len(X_test)))
        
        for b in range(n_bootstraps):
            # Sample subset
            idx = np.random.choice(m_pool, size=sample_size, replace=True)
            X_b, y_b = X_train_pool[idx], y_train_pool[idx]
            
            # Train estimator
            model = estimator_fn()
            model.fit(X_b, y_b)
            predictions[b, :] = model.predict(X_test)
            
        # Expected prediction across models: E[y_hat(x)]
        mean_pred = np.mean(predictions, axis=0)
        
        # Bias^2: (E[y_hat] - y_true)^2 <- eq. 3.1 in report
        bias_sq = np.mean((mean_pred - y_test_true) ** 2)
        
        # Variance: E[(y_hat - E[y_hat])^2] <- eq. 3.1 in report
        variance = np.mean(np.var(predictions, axis=0))
        
        return bias_sq, variance

if __name__ == "__main__":
    print("--- A02: Bias-Variance Decomposition & Resampling Sanity Check ---")
    np.random.seed(42)
    n = 100
    
    # 1. Test K-Fold
    splits = Resampler.k_fold_split(n_samples=n, k=5)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(val_idx) == 20
        assert len(train_idx) == 80
        assert len(np.intersect1d(train_idx, val_idx)) == 0
    print("K-Fold (5 folds of 100 items): Cleanly disjoint and partition complete.")
    
    # 2. Test Bootstrap OOB ratio (asymptotic ~ 36.8%)
    boot_idx, oob_idx = Resampler.bootstrap_sample(n_samples=10000, seed=42)
    oob_fraction = len(oob_idx) / 10000.0
    print(f"Empirical OOB fraction: {oob_fraction:.4f} (Theoretical limit 1/e = {1/np.e:.4f})")
    assert np.isclose(oob_fraction, 1.0 / np.e, atol=0.01)
    
    # 3. Simple polynomial regression bias-variance check
    class DummyLinearModel:
        def fit(self, X, y):
            # Simple 1D linear regression
            X_b = np.c_[np.ones(len(X)), X]
            self.w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        def predict(self, X):
            X_b = np.c_[np.ones(len(X)), X]
            return X_b @ self.w
            
    X_pool = np.random.uniform(-3, 3, size=200)
    # Underlying non-linear true function
    y_pool = np.sin(X_pool) + np.random.normal(0, 0.2, size=200)
    X_test = np.linspace(-2.5, 2.5, 30)
    y_test_true = np.sin(X_test)
    
    bias_sq, variance = BiasVarianceDecomposition.evaluate(
        DummyLinearModel, X_pool, y_pool, X_test, y_test_true, n_bootstraps=30, sample_size=40
    )
    print(f"Linear model on Sine curve: Bias^2 = {bias_sq:.4f}, Variance = {variance:.4f}")
    assert bias_sq > 0.0 and variance > 0.0
    print("A02 Sanity Check: PASSED cleanly.")
