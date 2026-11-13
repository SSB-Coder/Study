"""
D04. Stacking & Blending Ensembles (Out-Of-Fold Meta-Learning)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class StackingClassifierScratch:
    """Stacking classifier using K-fold out-of-fold feature generation."""
    def __init__(self, base_estimators: list, meta_estimator, k_folds: int = 4):
        self.base_estimators = base_estimators # List of (name, model_cls)
        self.meta_estimator = meta_estimator # Instantiated meta-model
        self.k_folds = k_folds
        self.fitted_base_models_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        m = len(X)
        n_bases = len(self.base_estimators)

        # 1. K-Fold splitting for Out-of-Fold features <- eq. 3.1
        indices = np.arange(m)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.k_folds)

        oof_features = np.zeros((m, n_bases))

        for fold_idx, val_idx in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(m), val_idx)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            for b_idx, (name, model_cls) in enumerate(self.base_estimators):
                model = model_cls()
                model.fit(X_tr, y_tr)
                oof_features[val_idx, b_idx] = model.predict(X_val)

        # 2. Train Meta-Estimator on OOF features <- eq. 3.2
        self.meta_estimator.fit(oof_features, y)

        # 3. Retrain all base models on the FULL dataset
        self.fitted_base_models_ = []
        for name, model_cls in self.base_estimators:
            full_model = model_cls()
            full_model.fit(X, y)
            self.fitted_base_models_.append(full_model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        m = len(X)
        n_bases = len(self.fitted_base_models_)
        meta_features = np.zeros((m, n_bases))

        for b_idx, model in enumerate(self.fitted_base_models_):
            meta_features[:, b_idx] = model.predict(X)

        return self.meta_estimator.predict(meta_features)

if __name__ == "__main__":
    print("--- D04: Stacking Ensembles Sanity Check ---")
    np.random.seed(42)
    # Binary classification problem
    m = 100
    X_stk = np.random.randn(m, 4)
    y_stk = (X_stk[:, 0] + X_stk[:, 1] > 0).astype(int)

    # Simple diverse base models
    class DummyLogistic:
        def fit(self, X, y):
            X_b = np.c_[np.ones(len(X)), X]
            self.w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        def predict(self, X):
            X_b = np.c_[np.ones(len(X)), X]
            return (X_b @ self.w >= 0.5).astype(int)

    class DummyKNN:
        def fit(self, X, y):
            self.X, self.y = X, y
        def predict(self, X):
            preds = []
            for row in X:
                dists = np.sum((self.X - row)**2, axis=1)
                nearest = self.y[np.argsort(dists)[:3]]
                preds.append(int(np.mean(nearest) >= 0.5))
            return np.array(preds)

    base_models = [("logreg", DummyLogistic), ("knn", DummyKNN)]
    meta_model = DummyLogistic()

    stacker = StackingClassifierScratch(base_models, meta_model, k_folds=3).fit(X_stk, y_stk)
    preds = stacker.predict(X_stk)
    acc = np.mean(preds == y_stk)
    print(f"Stacking Ensemble Accuracy: {acc * 100:.2f}%")
    assert acc > 0.85
    print("D04 Sanity Check: PASSED cleanly.")
