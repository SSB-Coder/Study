"""
B04. Naive Bayes (Gaussian, Multinomial, Bernoulli)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class GaussianNaiveBayesScratch:
    """Gaussian Naive Bayes for continuous features."""
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        d = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, d))
        self.vars_ = np.zeros((n_classes, d))

        for k, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.priors_[k] = len(X_c) / len(X)
            self.means_[k, :] = np.mean(X_c, axis=0) # <- eq. 3.2
            self.vars_[k, :] = np.var(X_c, axis=0) + self.var_smoothing
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_posteriors = []
        for k in range(len(self.classes_)):
            # Log Prior + Sum Log Likelihoods <- eq. 3.1 & 3.2
            log_prior = np.log(self.priors_[k])
            mean = self.means_[k, :]
            var = self.vars_[k, :]
            log_lik = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_posteriors.append(log_prior + log_lik)
        
        log_posteriors = np.vstack(log_posteriors).T
        return self.classes_[np.argmax(log_posteriors, axis=1)]

class MultinomialNaiveBayesScratch:
    """Multinomial Naive Bayes for count/frequency data with Laplace smoothing."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.priors_ = None
        self.feature_probs_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        d = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.feature_probs_ = np.zeros((n_classes, d))

        for k, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.priors_[k] = len(X_c) / len(X)
            # Total counts per feature + alpha smoothing <- eq. 3.3
            counts = np.sum(X_c, axis=0)
            total_count = np.sum(counts)
            self.feature_probs_[k, :] = (counts + self.alpha) / (total_count + self.alpha * d)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        log_probs = np.log(self.feature_probs_)
        log_priors = np.log(self.priors_)
        
        # Log likelihood is dot product of counts with log probabilities
        scores = X @ log_probs.T + log_priors
        return self.classes_[np.argmax(scores, axis=1)]

if __name__ == "__main__":
    print("--- B04: Naive Bayes Sanity Check ---")
    np.random.seed(42)
    # 1. Gaussian Naive Bayes check
    X_g = np.vstack([np.random.normal(-2, 1, (50, 2)), np.random.normal(2, 1, (50, 2))])
    y_g = np.array([0]*50 + [1]*50)
    gnb = GaussianNaiveBayesScratch().fit(X_g, y_g)
    preds_g = gnb.predict(X_g)
    acc_g = np.mean(preds_g == y_g)
    print(f"Gaussian NB Accuracy: {acc_g * 100:.2f}%")
    assert acc_g > 0.90

    # 2. Multinomial Naive Bayes check (word counts)
    # Class 0: topic tech (features 0,1 high); Class 1: topic sports (features 2,3 high)
    X_m = np.array([
        [5, 4, 0, 1],
        [4, 6, 1, 0],
        [6, 5, 0, 0],
        [0, 1, 6, 5],
        [1, 0, 5, 7],
        [0, 0, 4, 6]
    ])
    y_m = np.array([0, 0, 0, 1, 1, 1])
    mnb = MultinomialNaiveBayesScratch(alpha=1.0).fit(X_m, y_m)
    test_doc = np.array([[5, 5, 0, 0], [0, 0, 6, 5]])
    preds_m = mnb.predict(test_doc)
    print(f"Multinomial NB test doc predictions: {preds_m} (Expected: [0, 1])")
    assert np.array_equal(preds_m, [0, 1])
    print("B04 Sanity Check: PASSED cleanly.")
