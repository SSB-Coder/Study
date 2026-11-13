"""
X09. Random Sample Consensus (RANSAC) for Robust Line Fitting
From-scratch implementation in pure NumPy.
"""

import numpy as np

class RANSACLineFitter:
    """
    RANSAC robust linear regression model fitter.
    Estimates 2D line model ax + by + c = 0 in the presence of extreme outlier noise (>50%).
    """
    
    def __init__(self, inlier_threshold: float = 0.5, max_iterations: int = 1000,
                 desired_prob: float = 0.99):
        self.thresh = inlier_threshold
        self.max_iters = max_iterations
        self.p = desired_prob
        self.best_model = None # (a, b, c) normalized
        self.best_inliers = None

    def fit(self, points: np.ndarray):
        """
        points: (N, 2) array of 2D coordinates.
        Returns (a, b, c) normalized line parameters and boolean inlier mask.
        """
        N = len(points)
        best_inlier_count = -1
        best_model = None
        best_mask = None
        
        # Iterative consensus loop <- eq. 3.1
        for it in range(self.max_iters):
            # 1. Randomly sample minimal sample set (s = 2 points for a line)
            idx = np.random.choice(N, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            
            # Line model from 2 points: (y2 - y1)*x - (x2 - x1)*y + (x2*y1 - x1*y2) = 0
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            denom = np.sqrt(dx**2 + dy**2)
            if denom < 1e-8:
                continue
                
            a = dy / denom
            b = -dx / denom
            c = (p2[0] * p1[1] - p1[0] * p2[1]) / denom
            
            # 2. Compute perpendicular distance from all points to candidate line <- eq. 3.2
            # dist = |a*x + b*y + c| / sqrt(a^2 + b^2) = |a*x + b*y + c| since normalized
            dists = np.abs(a * points[:, 0] + b * points[:, 1] + c)
            inlier_mask = dists < self.thresh
            inlier_count = int(np.sum(inlier_mask))
            
            # 3. Consensus test
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_model = (a, b, c)
                best_mask = inlier_mask
                
        # 4. Final Least-Squares re-fit using all consensus inliers <- eq. 3.3
        inlier_pts = points[best_mask]
        # Fit y = m*x + k via OLS on inliers
        X = np.column_stack([inlier_pts[:, 0], np.ones(len(inlier_pts))])
        sol, _, _, _ = np.linalg.lstsq(X, inlier_pts[:, 1], rcond=None)
        m, k = sol[0], sol[1]
        
        # Convert to ax + by + c = 0
        norm = np.sqrt(m**2 + 1.0)
        final_model = (m / norm, -1.0 / norm, k / norm)
        
        self.best_model = final_model
        self.best_inliers = best_mask
        return final_model, best_mask

if __name__ == "__main__":
    print("--- X09: RANSAC Robust Model Fitting Sanity Check ---")
    np.random.seed(42)
    # Generate ground truth line: y = 2.0 * x + 3.0
    N_inliers = 100
    x_in = np.linspace(0.0, 10.0, N_inliers)
    y_in = 2.0 * x_in + 3.0 + np.random.normal(0, 0.1, N_inliers) # 100 clean inliers
    inliers = np.column_stack([x_in, y_in])
    
    # Generate 150 extreme random outliers (60% outlier ratio!)
    N_outliers = 150
    x_out = np.random.uniform(0.0, 10.0, N_outliers)
    y_out = np.random.uniform(-10.0, 30.0, N_outliers)
    outliers = np.column_stack([x_out, y_out])
    
    data = np.vstack([inliers, outliers])
    np.random.shuffle(data)
    
    # 1. Standard OLS without RANSAC (dragged by outliers)
    X_raw = np.column_stack([data[:, 0], np.ones(len(data))])
    ols_raw, _, _, _ = np.linalg.lstsq(X_raw, data[:, 1], rcond=None)
    print(f"Standard OLS with Outliers: slope={ols_raw[0]:.2f}, intercept={ols_raw[1]:.2f} (True: 2.0, 3.0)")
    
    # 2. RANSAC robust fit
    ransac = RANSACLineFitter(inlier_threshold=0.3, max_iterations=500)
    (a, b, c), inlier_mask = ransac.fit(data)
    
    # Recover slope m = -a / b, intercept = -c / b
    m_ransac = -a / b
    k_ransac = -c / b
    print(f"RANSAC Robust Fit: slope={m_ransac:.2f}, intercept={k_ransac:.2f}")
    print(f"Inliers identified: {np.sum(inlier_mask)}/{len(data)}")
    
    # Asserts:
    # 1. RANSAC identifies at least 90% of inliers
    assert np.sum(inlier_mask) >= 90
    # 2. RANSAC recovers true slope and intercept accurately despite 60% outliers
    assert abs(m_ransac - 2.0) < 0.1, "RANSAC failed to recover slope"
    assert abs(k_ransac - 3.0) < 0.3, "RANSAC failed to recover intercept"
    print("X09 Sanity Check: PASSED cleanly.")
