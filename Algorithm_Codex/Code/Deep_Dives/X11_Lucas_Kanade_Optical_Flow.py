"""
X11. Lucas-Kanade Sparse Optical Flow (Pure NumPy, Zero OpenCV)
From-scratch implementation in pure NumPy.
"""

import numpy as np

class LucasKanadeOpticalFlow:
    """
    Computes sparse optical flow velocity vectors [u, v] between two consecutive frames
    using the Lucas-Kanade local window least-squares algorithm.
    """
    
    def __init__(self, window_size: int = 5, eigen_threshold: float = 1e-4):
        self.win = window_size
        self.k = window_size // 2
        self.tau = eigen_threshold

    def _derivatives(self, I1: np.ndarray, I2: np.ndarray):
        """
        Computes spatial derivatives Ix, Iy and temporal derivative It <- eq. 3.1.
        """
        # Central difference spatial gradients on I1
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float) / 8.0
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float) / 8.0
        
        # Convolve
        H, W = I1.shape
        Ix = np.zeros_like(I1)
        Iy = np.zeros_like(I1)
        
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                patch = I1[r-1:r+2, c-1:c+2]
                Ix[r, c] = np.sum(patch * sobel_x)
                Iy[r, c] = np.sum(patch * sobel_y)
                
        # Temporal gradient: It = I2 - I1
        It = I2 - I1
        return Ix, Iy, It

    def compute_flow(self, I1: np.ndarray, I2: np.ndarray, feature_points: list):
        """
        Computes flow (u, v) for each feature point (r, c) <- eq. 3.2.
        Returns list of ((r, c), (u, v)).
        """
        I1 = np.asarray(I1, dtype=float)
        I2 = np.asarray(I2, dtype=float)
        if I1.max() > 1.0: I1 /= 255.0
        if I2.max() > 1.0: I2 /= 255.0
            
        Ix, Iy, It = self._derivatives(I1, I2)
        H, W = I1.shape
        flow_vectors = []
        
        for r, c in feature_points:
            r, c = int(round(r)), int(round(c))
            if r - self.k < 0 or r + self.k >= H or c - self.k < 0 or c + self.k >= W:
                continue
                
            # Extract local window
            ix = Ix[r - self.k : r + self.k + 1, c - self.k : c + self.k + 1].flatten()
            iy = Iy[r - self.k : r + self.k + 1, c - self.k : c + self.k + 1].flatten()
            it = It[r - self.k : r + self.k + 1, c - self.k : c + self.k + 1].flatten()
            
            # Form 2x2 structure tensor A = [Ix, Iy]^T [Ix, Iy] <- eq. 3.3
            A = np.array([
                [np.sum(ix**2), np.sum(ix * iy)],
                [np.sum(ix * iy), np.sum(iy**2)]
            ])
            b = -np.array([np.sum(ix * it), np.sum(iy * it)])
            
            # Check for aperture problem via eigenvalues of A
            evals = np.linalg.eigvals(A)
            if min(evals) < self.tau:
                # Flat or single-edge region -> aperture problem
                continue
                
            uv = np.linalg.solve(A, b)
            flow_vectors.append(((r, c), (uv[0], uv[1])))
            
        return flow_vectors

if __name__ == "__main__":
    print("--- X11: Lucas-Kanade Optical Flow Sanity Check ---")
    np.random.seed(42)
    # Generate Frame 1: 50x50 black image with a bright 2D Gaussian spot centered at (25, 25)
    H, W = 50, 50
    y, x = np.mgrid[0:H, 0:W]
    
    c1_x, c1_y = 25.0, 25.0
    sigma = 4.0
    frame1 = np.exp(-((x - c1_x)**2 + (y - c1_y)**2) / (2.0 * sigma**2))
    
    # Generate Frame 2: Blob shifted by dx = +1.0, dy = +0.5
    true_dx, true_dy = 1.0, 0.5
    frame2 = np.exp(-((x - (c1_x + true_dx))**2 + (y - (c1_y + true_dy))**2) / (2.0 * sigma**2))
    
    lk = LucasKanadeOpticalFlow(window_size=9, eigen_threshold=1e-4)
    # Track points around the high-gradient flank of the Gaussian spot
    track_points = [(23, 23), (25, 22), (22, 25), (27, 27)]
    flow = lk.compute_flow(frame1, frame2, track_points)
    
    print(f"Tracked Points: {len(flow)}")
    for pt, (u, v) in flow:
        print(f"Point {pt}: Estimated Flow (u={u:.3f}, v={v:.3f}) [True: dx={true_dx}, dy={true_dy}]")
        
    assert len(flow) > 0, "Failed to compute optical flow"
    # Average estimated velocity across tracked points
    mean_u = np.mean([uv[0] for _, uv in flow])
    mean_v = np.mean([uv[1] for _, uv in flow])
    print(f"Mean Flow: u={mean_u:.3f}, v={mean_v:.3f}")
    
    assert abs(mean_u - true_dx) < 0.2, "Horizontal flow estimate error too high"
    assert abs(mean_v - true_dy) < 0.2, "Vertical flow estimate error too high"
    print("X11 Sanity Check: PASSED cleanly.")
