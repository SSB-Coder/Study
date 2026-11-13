"""
X10. Canny Edge Detection Pipeline (Pure NumPy, Zero OpenCV)
From-scratch implementation in pure NumPy.
"""

import numpy as np
from collections import deque

class CannyEdgeDetector:
    """
    Complete 5-stage Canny Edge Detection Pipeline in pure NumPy:
    1. Gaussian Blur smoothing
    2. Sobel Gradient magnitude and orientation
    3. Non-Maximum Suppression (NMS) along gradient directions
    4. Double Thresholding (strong, weak, non-edge)
    5. Hysteresis Edge Tracking
    """
    
    def __init__(self, low_threshold: float = 0.1, high_threshold: float = 0.3, sigma: float = 1.0):
        self.low_th = low_threshold
        self.high_th = high_threshold
        self.sigma = sigma

    def _gaussian_kernel(self, size: int = 5) -> np.ndarray:
        """2D Gaussian convolution kernel <- eq. 3.1."""
        k = size // 2
        y, x = np.mgrid[-k:k+1, -k:k+1]
        g = np.exp(-(x**2 + y**2) / (2.0 * self.sigma**2))
        return g / np.sum(g)

    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Zero-padded 2D spatial convolution in pure NumPy."""
        ih, iw = img.shape
        kh, kw = kernel.shape
        pad_h = kh // 2
        pad_w = kw // 2
        
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out = np.zeros_like(img, dtype=float)
        
        for r in range(ih):
            for c in range(iw):
                patch = padded[r:r+kh, c:c+kw]
                out[r, c] = np.sum(patch * kernel)
        return out

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the full 5-stage Canny pipeline on a normalized grayscale image in [0, 1].
        Returns binary edge mask (1 for edge, 0 for background).
        """
        img = np.array(image, dtype=float)
        if img.max() > 1.0:
            img = img / 255.0
            
        # Stage 1: Gaussian Blur <- eq. 3.1
        kernel_g = self._gaussian_kernel(size=5)
        blurred = self._convolve2d(img, kernel_g)
        
        # Stage 2: Sobel Gradients <- eq. 3.2
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
        
        Gx = self._convolve2d(blurred, sobel_x)
        Gy = self._convolve2d(blurred, sobel_y)
        
        G = np.sqrt(Gx**2 + Gy**2)
        if G.max() > 0:
            G = G / G.max() # Normalize gradients to [0, 1]
            
        theta = np.arctan2(Gy, Gx) # Angle in [-pi, pi]
        theta_deg = np.rad2deg(theta) % 180.0
        
        # Stage 3: Non-Maximum Suppression (NMS) <- eq. 3.3
        H, W = G.shape
        nms = np.zeros_like(G)
        
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                ang = theta_deg[r, c]
                # 0 degrees: East-West neighbors
                if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                    p1 = G[r, c + 1]
                    p2 = G[r, c - 1]
                # 45 degrees: North-East / South-West neighbors
                elif 22.5 <= ang < 67.5:
                    p1 = G[r - 1, c + 1]
                    p2 = G[r + 1, c - 1]
                # 90 degrees: North-South neighbors
                elif 67.5 <= ang < 112.5:
                    p1 = G[r - 1, c]
                    p2 = G[r + 1, c]
                # 135 degrees: North-West / South-East neighbors
                else:
                    p1 = G[r - 1, c - 1]
                    p2 = G[r + 1, c + 1]
                    
                if G[r, c] >= p1 and G[r, c] >= p2:
                    nms[r, c] = G[r, c]
                    
        # Stage 4: Double Thresholding
        STRONG = 1.0
        WEAK = 0.5
        thresh_img = np.zeros_like(nms)
        strong_row, strong_col = np.where(nms >= self.high_th)
        weak_row, weak_col = np.where((nms >= self.low_th) & (nms < self.high_th))
        
        thresh_img[strong_row, strong_col] = STRONG
        thresh_img[weak_row, weak_col] = WEAK
        
        # Stage 5: Edge Tracking by Hysteresis
        edges = np.zeros_like(thresh_img)
        edges[thresh_img == STRONG] = 1.0
        queue = deque(zip(strong_row, strong_col))
        
        while queue:
            cr, cc = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if thresh_img[nr, nc] == WEAK and edges[nr, nc] == 0.0:
                            edges[nr, nc] = 1.0
                            queue.append((nr, nc))
                            
        return edges

if __name__ == "__main__":
    print("--- X10: Canny Edge Detection Sanity Check ---")
    # Generate synthetic 40x40 test image with a high-contrast white square in center
    img = np.zeros((40, 40), dtype=float)
    img[10:30, 10:30] = 1.0 # White square on black background
    
    canny = CannyEdgeDetector(low_threshold=0.15, high_threshold=0.3, sigma=1.0)
    edges = canny.detect_edges(img)
    
    edge_pixel_count = int(np.sum(edges))
    print(f"Total Edge Pixels Detected: {edge_pixel_count}")
    
    # A 20x20 square has 4 edges of length ~20 -> ~80-120 perimeter pixels
    assert 60 <= edge_pixel_count <= 150, f"Unexpected edge count: {edge_pixel_count}"
    
    # Assert interior is empty (edges are thin boundaries, not solid fills)
    assert edges[20, 20] == 0.0, "Square interior must be non-edge"
    # Assert exterior background is empty
    assert edges[2, 2] == 0.0, "Background must be non-edge"
    # Assert square top boundary is detected
    assert edges[9, 15] == 1.0, "Square top border must be detected as edge"
    print("X10 Sanity Check: PASSED cleanly.")
