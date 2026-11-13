# Deep Dive: X10 — Canny Edge Detection (From Scratch)

## 1. The Beginner's Analogy: Tracing Cartoon Outlines with Ink
Imagine turning a photograph into a coloring book drawing:
- If you simply find where light turns dark, you get fuzzy, blurry borders 5 pixels wide.
- **The Canny Edge Detector** is a master draftsperson that follows five strict rules:
  1. **Blur the photograph** slightly with a soft brush so film grain noise disappears.
  2. **Measure the steepness and direction of the cliff** under every pixel (Sobel gradients).
  3. **Non-Maximum Suppression:** Walk along the ridgeline of the cliff and shave off all surrounding clay, leaving a razor-sharp edge exactly 1 pixel wide!
  4. **Two Thresholds:** Mark bold lines as guaranteed edges, and faint lines as "maybes".
  5. **Hysteresis:** Keep the faint lines only if they connect directly to a bold line, ensuring smooth, unbroken outlines without stray noise dots.

---

## 2. The Deep Architecture/Algorithm Breakdown
John F. Canny (1986) formulated edge detection as an optimal filtering problem under three criteria:
1. **Low Error Rate:** No false edges from noise; no missed genuine edges.
2. **Localization:** Detected edge pixels must be as close as possible to the true edge center.
3. **Single Response:** Only one detector response per true edge (no multiple thick responses).

---

## 3. The Hardcore Mathematics

### 3.1 2D Gaussian Kernel
$$G_\sigma(x, y) = \frac{1}{2\pi \sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

### 3.2 Sobel Operator & Direction Quantization
Convolution with horizontal $K_x$ and vertical $K_y$ masks yields spatial derivatives:
$$G_x = K_x * I, \quad G_y = K_y * I$$
$$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \text{atan2}(G_y, G_x)$$
Orientations $\theta$ are rounded to four primary sectors: $0^\circ$ (horizontal), $45^\circ$ (diagonal), $90^\circ$ (vertical), $135^\circ$ (anti-diagonal).

### 3.3 Non-Maximum Suppression (NMS)
For pixel $(r, c)$ with gradient orientation $\theta$:
$$M_{\text{NMS}}(r, c) = \begin{cases} |G(r, c)| & \text{if } |G(r, c)| \ge \max(|G(p_1)|, |G(p_2)|) \\ 0 & \text{otherwise} \end{cases}$$
where $p_1, p_2$ are the two neighbors along the gradient normal $\theta$.

---

## 4. The Algorithm Step-by-Step
1. **Gaussian Smooth:** Convolve image with $G_\sigma$.
2. **Sobel Gradient:** Compute magnitude $G$ and direction $\theta$.
3. **NMS:** Thin thick gradient ridges to 1-pixel width.
4. **Double Threshold:** Classify pixels into Strong ($>T_{\text{high}}$), Weak ($[T_{\text{low}}, T_{\text{high}}]$), and Suppressed ($<T_{\text{low}}$).
5. **Hysteresis Flood Fill:** Breadth-first search from all strong pixels converts adjacent weak pixels into confirmed edges.

---

## 5. Limitations / When It Breaks
- **Lighting & Contrast Variations:** Fixed static thresholds $(T_h, T_l)$ fail when an image transitions from bright sunshine to dark shadow. Adaptive local thresholding (Otsu's method) is required.

---

## 6. Where This Fits
- **Deep Learning Analogue:** Cross-reference `Transformer/Reports/Deep_Dives/02_CNN_Convolutional.md`. Canny's Sobel and Gaussian kernels are the hand-crafted mathematical feature extractors that early convolutional neural network layers automatically learn from data.
