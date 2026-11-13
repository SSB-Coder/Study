# Deep Dive: X09 — Random Sample Consensus (RANSAC)

## 1. The Beginner's Analogy: Finding the Laser Beam in Confetti
Imagine a laser beam shooting across a room where someone just popped a party cracker, filling the air with 1,000 fluttering confetti flakes:
- If you use **Standard Ordinary Least Squares (B01)**, every single confetti flake pulls on the estimated line like a rubber band. A single rogue flake in the corner drags the line completely off target.
- **RANSAC** says: *"Ignore the crowd. Pick two random specks of dust, stretch a tight thread between them, and count how many other specks lie along that thread."*
- If the thread was drawn between two random confetti flakes, almost nobody else sits on that line (Consensus = 3).
- But the instant you pick two genuine laser photons, all 100 laser photons snap into alignment along the thread (Consensus = 100)! RANSAC crowns that consensus winner and throws the confetti in the trash.

---

## 2. The Deep Architecture/Algorithm Breakdown
Fischler and Bolles (1981) introduced RANSAC for robust model fitting in computer vision:
1. **Hypothesize-and-Verify:**
   - Instead of fitting all data points and iteratively pruning outliers, RANSAC repeatedly fits candidate models to the **minimum number of data points** ($s$) necessary to define the model (e.g. $s=2$ for a 2D line, $s=4$ for a planar homography, $s=5$ or $s=8$ for epipolar fundamental matrices).
2. **Consensus Score:**
   The quality of model $\Theta$ is the cardinality of its inlier consensus set:
   $$S(\Theta) = \{ (x_i, y_i) \mid \text{dist}((x_i, y_i), \Theta) < \epsilon \}$$
3. **Re-Estimation:**
   Once the winning consensus set is identified, the final model parameters are re-estimated using all consensus inliers via standard least squares.

---

## 3. The Hardcore Mathematics

### 3.1 Stopping Criterion & Iteration Count
Let $w$ be the inlier ratio (fraction of clean data points $w = N_{\text{inliers}} / N$).
The probability that at least one of $k$ random samples contains only inliers (yielding a clean model) is:
$$P(\text{Success}) = 1 - (1 - w^s)^k = p$$
Solving for the required number of iterations $k$:
$$k = \frac{\log(1 - p)}{\log(1 - w^s)}$$
For a line ($s=2$) with $50\%$ outliers ($w=0.5$) and desired confidence $p=0.99$:
$$k = \frac{\log(1 - 0.99)}{\log(1 - 0.5^2)} = \frac{\log(0.01)}{\log(0.75)} \approx 16 \text{ iterations!}$$

---

## 4. The Algorithm Step-by-Step
1. **Sample:** Randomly select $s$ data points.
2. **Fit Hypothesis:** Compute model $\Theta$ passing through sample.
3. **Score:** Count points within distance threshold $\epsilon$.
4. **Update Best:** If $|S(\Theta)| > |S(\Theta^*)|$, set $\Theta^* = \Theta$.
5. **Re-Fit:** Fit final model on consensus inliers via least squares.

---

## 5. Limitations / When It Breaks
- **Threshold Sensitivity:** $\epsilon$ must be tuned to true measurement sensor noise variance $\sigma$. Too tight discards real inliers; too loose admits outliers.

---

## 6. Where This Fits
- **Cross-Reference Linear Regression (B01):** RANSAC exists specifically because OLS has a breakdown point of $0\%$, whereas RANSAC tolerates $>80\%$ outliers.
- **Visual Odometry & Feature Matching (X11):** Filtering false feature descriptor matches in camera SLAM.
