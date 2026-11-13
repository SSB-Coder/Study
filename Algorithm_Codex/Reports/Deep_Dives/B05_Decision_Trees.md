# Deep Dive: Decision Trees (CART Algorithm & Cost-Complexity Pruning)

## 1. The Beginner's Analogy: The Game of 20 Questions
Imagine playing "20 Questions" to identify an animal:
- At the start, you have hundreds of candidate animals. You don't ask, "Is it a platypus?" on question 1; that provides almost zero information if the answer is "no".
- Instead, you ask: *"Does it have a backbone?"* This divides the entire animal kingdom into two roughly equal, much purer buckets (vertebrates vs. invertebrates).
- Next, on the vertebrate branch, you ask: *"Is it warm-blooded?"*
- A **Decision Tree** is a mathematical 20-Questions player. At every fork in the road, it evaluates every available feature and threshold, picking the single split that maximizes the **purity** of the resulting child buckets.
- **Cost-Complexity Pruning** cuts off over-specific twigs (e.g. "Does it have a scar on its left paw from a Tuesday in March?") that describe only a single training squirrel.

---

## 2. The Deep Architecture/Algorithm Breakdown
Classification and Regression Trees (CART) recursively partition the input space into axis-aligned rectangular regions:
1. **Recursive Binary Splitting:** At each node, search all features $j \in \{1, \dots, d\}$ and all unique split thresholds $t \in \mathbb{R}$ to find the partition $(X_L, y_L)$ and $(X_R, y_R)$ minimizing weighted impurity.
2. **Impurity Criteria:**
   - **Gini Impurity:** $I_G(p) = 1 - \sum_{k=1}^K p_k^2$. Measures the probability that a randomly chosen element would be incorrectly labeled.
   - **Shannon Entropy:** $H(p) = -\sum_{k=1}^K p_k \ln p_k$. Measures information disorder.
   - **Mean Squared Error (Regression):** $\text{MSE} = \frac{1}{N} \sum (y_i - \bar{y})^2$.
3. **Stopping Criteria:** Maximum depth reached, node sample size $< \text{min\_samples\_split}$, or purity reached ($I(p) = 0$).
4. **Minimal Cost-Complexity Pruning ($L_\alpha$):** Defines tree cost function $R_\alpha(T) = R(T) + \alpha |T|$, penalizing the total number of terminal leaf nodes $|T|$.

---

## 3. The Hardcore Mathematics

### 3.1 Information Gain / Impurity Reduction
For a node $D$ split into subsets $D_L$ and $D_R$ by condition $(x_j \le t)$:
$$\Delta I(D, j, t) = I(D) - \left( \frac{|D_L|}{|D|} I(D_L) + \frac{|D_R|}{|D|} I(D_R) \right)$$
CART chooses the optimal split $(j^*, t^*)$ by solving:
$$(j^*, t^*) = \arg\max_{j, t} \Delta I(D, j, t)$$

### 3.2 Gini Impurity vs. Entropy
For class proportions $p_k = \frac{1}{N} \sum_{i \in D} \mathbb{I}(y_i = k)$:
$$I_{\text{Gini}}(D) = 1 - \sum_{k=1}^K p_k^2$$
$$H(D) = -\sum_{k=1}^K p_k \ln(p_k + 10^{-15})$$

### 3.3 Cost-Complexity Pruning
Let $R(T) = \sum_{t \in \text{leaves}(T)} \sum_{i \in R_t} \mathbb{I}(y_i \neq \hat{y}_t)$ be the misclassification error of tree $T$.
The cost-complexity measure:
$$R_\alpha(T) = R(T) + \alpha |T|$$
For an internal node $t$, the effective cost-complexity parameter at which collapsing node $t$ into a leaf becomes optimal:
$$g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$
Where $T_t$ is the subtree rooted at $t$. The node with the smallest $g(t)$ is pruned first as $\alpha$ increases.

---

## 4. The Algorithm Step-by-Step
1. **Find Best Split:** For each feature $j$:
   - Sort feature values.
   - For each adjacent midpoint threshold $t$:
     - Compute child impurities $I(D_L), I(D_R)$ and weighted reduction $\Delta I$.
2. **Create Branch:** Split current node into Left ($x_j \le t$) and Right ($x_j > t$).
3. **Recursion:** If stopping conditions are not met, recurse on Left and Right children.
4. **Leaf Assignment:** When stopped, set node as leaf with majority class vote or target mean.
5. **Post-Pruning:** Evaluate validation loss across pruning sequence $\alpha_0 < \alpha_1 < \dots < \alpha_k$ to select optimal sub-tree.

---

## 5. Limitations / When It Breaks
- **Axis-Aligned Orthogonality:** Decision trees can only split orthogonally to feature axes. A simple diagonal boundary ($x_1 = x_2$) requires an inefficient staircase of dozens of splits.
- **High Variance:** A small change in a single training observation can completely change the root split, cascading down to an entirely different tree structure.
- **Historical Fix:** Random Forests (Module D01) average hundreds of decorrelated trees to eliminate variance; Gradient Boosted Trees (Module D03) sequentially fit residuals.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/20_KAN_Kolmogorov_Arnold.md`. Decision trees partition feature space into piecewise constant steps; Kolmogorov-Arnold Networks (KAN) modernize this by placing smooth, adaptive univariate spline functions along computational edges.
