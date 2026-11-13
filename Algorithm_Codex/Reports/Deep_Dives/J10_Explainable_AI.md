# J10: Explainable AI (Permutation Importance, SHAP, LIME)

## 1. The Beginner's Analogy
When a bank's AI rejects your mortgage application, telling you *"the neural network output was 0.12"* is unacceptable and legally non-compliant. You need to know: *"Was it my debt-to-income ratio? My credit score? My employment length?"*
**Explainable AI (XAI)** opens the black box. It calculates fair, mathematically rigorous credit assignments showing exactly how much each individual input feature contributed to pushing the model toward its final decision.

## 2. The Deep Architecture & Mechanics
Modern model-agnostic explainability relies on three fundamental techniques:
1. **Permutation Feature Importance (Global):** Measures model sensitivity by breaking the relationship between feature $X_j$ and target $Y$ through column shuffling.
2. **SHAP (SHapley Additive exPlanations):** Grounded in cooperative game theory (Lloyd Shapley, 1953). Treats features as players in a coalition, computing the unique attribution allocation satisfying efficiency, symmetry, dummy, and additivity axioms.
3. **LIME (Local Interpretable Model-agnostic Explanations):** Explains a complex non-linear model locally around instance $x$ by drawing localized perturbations and fitting an interpretable surrogate (linear model) weighted by spatial proximity.

## 3. The Hardcore Mathematics

### Shapley Value Formula
The marginal contribution of feature $i$ across all possible feature subsets $S \subseteq F \setminus \{i\}$:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left( v(S \cup \{i\}) - v(S) \right)$$
Where $v(S) = \mathbb{E}[f(x) \mid x_S]$ evaluates model prediction conditioned on subset $S$.

### The Four Shapley Axioms
1. **Efficiency:** $\sum_{i=1}^D \phi_i(x) = f(x) - \mathbb{E}[f(X)]$.
2. **Symmetry:** If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$.
3. **Dummy (Null Player):** If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$.
4. **Additivity:** For independent models $f + g$, $\phi_i(f + g) = \phi_i(f) + \phi_i(g)$.

### LIME Local Loss Formulation
$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
Where:
- $g(z) = w^T z$: Sparse linear surrogate model.
- $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$: Exponential proximity kernel.
- $\mathcal{L}(f, g, \pi_x) = \sum_{z} \pi_x(z) (f(z) - g(z))^2$: Weighted fidelity loss.

## 4. The Algorithm Step-by-Step
1. **Permutation Importance:** Record baseline loss; shuffle feature column $j$; measure loss deterioration; repeat across all features.
2. **Shapley Computation:** Enumerate all $2^D$ feature subsets; compute combinatorial weights; evaluate marginal values; aggregate weighted sums.
3. **LIME Sampling:** Draw $N$ Gaussian perturbations $z$ around $x$.
4. **Local Evaluation:** Predict $f(z)$ for all perturbations using the black-box model.
5. **Surrogate Fit:** Solve locally weighted least squares $w = (Z^T W Z)^{-1} Z^T W y$ and output weights $w$ as local explanations.

## 5. Limitations & Failure Modes
- **Combinatorial Complexity of SHAP:** Exact Shapley values scale exponentially $\mathcal{O}(2^D)$, requiring sampling approximations (KernelSHAP) for $D > 15$.
- **Out-of-Distribution Perturbations:** Both SHAP and LIME evaluate models on synthetic feature combinations that may violate natural data manifold constraints.

## 6. Where This Fits
Cross-reference: Explainability mechanisms directly connect to Mechanistic Interpretability, activation patching, attention rollout analysis, and circuit tracing in Transformers.
