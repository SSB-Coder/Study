# L05: Fuzzy Logic Systems (Mamdani Inference)

## 1. The Beginner's Analogy
In classical binary logic, a room is either "cold" (0) or "hot" (1). If you define cold as below 20°C, then 19.9°C is freezing and 20.0°C is instantly warm—a brittle and unrealistic cliff.
In real life, temperature is continuous. At 19.5°C, the room is *partially cold* (0.3) and *partially comfortable* (0.7).
**Fuzzy Logic** replaces crisp 0/1 binary truths with continuous degrees of membership in $[0, 1]$, enabling automated control systems (like anti-lock braking or camera autofocus) to reason smoothly with linguistic terms like "slightly warm" or "very fast."

## 2. The Deep Architecture & Mechanics
Mamdani Fuzzy Inference proceeds through five sequential processing stages:
1. **Fuzzification:** Maps crisp scalar numerical inputs to continuous membership degrees $\mu_A(x) \in [0, 1]$ using geometric membership curves (triangular, trapezoidal, Gaussian).
2. **Rule Base:** Linguistic production rules of the form:
   $$\text{IF } x_1 \text{ is } A_1 \text{ AND } x_2 \text{ is } A_2 \text{ THEN } y \text{ is } B$$
3. **Rule Evaluation (T-Norm / S-Norm):** Evaluates antecedent firing strength $\alpha_k$ via min-conjunction (AND) or max-disjunction (OR).
4. **Implication:** Clips or scales consequent output fuzzy sets at degree $\alpha_k$.
5. **Aggregation & Defuzzification:** Combines all clipped output sets via pointwise maximum and computes the **Center of Gravity (Centroid)** to return a crisp output scalar.

## 3. The Hardcore Mathematics

### Triangular Membership Function
$$\text{trimf}(x; a, b, c) = \begin{cases} 0 & x \le a \\ \frac{x - a}{b - a} & a < x \le b \\ \frac{c - x}{c - b} & b < x < c \\ 0 & x \ge c \end{cases}$$

### Fuzzy Set Operations
- T-Norm (Intersection / Min): $\mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))$.
- S-Norm (Union / Max): $\mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))$.
- Complement (Negation): $\mu_{\neg A}(x) = 1 - \mu_A(x)$.

### Centroid (Center of Gravity) Defuzzification
The continuous centroid of the aggregated fuzzy membership function $\mu_{\text{agg}}(y)$:
$$y^* = \frac{\int_{Y} y \cdot \mu_{\text{agg}}(y) \, dy}{\int_{Y} \mu_{\text{agg}}(y) \, dy}$$
Discretized over a finite evaluation grid $\{y_1, \dots, y_M\}$:
$$y^* = \frac{\sum_{m=1}^M y_m \cdot \mu_{\text{agg}}(y_m)}{\sum_{m=1}^M \mu_{\text{agg}}(y_m)}$$

## 4. The Algorithm Step-by-Step
1. **Define Universes of Discourse:** Establish domains for all input and output variables.
2. **Fuzzify Inputs:** Compute membership values $\mu_{A_i}(x_i)$ for all linguistic terms.
3. **Evaluate Antecedents:** For each rule $k$, compute firing strength $\alpha_k = \min_j \mu_{A_{kj}}(x_j)$.
4. **Apply Implication:** Clip each rule's output membership function $\mu_{B_k}'(y) = \min(\alpha_k, \mu_{B_k}(y))$.
5. **Aggregate Sets:** Form global envelope $\mu_{\text{agg}}(y) = \max_k \mu_{B_k}'(y)$.
6. **Defuzzify:** Compute centroid $y^*$ to yield final control action.

## 5. Limitations & Failure Modes
- **Curse of Dimensionality:** If each of $D$ input variables has $M$ linguistic sets, a complete rule base requires $M^D$ rules.
- **Subjective Heuristic Tuning:** Shape parameters of membership curves require manual domain expertise or neuro-fuzzy training (ANFIS).

## 6. Where This Fits
Cross-reference: Fuzzy logic represents the non-probabilistic ancestor of Softmax Attention. Attention weights in Transformers dynamically compute continuous degrees of relevance over key-value memories.
