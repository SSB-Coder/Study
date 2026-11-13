# J07: Causal Inference (SCMs, Propensity Score Matching, DiD)

## 1. The Beginner's Analogy
Correlation is not causation. Imagine observing that people who carry lighters in their pockets have a vastly higher incidence of lung cancer. A naive machine learning model would predict that confiscating lighters cures cancer. In reality, smoking is a **confounding variable** that causes both lighter-carrying and cancer.
**Causal Inference** provides mathematical tools to answer counterfactual questions: *"What would have happened to patient $Y$ had we administered treatment $T$ instead of control, holding all background confounders fixed?"*

## 2. The Deep Architecture & Mechanics
Causal analysis operates under two primary mathematical frameworks:
1. **Structural Causal Models (SCM & DAGs):** Directed Acyclic Graphs where edges represent physical causal mechanisms $Y := f(X, U)$. Pearl's $do(\cdot)$ calculus distinguishes observational conditioning $P(Y \mid X=x)$ from interventional manipulation $P(Y \mid do(X=x))$.
2. **Potential Outcomes Framework (Rubin):** Each unit has two potential outcomes $Y(1)$ (under treatment) and $Y(0)$ (under control). Because we can only observe one in reality (the Fundamental Problem of Causal Inference), we estimate the **Average Treatment Effect (ATE)**:
   $$\text{ATE} = \mathbb{E}[Y(1) - Y(0)]$$

### Propensity Score Matching (PSM)
Rosenbaum & Rubin proven theorem: conditioning on the scalar propensity score $e(X) = P(T=1 \mid X)$ eliminates all confounding bias from observed covariates $X$.

### Difference-in-Differences (DiD)
Exploits longitudinal panel data to eliminate unobserved, time-invariant confounders by comparing temporal changes in treated units against temporal changes in control units under the **Parallel Trends Assumption**.

## 3. The Hardcore Mathematics

### Average Treatment Effect via Propensity Matching
Given propensity score $e(x) = P(T = 1 \mid X = x)$, conditional exchangeability holds:
$$(Y(1), Y(0)) \perp T \mid e(X)$$
For each treated unit $i$ ($T_i = 1$), identify matched control $j(i) = \arg\min_{j: T_j = 0} |e(X_i) - e(X_j)|$:
$$\widehat{\text{ATE}} = \frac{1}{N_1} \sum_{i: T_i = 1} (Y_i - Y_{j(i)})$$

### Difference-in-Differences Formulation
Under standard 2x2 panel regression:
$$Y_{it} = \beta_0 + \beta_1 \text{Treated}_i + \beta_2 \text{Post}_t + \beta_3 (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}$$
The double difference isolates the causal parameter $\beta_3$:
$$\hat{\beta}_3 = (\bar{Y}_{\text{post}, \text{treated}} - \bar{Y}_{\text{pre}, \text{treated}}) - (\bar{Y}_{\text{post}, \text{control}} - \bar{Y}_{\text{pre}, \text{control}})$$

## 4. The Algorithm Step-by-Step
1. **Model Assignment:** Fit logistic regression on observed covariates $X$ to estimate propensity scores $\hat{e}(X)$.
2. **Check Overlap:** Verify common support $\hat{e}(X) \in (0, 1)$ across both treated and control populations.
3. **Match Units:** Pair each treated unit with its nearest control unit using a KD-tree or 1-NN linear scan over $\hat{e}(X)$.
4. **Compute ATE:** Compute average difference between matched pairs.
5. **Sensitivity Check:** Assess sensitivity to hidden unobserved confounders (Rosenbaum bounds).

## 5. Limitations & Failure Modes
- **Unobserved Confounders:** PSM only controls for *measured* covariates; hidden unobserved confounders continue to bias treatment estimates.
- **Violation of Parallel Trends:** If treated and control groups would have evolved along different trajectories in the absence of treatment, DiD produces spurious causal estimates.

## 6. Where This Fits
Cross-reference: Causal inference directly governs offline RL policy evaluation, counterfactual prompt editing, and LLM safety evaluations (distinguishing correlation from causal hallucination).
