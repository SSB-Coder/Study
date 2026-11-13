# Deep Dive: Hypothesis Testing Toolkit

## 1. The Beginner's Analogy: The Courtroom Trial
A statistical hypothesis test is designed exactly like a criminal trial:
- **The Null Hypothesis ($H_0$):** The defendant is presumed innocent (no effect, no difference, no relationship).
- **The Alternative Hypothesis ($H_1$):** The defendant is guilty (there is a real effect or difference).
- **The Evidence ($p$-value):** The probability of observing evidence this extreme if the defendant were completely innocent. If the security footage is so damning that it had only a 1-in-a-million chance ($p < 0.05$) of happening by innocent coincidence, we reject the presumption of innocence ($H_0$) and declare guilt.

---

## 2. The Deep Architecture/Algorithm Breakdown
Hypothesis testing provides formal machinery to determine whether empirical patterns are statistically significant or mere artifacts of random sampling noise:
1. **Two-Sample Student's $t$-Test:** Tests whether the population means of two continuous independent groups differ significantly (assumes normality).
2. **Pearson's Chi-Squared ($\chi^2$) Contingency Test:** Tests whether two categorical variables are independent by comparing observed frequencies $O_{ij}$ against expected frequencies $E_{ij}$.
3. **Non-Parametric Permutation Test:** An exact test requiring zero distributional assumptions (normal, Poisson, etc.). Under $H_0$, group labels are interchangeable; the test computes the exact empirical null distribution by repeatedly shuffling group labels.

---

## 3. The Hardcore Mathematics

### 3.1 Two-Sample Pooled $t$-Test
Given samples $X = \{x_1, \dots, x_{n_1}\}$ and $Y = \{y_1, \dots, y_{n_2}\}$ with sample means $\bar{X}, \bar{Y}$ and sample variances $s_1^2, s_2^2$:
Under equal variance assumption, the pooled sample variance:
$$s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}$$
The test statistic:
$$t = \frac{\bar{X} - \bar{Y}}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}, \quad \text{dof} = n_1 + n_2 - 2$$

### 3.2 Pearson's Chi-Squared Contingency Test
For an $R \times C$ contingency table of observed counts $O_{ij}$:
The expected counts under the independence hypothesis $H_0$:
$$E_{ij} = \frac{(\text{Row } i \text{ total}) \times (\text{Column } j \text{ total})}{\text{Grand Total } N}$$
The test statistic:
$$\chi^2 = \sum_{i=1}^R \sum_{j=1}^C \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad \text{dof} = (R-1)(C-1)$$

### 3.3 Exact Permutation Test
Given observed test statistic $T_{\text{obs}} = |\bar{X} - \bar{Y}|$:
Under $H_0$, combine all $N = n_1 + n_2$ observations into a single pool.
For $B$ random permutations:
1. Randomly partition the pool into two pseudo-groups of size $n_1$ and $n_2$.
2. Compute permutation test statistic $T_b = |\bar{X}^*_b - \bar{Y}^*_b|$.
The two-tailed empirical $p$-value:
$$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(T_b \ge T_{\text{obs}})}{B + 1}$$
*(The $+1$ serves as a continuity correction preventing an exact $p=0$).*

---

## 4. The Algorithm Step-by-Step
1. **Formulate Hypotheses:** State $H_0$ (null) and $H_1$ (alternative), set significance threshold $\alpha = 0.05$.
2. **Compute Summary Statistics:** Evaluate means, variances, contingency table margins, or absolute group divergence.
3. **Evaluate Test Statistic:** Apply raw mathematical formulas to compute $t$, $\chi^2$, or permutation divergence.
4. **Determine $p$-value:** For permutation testing, calculate the fraction of shuffled resamples exceeding observed divergence.
5. **Decision:** If $p < \alpha$, reject $H_0$; otherwise, fail to reject $H_0$.

---

## 5. Limitations / When It Breaks
- **$p$-Hacking & Multiple Testing:** Running 20 tests at $\alpha = 0.05$ on pure noise yields a $1 - (0.95)^{20} \approx 64.2\%$ chance of at least one false positive (Type I error). Requires Bonferroni or Benjamini-Hochberg FDR correction.
- **Sample Size Distortion:** With $m = 10^7$, even microscopic, scientifically meaningless differences produce $p < 10^{-15}$.
- **Historical Fix:** Effect size reporting (Cohen's $d$, odds ratios) alongside $p$-values.

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/28_The_Emergence_of_Intelligence_and_Evaluations.md`. Statistical hypothesis testing is required when benchmarking LLM benchmark performance (e.g. MMLU, GSM8K) to ensure benchmark gains are statistically significant and not noise.
