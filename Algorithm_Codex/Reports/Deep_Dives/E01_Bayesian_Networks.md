# Deep Dive: Bayesian Networks & Variable Elimination

## 1. The Beginner's Analogy: The Smoke Alarm and the Burnt Toast
Imagine a household safety system with interrelated events:
- *Burglary ($B$)* or *Earthquake ($E$)* can trigger your *House Alarm ($A$)*.
- If the alarm rings, your neighbor *John ($J$)* might call you, or your neighbor *Mary ($M$)* might call you.
- If Mary calls saying the alarm is blaring, how likely is it that there was an actual burglary?
- If the radio immediately announces an Earthquake just occurred, your suspicion of a burglary drops dramatically (the earthquake "explains away" the alarm).
- **A Bayesian Network** draws these causal arrows as a Directed Acyclic Graph (DAG), decomposing the terrifying joint probability of all $2^5 = 32$ combinations into small, manageable conditional probability tables (CPTs).
- **Variable Elimination** answers queries by systematically summing out (eliminating) irrelevant hidden variables one by one like algebraic factoring: $a \cdot b + a \cdot c = a(b + c)$.

---

## 2. The Deep Architecture/Algorithm Breakdown
Bayesian Networks represent joint probability distributions over random variables $X = \{X_1, \dots, X_n\}$ using a DAG $G = (V, E)$:
1. **Chain Rule of Bayesian Networks:** The joint distribution factorizes into the product of local conditional distributions given parent nodes:
   $$P(X_1, \dots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))$$
2. **Conditional Independence ($d$-Separation):** A node is conditionally independent of its non-descendants given its parents. Its Markov blanket consists of its parents, children, and spouses (other parents of its children).
3. **Factors $\phi(Y)$:** Multi-dimensional arrays over variable subsets $Y \subseteq X$.
4. **Factor Operations:**
   - **Factor Product (Pointwise Multiplication):** $\psi(Y \cup Z) = \phi_1(Y) \cdot \phi_2(Z)$.
   - **Marginalization (Summing Out):** $\tau(Y \setminus \{X_j\}) = \sum_{x_j} \phi(Y)$.
5. **Exact Inference via Variable Elimination:** Answers queries $P(Q \mid E=e)$ by restricting factors to observed evidence $E=e$, multiplying relevant factors, and summing out unobserved nuisance variables in an optimal topological order.

---

## 3. The Hardcore Mathematics

### 3.1 Variable Elimination Derivation
Given query variable $Q$, evidence variables $E = e$, and hidden variables $H = \{H_1, \dots, H_k\}$:
The posterior distribution is:
$$P(Q \mid E=e) = \frac{P(Q, E=e)}{P(E=e)} = \frac{1}{Z} \sum_{h_1} \dots \sum_{h_k} \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))$$
Using the distributive law of multiplication over addition, we push summations inside factor products:
$$\sum_{h_1} \left( \phi_1(h_1, \dots) \sum_{h_2} \left( \phi_2(h_2, \dots) \dots \right) \right)$$
Each step eliminates variable $H_j$:
1. Collect all factors $\Phi_{H_j}$ containing variable $H_j$.
2. Multiply factors: $\psi = \prod_{\phi \in \Phi_{H_j}} \phi$.
3. Sum out $H_j$: $\tau = \sum_{h_j} \psi$.
4. Replace $\Phi_{H_j}$ in active factor pool with new factor $\tau$.

---

## 4. The Algorithm Step-by-Step
1. **Instantiate Evidence:** Filter and slice initial CPT factors to fix observed evidence variables $E = e$.
2. **Determine Elimination Ordering:** Order hidden variables $H$ (e.g. min-neighbors or topological ordering).
3. **Execute Elimination:** For each variable $h \in H$:
   - Remove factors containing $h$ from the active pool.
   - Compute their pointwise product.
   - Marginalize (sum out) $h$ from the resulting product factor.
   - Insert new marginalized factor back into pool.
4. **Normalize Query:** Multiply remaining factors containing query $Q$ and normalize so $\sum_q P(Q=q \mid E=e) = 1$.

---

## 5. Limitations / When It Breaks
- **Treewidth & Exponential Complexity:** The time and space complexity is $O(n \cdot d^{w+1})$, where $w$ is the **induced treewidth** of the graph under the chosen elimination ordering. For densely connected cyclic graphs, treewidth $w \approx n$, making exact inference NP-hard.
- **Historical Fix:** Approximate inference: Loopy Belief Propagation, Markov Chain Monte Carlo (Module E03), and Variational Inference (Module E04).

---

## 6. Where This Fits
Cross-reference with `Transformer/Reports/Deep_Dives/19_Neurosymbolic_AI_Hybrids.md`. Bayesian networks provide the probabilistic reasoning foundation for neurosymbolic architectures that combine symbolic dependency graphs with neural perception backbones.
