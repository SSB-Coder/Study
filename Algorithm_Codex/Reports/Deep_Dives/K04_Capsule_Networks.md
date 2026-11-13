# K04: Capsule Networks (Dynamic Routing-by-Agreement)

## 1. The Beginner's Analogy
A standard Convolutional Neural Network (CNN) is like a face detector that uses a loose checklist: *"Contains two eyes? Yes. Contains a nose? Yes. Contains a mouth? Yes. Conclusion: It is a face!"*
Even if Picasso paints an eye on the chin and the mouth on the forehead, max-pooling discards spatial relationships, and the CNN happily classifies it as a face.
**Capsule Networks** replace scalar neurons with **vectors (capsules)**. A capsule's length represents the probability that the entity exists, while its *vector orientation* encodes the entity's exact geometric pose, rotation, scale, and spatial relationship.

## 2. The Deep Architecture & Mechanics
Capsule networks replace max-pooling with iterative dynamic routing:
1. **Vector-Valued Neurons:** Unlike scalar activations, capsules output an entire coordinate vector $v_j$.
2. **Transformation Matrices $W_{ij}$:** Multiply input capsule $u_i$ by transformation matrix $W_{ij}$ to produce prediction vector $\hat{u}_{j|i}$. This predicts where part $i$ expects whole $j$ to be positioned.
3. **Routing-by-Agreement:** If prediction vectors from multiple low-level parts tightly cluster and agree on the position of a high-level entity, their coupling coefficient $c_{ij}$ increases, routing information to that capsule.
4. **Squashing Function:** A non-linear vector squashing function maps short vectors to length near 0 and long vectors to length approaching 1.

## 3. The Hardcore Mathematics

### Vector Squashing Non-Linearity
$$v_j = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|}$$
Properties:
- Direction is strictly preserved: $\frac{v_j}{\|v_j\|} = \frac{s_j}{\|s_j\|}$.
- Norm is compressed into the unit interval: $\|v_j\| \in [0, 1)$.

### Dynamic Routing-by-Agreement Algorithm
Given prediction vectors $\hat{u}_{j|i} = W_{ij} u_i$:
1. Initialize prior routing logits $b_{ij} = 0$.
2. For routing iteration $r = 1 \dots R$:
   - Coupling coefficients:
     $$c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}$$
   - Weighted sum:
     $$s_j = \sum_i c_{ij} \hat{u}_{j|i}$$
   - Squashed output:
     $$v_j = \text{squash}(s_j)$$
   - Agreement update:
     $$b_{ij} \leftarrow b_{ij} + \hat{u}_{j|i} \cdot v_j$$

### Margin Loss for Digit Classification
For class $k$ with target $T_k \in \{0, 1\}$:
$$\mathcal{L}_k = T_k \max(0, m^+ - \|v_k\|)^2 + \lambda (1 - T_k) \max(0, \|v_k\| - m^-)^2$$
Where $m^+ = 0.9$, $m^- = 0.1$, and $\lambda = 0.5$.

## 4. The Algorithm Step-by-Step
1. **Primary Capsules:** Extract low-level feature capsules from convolutional features.
2. **Part-Whole Projections:** Multiply each capsule by learned projection tensor $\hat{u}_{j|i} = W_{ij} u_i$.
3. **Iterative Routing:** Run 3 iterations of dynamic routing to update coupling coefficients $c_{ij}$.
4. **Readout:** Measure vector lengths $\|v_k\|$; the capsule with the largest norm indicates the predicted class.

## 5. Limitations & Failure Modes
- **Computational Routing Bottleneck:** Dynamic routing inner loops cannot be parallelized as easily as standard tensor contractions.
- **Crowded Scenes:** Routing struggles to assign parts when identical visual primitives overlap.

## 6. Where This Fits
Cross-reference: Vector-valued representations and dynamic routing directly inspired Part-Whole Hierarchies and Coordinate Frame Attention in modern Vision Transformers (ViTs).
