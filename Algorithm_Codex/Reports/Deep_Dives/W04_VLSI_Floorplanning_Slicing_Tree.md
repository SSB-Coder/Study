# Deep Dive: W04 — VLSI Floorplanning via Slicing Trees

## 1. The Beginner's Analogy: Packing Wooden Blocks into a Suitcase
Imagine packing five bulky electronic devices of fixed rectangular sizes into a suitcase:
- A **Slicing Floorplan** is created by taking a large wooden board and making straight guillotine cuts all the way across: first a vertical cut splitting the board into left and right halves, then horizontal cuts splitting those halves into smaller compartments.
- A **Polish Expression** encodes this sequence of cuts as a compact string of numbers and letters (e.g. `1 2 V 3 H`).
- By shuffling this text string like playing cards and testing different cut combinations with simulated annealing, you discover the tightest possible packing with minimum wasted empty space.

---

## 2. The Deep Architecture/Algorithm Breakdown
Floorplanning establishes the coarse geometric positions of major architectural functional blocks (ALUs, caches, memory controllers) on a silicon die before standard-cell placement:
1. **Slicing vs. Non-Slicing:** Slicing floorplans can be recursively divided by guillotine cuts into smaller rectangles. Non-slicing floorplans (e.g. pinwheel structures) require more general representations like Sequence Pairs or B*-Trees.
2. **Polish Expression Representation:** An expression of $n$ operands (blocks) and $n-1$ operators (`V` for vertical cuts, `H` for horizontal cuts) satisfies the **balloting property**:
   - Number of operands in any prefix strictly exceeds number of operators.
3. **Area Evaluation:** A simple post-fix stack evaluates total bounding box dimensions in $\mathcal{O}(n)$ time.

---

## 3. The Hardcore Mathematics

### 3.1 Guillotine Cut Area Composition
Let two sub-floorplans have bounding dimensions $(w_1, h_1)$ and $(w_2, h_2)$:
- **Vertical Cut (`V`):** Blocks placed side-by-side:
  $$w_{\text{combined}} = w_1 + w_2, \quad h_{\text{combined}} = \max(h_1, h_2)$$
- **Horizontal Cut (`H`):** Blocks stacked vertically:
  $$w_{\text{combined}} = \max(w_1, w_2), \quad h_{\text{combined}} = h_1 + h_2$$

### 3.2 Wong-Liu Neighborhood Moves
To explore the configuration space ergodicly via Simulated Annealing:
- **Move 1 ($M_1$):** Swap two adjacent operands.
- **Move 2 ($M_2$):** Complement an operator sequence (`V` $\leftrightarrow$ `H`).
- **Move 3 ($M_3$):** Swap adjacent operand and operator while preserving the balloting property.

---

## 4. The Algorithm Step-by-Step
1. **Initialize:** Generate a valid Polish expression (e.g. $1, 2, V, 3, H, \dots$).
2. **Annealing Loop:**
   - Perturb expression via Move $M_1, M_2,$ or $M_3$.
   - Evaluate new width $W$ and height $H$ via stack.
   - Compute cost $\mathcal{C} = \alpha (W \cdot H) + (1-\alpha) \text{wirelength}$.
   - Accept with probability $\min(1, \exp(-\Delta \mathcal{C} / T))$.
   - Cool temperature $T \leftarrow \gamma T$.
3. **Output:** Optimal slicing tree and floorplan aspect ratio.

---

## 5. Limitations / When It Breaks
- **Non-Slicing Topologies:** Cannot represent cyclic arrangements where four blocks wrap around a central hole without guillotine cuts. Modern macro floorplanning combines non-slicing B*-trees with RL (AlphaChip).

---

## 6. Where This Fits
- **Google AlphaChip Cross-Reference:** AlphaChip operates at this exact floorplanning stage, replacing simulated annealing on slicing/B*-trees with deep RL graph networks.
