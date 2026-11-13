# Deep Dive: T03 — Molecular Fingerprints & Chemical Tanimoto Similarity

## 1. The Beginner's Analogy: The Molecular Barcode
Imagine searching a chemical database of 100 million drug compounds:
- You cannot search by comparing 3D atomic coordinates (molecules twist and bend continuously in solution).
- **Circular Fingerprints (ECFP / Morgan)** act like a digital barcode scanner for molecules:
  1. It visits every atom and notes its element (Carbon, Oxygen).
  2. It expands outward in concentric rings (Radius 1, Radius 2), recording local chemical neighborhoods (e.g. "A Carbon bonded to an Oxygen with a single bond").
  3. It hashes every unique neighborhood into a fixed-length 128-bit binary string.
- Two molecules with high **Tanimoto similarity** share identical sub-structures and are highly likely to bind to the same biological disease receptors.

---

## 2. The Deep Architecture/Algorithm Breakdown
David Rogers and Mathew Hahn (2010) formalized Extended-Connectivity Fingerprints (ECFP):
1. **Topological Invariance:** Invariant to atom numbering/indexing, translation, and 3D rotation.
2. **The Morgan Algorithm Foundation:**
   - Adapts the Morgan (1965) algorithm for chemical graph isomorphism.
   - Initial atom invariants encode atomic number, heavy-atom degree, valence, and formal charge.
3. **Circular Neighborhood Hashing:**
   At each radius iteration $r \in \{1, \dots, R\}$:
   - Collect neighbor descriptors $(b_{ij}, h_j)$.
   - Sort neighbors to enforce canonical representation.
   - Hash $(h_i, r, \text{sorted\_neighbors})$.
4. **Folding:**
   Large integer hash values are mapped onto a fixed-width binary vector $\mathbf{v} \in \{0, 1\}^M$ via modulo division ($h \pmod M$).

---

## 3. The Hardcore Mathematics

### 3.1 Initial Atom Invariant
$$h_i^{(0)} = \text{Hash}(Z_i, d_i, v_i)$$
where $Z_i$ is atomic number, $d_i$ is non-hydrogen degree, and $v_i$ is valence.

### 3.2 Circular Hash Recurrence
$$h_i^{(r+1)} = \text{Hash}\left( h_i^{(r)}, r, \text{sort}\left( \{ (b_{ij}, h_j^{(r)}) \mid j \in N(i) \} \right) \right)$$

### 3.3 Tanimoto / Jaccard Similarity Coefficient
For binary bit vectors $\mathbf{a}, \mathbf{b} \in \{0, 1\}^M$:
$$T(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - \mathbf{a} \cdot \mathbf{b}} = \frac{|A \cap B|}{|A \cup B|}$$
In computational chemistry, $T \ge 0.85$ is the standard empirical threshold denoting high probability of shared pharmacological biological activity.

---

## 4. The Algorithm Step-by-Step
1. **Parse Molecular Graph:** Represent atoms as vertices $V$ and covalent bonds as edges $E$.
2. **Compute Base Hashes:** Assign $h_i^{(0)}$ to each atom.
3. **Iterate Radii:** For $r = 1 \dots R$, update atom identifiers by hashing circular neighbors. Collect all hashes into set $\mathcal{S}$.
4. **Bit Folding:** Set bit $h \pmod M = 1$ for all $h \in \mathcal{S}$.
5. **Score:** Compute bitwise AND/OR Tanimoto similarity.

---

## 5. Limitations / When It Breaks
- **Bit Collisions:** Folding millions of distinct chemical subgraphs into a small $M=1024$ bit vector causes hash collisions where unrelated chemical fragments map to the same bit index.
- **Stereochemistry & Chirality:** Basic ECFP ignores 3D spatial chirality ($R/S$ enantiomers that have identical 2D connectivity but opposite biological drug effects).

---

## 6. Where This Fits
- **Deep Learning Analogue:** Modern Molecular Graph Neural Networks (GNNs, SchNet, DimeNet) replace hand-crafted ECFP hash functions with learnable message-passing convolutions across molecular graphs.
