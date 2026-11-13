# H08: NEAT (NeuroEvolution of Augmenting Topologies)

## 1. The Beginner's Analogy
Conventional neural network training is like tuning a musical instrument with a fixed number of strings and frets. You adjust the tension (weights), but you cannot add a string or invent a new acoustic chamber.
**NEAT (NeuroEvolution of Augmenting Topologies)** begins with the simplest possible instrument—just two pins and a wire—and allows evolution to add new wires (synaptic connections) and sound posts (hidden nodes) over generations. By starting minimal and complexifying only when beneficial, NEAT discovers both the optimal wiring diagram and the weights simultaneously.

## 2. The Deep Architecture & Mechanics
NEAT solves three fundamental challenges in neuroevolution:
1. **Tracking Historical Markings (Global Innovation Numbers):** How do you cross over two neural network genomes with completely different graph topologies without solving intractable graph isomorphism? NEAT assigns a monotonically increasing global innovation number to every newly arisen gene. Matching innovation numbers represent homologous genes that can be crossed over cleanly.
2. **Speciation via Compatibility Distance:** Structural mutations (adding a new node or connection) are initially disruptive and typically decrease fitness before their weights can be optimized. NEAT groups similar topologies into *species* based on a genetic distance metric, protecting novel topological innovations through fitness sharing.
3. **Complexification from Minimal Structure:** Populations start with minimal connectivity (inputs directly connected to outputs) and incrementally add nodes and connections, ensuring search operates in low-dimensional subspaces first.

## 3. The Hardcore Mathematics

### Genetic Distance & Speciation
The compatibility distance $\delta$ between two genomes $G_1$ and $G_2$ is defined as:
$$\delta = \frac{c_1 E}{N} + \frac{c_2 D}{N} + c_3 \bar{W}$$
Where:
- $E$: Number of **Excess genes** (innovations beyond the maximum innovation of the other genome).
- $D$: Number of **Disjoint genes** (innovations present in one genome but absent in the other, below the cutoff).
- $\bar{W}$: Average absolute weight difference of matching (homologous) connection genes.
- $N$: Total number of genes in the larger genome (normalizer, set to $1$ for small genomes).
- $c_1, c_2, c_3$: Hyperparameters weighting structural vs. parametric divergence.

### Explicit Fitness Sharing
Within species $k$ of size $N_k$, an individual $i$'s raw fitness $f_i$ is shared:
$$f_i' = \frac{f_i}{\sum_{j=1}^{P} \text{sh}(\delta(i, j))}, \quad \text{sh}(\delta) = \begin{cases} 1 & \text{if } \delta < \delta_t \\ 0 & \text{otherwise} \end{cases}$$
Species assign offspring allocations proportionally to their average adjusted fitness:
$$O_k = \frac{\bar{f}_k'}{\sum_{m=1}^{S} \bar{f}_m'} \cdot P_{\text{pop}}$$

## 4. The Algorithm Step-by-Step
1. **Initialize Population:** Generate minimal genomes with direct input-to-output connections and randomized weights.
2. **Evaluate Fitness:** Pass training inputs through each genome's acyclic network and evaluate objective task fitness.
3. **Speciate:** Compute compatibility distance $\delta(i, r_k)$ between genome $i$ and species representatives $r_k$. Assign to existing species if $\delta < \delta_t$, else spawn a new species.
4. **Adjust Fitness:** Apply explicit fitness sharing to penalize overpopulated niches.
5. **Reproduce & Mutate:**
   - Crossover matching genes from fit parents.
   - Mutate connection weights ($80\%$ probability).
   - Mutate topology: Add Connection ($5\%$ probability) or Add Node ($3\%$ probability, splitting an existing connection and generating 2 new innovation numbers).
6. **Repeat:** Continue until task criterion is achieved.

## 5. Limitations & Failure Modes
- **Computational Overhead on Dense Topologies:** Speciation distance comparisons scale quadratically with population size $\mathcal{O}(P^2)$ if species representatives are not efficiently cached.
- **Topological Bloat:** Without explicit parsimony pressure or complexity penalization, topologies can accumulate non-functional "silent" nodes and connections.

## 6. Where This Fits
Cross-reference: NEAT pioneered Neural Architecture Search (NAS) decades before deep learning. In modern LLMs, dynamic expert routing in Mixture-of-Experts (MoE) and sparse neural wiring in Mechanistic Interpretability mirror NEAT's topological sparsification principles.
