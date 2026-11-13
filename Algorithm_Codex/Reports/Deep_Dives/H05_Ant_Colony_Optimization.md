# H07: Ant Colony Optimization (ACO) & Combinatorial Search

## 1. Intuition & The "Grand Analogy"
Imagine an ant colony searching for sugar dropped across a kitchen floor. Initially, individual ants wander randomly, laying down faint scented trails of pheromones as they walk. Ants that accidentally stumble upon the shortest path return to the nest faster than those taking circuitous routes. Because the short path receives round-trips more frequently, its pheromone concentration accumulates much faster. Subsequent ants smell the stronger trail and preferentially follow it, laying down even more pheromones in a **positive feedback loop of stigmergic communication**.

**Ant Colony Optimization (ACO)** (Dorigo, 1992) applies this emergent collective behavior to solve notoriously hard combinatorial graph problems like the Traveling Salesperson Problem (TSP).

---

## 2. Architectural & Mathematical Breakdown
Let $G = (V, E)$ be a complete graph with $N$ vertices (cities) and edge distances $d_{ij}$.
Define heuristic visibility $\eta_{ij} \triangleq \frac{1}{d_{ij}}$.
Each edge $(i, j)$ maintains a dynamic pheromone trail concentration $\tau_{ij}$.

### Stochastic Action Choice Rule
An ant at city $i$ chooses next city $j \in \mathcal{N}_i$ with transition probability:
$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$$
where:
- $\alpha \ge 0$: Relative weight of pheromone memory (exploitation).
- $\beta \ge 1$: Relative weight of heuristic distance visibility (greedy exploration).

### Pheromone Evaporation & Deposition
To prevent infinite accumulation on early suboptimal paths:
$$\tau_{ij} \leftarrow (1 - \rho) \tau_{ij} + \sum_{k=1}^m \Delta \tau_{ij}^k$$
$$\Delta \tau_{ij}^k = \begin{cases} \frac{Q}{L_k} & \text{if ant } k \text{ used edge } (i, j) \\ 0 & \text{otherwise} \end{cases}$$
where $\rho \in (0, 1]$ is the evaporation rate, $L_k$ is the total tour length of ant $k$, and $Q$ is a scaling constant.

---

## 3. Hardcore Mathematical Derivations

### Equilibrium Balance of Pheromone Trails
In steady-state, the rate of pheromone deposition equals the rate of evaporation:
$$(1 - \rho) \tau^* + \frac{m Q}{L^*} = \tau^* \implies \rho \tau^* = \frac{m Q}{L^*}$$
$$\tau^* = \frac{m Q}{\rho L^*}$$
The equilibrium pheromone concentration $\tau^*$ is inversely proportional to tour length $L^*$. Shorter tours establish exponentially higher steady-state pheromone concentration, guaranteeing that the colony converges toward the optimal Hamiltonian cycle.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Ant System for TSP
Inputs: Distance matrix D, m ants, evaporation rho, parameters alpha, beta, Q
1. Initialize pheromone matrix tau_{ij} = tau_0 for all edges
2. For iteration = 1 to MaxIter:
     a. Place m ants at random starting cities
     b. For step = 1 to N-1:
          For each ant k:
            Select next unvisited city j according to probability p_{ij}^k
     c. Close tours: connect each ant's final city back to its start
     d. Calculate tour lengths L_k for all ants
     e. Update best tour found so far
     f. Evaporation: tau_{ij} = (1 - rho) * tau_{ij}
     g. Deposition: For each ant k:
          For each edge (i, j) in tour_k:
             tau_{ij} = tau_{ij} + Q / L_k
3. Return shortest tour and its length
```

---

## 5. Failure Modes & When It Breaks
1. **Pheromone Stagnation:** If evaporation $\rho$ is too small, early random tours deposit persistent pheromones that blind the colony, freezing all ants onto a mediocre sub-tour.
2. **Scale Imbalance ($\alpha$ vs $\beta$):** If $\alpha \gg \beta$, ants ignore distances and blindly follow existing trails. If $\beta \gg \alpha$, ants turn into independent greedy nearest-neighbor heuristics.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Routing in Mixture-of-Experts (MoE):** Allocating tokens across sparse expert networks can be viewed as multi-agent routing where expert routing weights adapt stigmergically.
- **Decoding Graph Navigation:** Exploring constrained beam search paths across formal grammatical transition graphs.
