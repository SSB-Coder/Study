# L01: Uninformed Search Strategies (BFS, DFS, UCS, IDS)

## 1. Intuition & The "Grand Analogy"
Imagine searching for a lost key inside a dark, labyrinthine castle. 
- In **Breadth-First Search (BFS)**, you search outward in concentric ripples: first checking every room on your floor, then every adjacent hallway, expanding like an expanding shockwave. You are guaranteed to find the closest key, but your memory swells exponentially as you track every room on the frontier.
- In **Depth-First Search (DFS)**, you pick a single dark corridor and sprint down it until you hit a dead end, backtrack one step, and sprint down the next branch. Your memory footprint is minimal (just the string in your hand), but you might wander down a bottomless staircase forever while the key was sitting 2 feet behind you on the first floor.
- **Iterative Deepening Search (IDS)** gives you the best of both worlds: running depth-limited DFS in expanding waves, achieving the memory efficiency of DFS with the completeness and optimality of BFS.

---

## 2. Architectural & Mathematical Breakdown
A search problem is defined by:
- Initial state $s_0$.
- Action space $\text{Actions}(s)$.
- Transition model $\text{Result}(s, a)$.
- Goal test $\text{IsGoal}(s)$.
- Step cost $c(s, a, s')$.

### Frontier Queuing Disciplines
The general graph search algorithm maintains an **Explored Set** (closed list) and a **Frontier** (open list):
- **BFS:** Frontier is a **FIFO Queue**. Explores shallowest unexpanded nodes.
- **DFS:** Frontier is a **LIFO Stack**. Explores deepest unexpanded nodes.
- **Uniform Cost Search (UCS / Dijkstra):** Frontier is a **Priority Queue** ordered by cumulative path cost $g(n) = \sum c(s_i, a_i, s_{i+1})$. Expands node with lowest path cost.
- **Iterative Deepening Search (IDS):** Calls depth-bounded DFS iteratively for limit $d = 0, 1, 2, \dots$.

---

## 3. Hardcore Mathematical Derivations

### Space and Time Complexity Comparison
Let $b$ be the branching factor, $d$ the depth of the shallowest goal, and $m$ the maximum depth of the state space:
| Algorithm | Time Complexity | Space Complexity | Complete? | Optimal? |
| :--- | :--- | :--- | :--- | :--- |
| **BFS** | $\mathcal{O}(b^d)$ | $\mathcal{O}(b^d)$ | Yes (if $b < \infty$) | Yes (if step costs equal) |
| **DFS** | $\mathcal{O}(b^m)$ | $\mathcal{O}(b \cdot m)$ | No (in infinite graphs) | No |
| **UCS** | $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$ | $\mathcal{O}(b^{1 + \lfloor C^* / \epsilon \rfloor})$ | Yes (if costs $\ge \epsilon > 0$) | Yes (general costs) |
| **IDS** | $\mathcal{O}(b^d)$ | $\mathcal{O}(b \cdot d)$ | Yes | Yes (if step costs equal) |

### Overhead of Iterative Deepening
Although IDS regenerates shallow nodes multiple times, the geometric sum is dominated by the bottom layer:
$$N(\text{IDS}) = \sum_{j=1}^d (d - j + 1) b^j = d \cdot b + (d - 1) b^2 + \dots + 1 \cdot b^d \le \frac{b}{(b - 1)^2} b^d \in \mathcal{O}(b^d)$$
For $b=10$ and $d=5$, the repeated generation overhead is only $\approx 11\%$, making IDS the asymptotically optimal uninformed search when memory is constrained.

---

## 4. Step-by-Step Algorithmic Walkthrough
```
Algorithm: Uniform Cost Search (UCS)
Inputs: Start state s_0, Goal test, Step cost c(s, a, s')
1. Initialize Frontier as Min-PriorityQueue containing (cost=0, node=s_0, path=[s_0])
2. Initialize Explored set = empty
3. While Frontier is not empty:
     pop (cost g, state s, path) with lowest g
     If IsGoal(s):
        return path, cost g
     If s not in Explored:
        add s to Explored
        For each action a in Actions(s):
          s_prime = Result(s, a)
          If s_prime not in Explored:
             new_cost = g + c(s, a, s_prime)
             push (new_cost, s_prime, path + [s_prime]) into Frontier
4. Return Failure
```

---

## 5. Failure Modes & When It Breaks
1. **Exponential Frontier Bloat in BFS:** In complex state spaces (e.g. 15-puzzle, Rubik's cube), memory is exhausted within seconds because the frontier stores the entire perimeter of the tree.
2. **Infinite Cycles in DFS:** Without cycle checking / explored sets, DFS loops endlessly in graphs with cycles.

---

## 6. Where This Fits in Modern LLMs & Transformers
- **Breadth vs Depth in Chain-of-Thought (CoT):** Multi-path decoding strategies in LLMs (e.g. Best-of-$N$, Self-Consistency) are direct manifestations of uninformed search across the token generation tree.
