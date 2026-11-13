# M03: Space Partitioning Trees (k-d Tree & Ball Tree)

## 1. The Beginner's Analogy
Suppose you lose your keys in a 10-room mansion. An exhaustive search checks every square inch of every room ($O(N)$).
A **space-partitioning tree** acts like a master blueprint of the mansion:
- It draws a dividing line across the hallway: "Are the keys on the East wing or West wing?"
- If you know you only visited the East wing, you eliminate the entire West wing with one single decision.
- Inside the East wing, it divides: "Upstairs or Downstairs?"
- With each question, you cut the search space in half.
- If you are searching near a doorway and your current closest key candidate is closer than the distance to the opposite room's door, you can mathematically prove that you don't even need to open that door!

## 2. The Deep Architecture & Mechanics
Space-partitioning trees organize points into hierarchical geometric bounding volumes:
1. **$k$-d Tree (k-Dimensional Tree):**
   - Partitions space using **axis-aligned orthogonal hyperplanes**.
   - At depth $m$, space is split along coordinate axis $d = m \bmod D$.
   - The split point is chosen as the median of points along axis $d$, ensuring a balanced binary tree of height $O(\log N)$.
2. **Ball Tree (Metric Tree):**
   - Replaces axis-aligned boxes with **metric hyperspheres** defined by a centroid $c \in \mathbb{R}^D$ and radius $r = \max_{x \in S} \|x - c\|_2$.
   - Uses the **triangle inequality** to prune subtrees:
     If $|d(q, c) - r| \ge d_{\text{best}}$, no point inside ball $B(c, r)$ can be closer to query $q$ than the current candidate.

## 3. The Hardcore Mathematics

### $k$-d Tree Branch-and-Bound Pruning
Let query point be $q \in \mathbb{R}^D$ and current nearest neighbor candidate have distance $r^* = d(q, x^*)$.
At node $N$ splitting along axis $d$ at coordinate value $v$:
1. Search the child containing $q$ first:
   $$\text{near\_child} = \begin{cases} \text{left} & \text{if } q_d \le v \\ \text{right} & \text{otherwise} \end{cases}$$
2. The distance from $q$ to the splitting hyperplane is simply:
   $$d_{\text{plane}} = |q_d - v|$$
3. **Pruning Condition:**
   If $d_{\text{plane}} \ge r^*$, the hypersphere of radius $r^*$ centered at $q$ does not intersect the opposite half-space. The entire opposite subtree is pruned without visiting!

### Ball Tree Triangle Inequality Pruning
Let a ball node have centroid $c$ and radius $r$. For any point $x$ inside the ball:
$$d(x, c) \le r$$
By the reverse triangle inequality:
$$d(q, x) \ge |d(q, c) - d(x, c)| \ge d(q, c) - r$$
Therefore, the minimum possible distance from query $q$ to any point in the ball is:
$$d_{\min}(q, B(c, r)) = \max(0, d(q, c) - r)$$
**Pruning Condition:**
If $d(q, c) - r \ge r^*$, prune the entire ball.

## 4. The Algorithm Step-by-Step
1. **$k$-d Tree Build:**
   - Base case: if points $\le$ leaf_size, return leaf.
   - Choose axis $d = \text{depth} \bmod D$.
   - Sort indices along axis $d$ and find median index.
   - Recursively construct left and right subtrees.
2. **Nearest Neighbor Search:**
   - Recursively descend to the target leaf node and update $r^*$.
   - Unwind the recursion back up the tree.
   - At each ancestor node, evaluate the pruning condition against the splitting hyperplane or ball boundary.
   - If not pruned, recursively search the opposite sibling.

## 5. Limitations & Failure Modes
- **Curse of Dimensionality in $k$-d Trees:** In high dimensions ($D > 20$), the volume of high-dimensional spheres concentrates near the surface. A ball of radius $r^*$ almost always intersects the splitting hyperplanes in all dimensions, causing the algorithm to degrade to an exhaustive $O(N)$ scan.
- **Ball Tree Advantages in Higher Dimensions:** Ball trees handle correlated manifolds and non-Euclidean metric spaces much better than axis-aligned $k$-d trees, but still suffer when intrinsic dimensionality approaches extrinsic ambient dimensionality.

## 6. Where This Fits
Cross-reference: Space-partitioning trees provide exact nearest neighbor retrieval with logarithmic expected query time for low-to-medium dimensions ($D \le 20$), contrasting with approximate graph methods (HNSW, M01) and hash methods (LSH, M02).
