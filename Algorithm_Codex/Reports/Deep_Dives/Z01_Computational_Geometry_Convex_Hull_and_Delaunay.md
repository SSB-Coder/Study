# Z01: Classical Computational Geometry (Convex Hull & Delaunay Triangulation)

## 1. Mathematical Formulation & Architecture
Computational geometry provides the spatial primitives underlying computer graphics, geographic information systems (GIS), scientific meshing (FEM), and spatial proximity search.

### Graham Scan 2D Convex Hull
The **Convex Hull** $\mathcal{CH}(P)$ of a finite point set $P \subset \mathbb{R}^2$ is the smallest convex polygon enclosing all points in $P$:
$$\mathcal{CH}(P) = \left\{ \sum_{i=1}^N \lambda_i p_i \;\middle|\; \lambda_i \ge 0, \sum_{i=1}^N \lambda_i = 1 \right\}$$
The Graham Scan evaluates the orientation of three ordered planar points $p_1, p_2, p_3$ using the signed area 2D cross product:
$$\Delta(p_1, p_2, p_3) = (p_2^x - p_1^x)(p_3^y - p_1^y) - (p_2^y - p_1^y)(p_3^x - p_1^x)$$
- $\Delta > 0$: Strict counter-clockwise turn (left turn).
- $\Delta < 0$: Strict clockwise turn (right turn / non-convex intrusion).
- $\Delta = 0$: Collinear points.

By sorting points by polar angle $\theta = \arctan2(y - y_{\text{pivot}}, x - x_{\text{pivot}})$ relative to the lowest $y$-coordinate pivot, we maintain a monotonic CCW turn stack, popping points that form right turns in $\mathcal{O}(N \log N)$ total time.

### Bowyer-Watson Delaunay Triangulation
A **Delaunay Triangulation** $\mathcal{DT}(P)$ maximizes the minimum angle of all triangles, preventing sliver triangles. Its defining property is the **Empty Circumcircle Criterion**: for every triangle $T \in \mathcal{DT}(P)$, no point in $P$ lies strictly within the circumcircle of $T$.

Testing whether a query point $D = (x_d, y_d)$ lies inside the circumcircle of counter-clockwise oriented vertices $A, B, C$ is computed via the parabolic lift determinant:
$$\det \begin{pmatrix} x_a - x_d & y_a - y_d & (x_a - x_d)^2 + (y_a - y_d)^2 \\ x_b - x_d & y_b - y_d & (x_b - x_d)^2 + (y_b - y_d)^2 \\ x_c - x_d & y_c - y_d & (x_c - x_d)^2 + (y_c - y_d)^2 \end{pmatrix} > 0$$

The **Bowyer-Watson incremental algorithm**:
1. Enclose all points within a large bounding super-triangle.
2. Incrementally insert point $p$:
   - Identify all "bad" triangles whose circumcircle contains $p$.
   - Extract the boundary edges of the union of bad triangles (edges shared by only one bad triangle).
   - Delete bad triangles and connect $p$ to each boundary edge.
3. Remove all triangles sharing vertices with the initial super-triangle.

## 2. Implementation Details
The module `Z01_Computational_Geometry_Convex_Hull_and_Delaunay.py`:
- Implements `ConvexHull2D.compute` with pivot detection, polar angle sorting, collinear edge pruning, and Graham stack maintenance.
- Implements `DelaunayTriangulation2D.compute` via the exact Bowyer-Watson algorithm using vectorized determinant evaluations.
- Evaluates empty circumcircle invariants dynamically across all inserted vertices.

## 3. Asymptotic & Numerical Analysis
- **Convex Hull**: $\mathcal{O}(N \log N)$ time due to initial polar sorting; stack maintenance runs in $\mathcal{O}(N)$ amortized time. Space complexity is $\mathcal{O}(N)$.
- **Bowyer-Watson Triangulation**: $\mathcal{O}(N \log N)$ average time under random spatial insertion; worst-case $\mathcal{O}(N^2)$ for degenerate configurations. Produces $\le 2N - 2 - k$ triangles (Euler's formula).

## 4. Edge Cases & Numerical Stability
- **Collinear Points**: In Graham scan, ties in polar angle must retain only the point farthest from the pivot to avoid zero-area degeneracies.
- **Co-circular Points**: If four points lie on a common circle, the triangulation is non-unique; the determinant evaluates to zero, handled with strict numerical thresholding ($\varepsilon = 10^{-9}$).

## 5. Standalone Verification
The test suite validates:
- Correct extraction of the 4 outer corner vertices of a square enclosing multiple random interior points.
- Generation of complete valid planar triangulations on 20 random spatial points.
- Rigorous verification that **every single triangle satisfies the empty circumcircle property** across all remaining point queries.

## 6. Real-World Applications & Context
Foundational for mesh generation in Finite Element Analysis (FEA), terrain elevation modeling in GIS, spatial neighborhood graph construction in machine learning, and Voronoi territory partitioning for wireless base stations and logistics routing.
