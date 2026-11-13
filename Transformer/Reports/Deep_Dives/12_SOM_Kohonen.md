# Deep Dive: Self-Organizing Maps (SOM) / Kohonen Networks

## 1. The Beginner's Analogy: The Magnetic Mesh

Imagine throwing a highly flexible, magnetic fishing net over a massive, chaotic pile of differently shaped metal objects.

At first, the net just drapes randomly. But then you turn on the magnets.
When a point on the net gets close to a heavy metal object, it snaps to it. Crucially, because it is a net, when one point snaps to an object, it pulls its neighboring strings along with it.

Eventually, the net completely covers the pile of objects, mapping its chaotic 3D shape into a smooth, 2D mesh. If two objects are similar and close together in the pile, they will be covered by parts of the net that are close together.

A **Self-Organizing Map (SOM)** is an unsupervised neural network that acts like this magnetic net. It takes highly complex, multidimensional data (like the DNA profiles of 10,000 patients) and maps it down to a 2D grid. It organizes itself so that similar data points are physically grouped together on the grid.

---

## 2. The Deep Architecture Breakdown

Invented by Teuvo Kohonen in the 1980s, SOMs are unique because they do not use error backpropagation. They use **Competitive Learning**.

*   **The Grid:** The network is usually a 2D grid of nodes (e.g., $10 \times 10$).
*   **The Weights:** Every node on the grid has a weight vector of the exact same dimension as the input data. (If inputting 50-dimensional DNA data, each node has a 50D weight vector).
*   **No Hidden Layers:** The input is connected directly to every node on the grid.

### The Competition
When a piece of data is fed into the network, all the nodes compete to see whose weight vector is most similar to the data. The node that is most similar is declared the **Best Matching Unit (BMU)** (the winner).

### The Cooperation and Adaptation
The BMU "wins" the right to adjust its weights to look even *more* like the input data. But it also acts like the magnetic net: it pulls its immediate physical neighbors on the 2D grid closer to the data as well.

---

## 3. The Hardcore Mathematics

Let $x(t)$ be the input vector at time step $t$.
Let $W_v(t)$ be the weight vector of node $v$.

### 3.1 Finding the BMU
We find the node $c$ whose weight vector is closest to the input vector $x$. "Closest" is defined mathematically using the Euclidean Distance metric:

$$ c = \arg \min_v \left\| x(t) - W_v(t) \right\| $$

### 3.2 The Neighborhood Function $\theta(c, v, t)$
Once the BMU $c$ is found, we need to mathematically define how much it influences its neighbors. We use a Gaussian function based on the physical distance on the 2D grid between the BMU $c$ and any other node $v$.

$$ \theta(c, v, t) = \exp \left( - \frac{\| r_c - r_v \|^2}{2 \sigma^2(t)} \right) $$

*   $r_c$ and $r_v$: The physical 2D coordinates of the nodes on the grid (e.g., node $c$ is at `[2, 5]` and node $v$ is at `[2, 6]`).
*   $\sigma(t)$: The "radius" of the neighborhood. Crucially, this radius shrinks over time. At the beginning of training, the BMU pulls the whole net. By the end, it only pulls itself.

### 3.3 The Weight Update Rule
The final formula updates the weights of every node $v$ in the network:

$$ W_v(t+1) = W_v(t) + \theta(c, v, t) \cdot \alpha(t) \cdot [x(t) - W_v(t)] $$

Let's break this down:
*   $[x(t) - W_v(t)]$: The direction from the node's current weight toward the data point.
*   $\alpha(t)$: The learning rate (which also decays/shrinks over time).
*   $\theta(c, v, t)$: The neighborhood function.
    *   If $v$ is the BMU ($v=c$), the distance is 0, $\theta=1$, and the node moves the maximum amount toward the data.
    *   If $v$ is far away from the BMU, $\theta \approx 0$, and the node doesn't move at all.

## 4. Why Use a SOM?
SOMs are primarily used for Data Visualization and Clustering. 
If you have high-dimensional financial data regarding credit card fraud, you cannot visualize it on a graph. By training a SOM, the network organizes itself so that all fraudulent transactions cluster into a specific red corner of a 2D map, allowing humans to easily see patterns in unseeable dimensions.
