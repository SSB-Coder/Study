# J04: Federated Learning (Federated Averaging - FedAvg)

## 1. The Beginner's Analogy
Imagine a network of hospitals training a diagnostic model to detect rare diseases. Due to strict patient privacy laws (HIPAA/GDPR), hospitals cannot pool their medical records into a centralized cloud server.
Instead, a central server sends a copy of the diagnostic model to each hospital. Each hospital trains the model locally on its private records for a week, and sends back only the *model updates* (weight diffs). The central server averages these weight updates and broadcasts the improved consensus model back to all hospitals. This is **Federated Learning**.

## 2. The Deep Architecture & Mechanics
Federated Learning operates under a client-server architecture with three defining stages:
1. **Model Distribution:** At round $t$, the central server broadcasts global model parameters $w_t$ to a selected subset of active edge clients $\mathcal{S}_t$.
2. **Local Client Computation:** Each client $k \in \mathcal{S}_t$ initializes its local model with $w_t$ and runs $E$ epochs of local stochastic gradient descent over its private dataset $\mathcal{D}_k$ of size $n_k$.
3. **Federated Aggregation:** Clients upload their updated weights $w_{t+1}^k$. The server aggregates them via a weighted coordinate-wise average based on client dataset proportions.

## 3. The Hardcore Mathematics

### Global Optimization Objective
$$\min_w f(w) = \sum_{k=1}^K \frac{n_k}{N} F_k(w)$$
Where:
- $K$: Total number of participating clients.
- $n_k = |\mathcal{D}_k|$: Local sample size of client $k$.
- $N = \sum_k n_k$: Total global sample size across all clients.
- $F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i)$: Local empirical risk.

### FedAvg Aggregation Rule
In round $t$, after each client performs local updates $w_{t+1}^k \leftarrow \text{LocalSGD}(w_t, \mathcal{D}_k, E, \eta)$, the server aggregates:
$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
Equivalently formulated in terms of model updates $\Delta w_t^k = w_{t+1}^k - w_t$:
$$w_{t+1} = w_t + \sum_{k=1}^K \frac{n_k}{N} \Delta w_t^k$$

## 4. The Algorithm Step-by-Step
1. **Initialize Server:** Set initial global parameters $w_0$.
2. **For each communication round $t = 0, 1, \dots, T-1$:**
   - Sample fraction $C$ of clients $\mathcal{S}_t \subseteq \{1, \dots, K\}$.
   - Broadcast $w_t$ to all clients $k \in \mathcal{S}_t$.
   - **On Client $k$:**
     - Initialize local parameters $w \leftarrow w_t$.
     - For local epoch $e = 1 \dots E$: update $w \leftarrow w - \eta \nabla F_k(w)$.
     - Send updated $w_{t+1}^k$ back to server.
   - **On Server:**
     - Aggregate parameters: $w_{t+1} = \sum_{k \in \mathcal{S}_t} \frac{n_k}{\sum_{j} n_j} w_{t+1}^k$.
3. **Output:** Return final global consensus parameters $w_T$.

## 5. Limitations & Failure Modes
- **Client Drift under Non-IID Data:** When local data distributions $P_k(x, y)$ diverge substantially, local updates pull models toward local optima, slowing or preventing global convergence (addressed by FedProx/SCAFFOLD).
- **Communication Bottlenecks:** Uploading massive model weights across slow wireless edge links is expensive (mitigated by LoRA adapters and quantization).

## 6. Where This Fits
Cross-reference: Federated Learning shares mathematical lineage with data-parallel distributed training (All-Reduce gradient averaging). Modern decentralized LLM fine-tuning on edge devices (smartphones, local servers) relies on FedAvg combined with PEFT (Federated LoRA).
