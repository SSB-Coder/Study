# J03: Meta-Learning (Model-Agnostic Meta-Learning - MAML)

## 1. The Beginner's Analogy
Standard machine learning is like preparing a student to take one specific exam by drilling thousands of past questions.
**Meta-Learning ("Learning to Learn")** is like teaching the student *how to study effectively*. Instead of mastering a single subject, MAML finds an optimal baseline mental state (weight initialization) from which the student can master any new, unfamiliar subject in just 10 minutes of review (a single gradient step on 5 examples).

## 2. The Deep Architecture & Mechanics
MAML optimizes for task-adaptability through a bilevel optimization loop:
1. **Task Distribution $p(\mathcal{T})$:** Tasks are drawn from a broad distribution (e.g., sine waves with varying amplitudes and phases).
2. **Inner Loop (Task Adaptation):** For task $\mathcal{T}_i$, the model computes gradients on a small **support set** $\mathcal{D}_i^{\text{supp}}$ and performs task-specific parameter updates:
   $$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta, \mathcal{D}_i^{\text{supp}})$$
3. **Outer Loop (Meta-Optimization):** The adapted parameters $\theta_i'$ are evaluated on the **query set** $\mathcal{D}_i^{\text{query}}$. The outer loop updates the initial meta-parameters $\theta$ by differentiating through the inner loop trajectory.

## 3. The Hardcore Mathematics

### The Meta-Objective
$$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}\left( \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta, \mathcal{D}_i^{\text{supp}}), \mathcal{D}_i^{\text{query}} \right)$$

### Meta-Gradient Derivation
By the chain rule, the gradient of the meta-objective with respect to initial weights $\theta$ is:
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta_i') = \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}(\theta_i') \cdot \frac{\partial \theta_i'}{\partial \theta} = \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}(\theta_i') \left( I - \alpha \nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}(\theta, \mathcal{D}_i^{\text{supp}}) \right)$$
Notice the second-order Hessian $\nabla_\theta^2 \mathcal{L}_{\mathcal{T}_i}$ term.

### First-Order MAML (FOMAML)
Computing exact Hessians is computationally intensive. FOMAML sets the Hessian factor to zero $\frac{\partial \theta_i'}{\partial \theta} \approx I$, evaluating the gradient directly at the adapted parameters:
$$\nabla_\theta J(\theta) \approx \sum_{\mathcal{T}_i} \nabla_{\theta_i'} \mathcal{L}_{\mathcal{T}_i}(\theta_i', \mathcal{D}_i^{\text{query}})$$
Meta-parameter update rule:
$$\theta \leftarrow \theta - \beta \nabla_\theta J(\theta)$$

## 4. The Algorithm Step-by-Step
1. **Initialize Meta-Parameters:** Initialize base network weights $\theta$.
2. **Sample Task Batch:** Draw a minibatch of tasks $\{\mathcal{T}_1, \dots, \mathcal{T}_K\}$.
3. **Inner Loop Adaptation:** For each task, evaluate support loss $\mathcal{L}_{\text{supp}}$ and compute adapted parameters $\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{supp}}$.
4. **Query Evaluation:** Compute query loss $\mathcal{L}_{\text{query}}$ using adapted parameters $\theta_i'$.
5. **Meta-Gradient Accumulation:** Accumulate gradients of query loss across all tasks.
6. **Meta-Update:** Update base parameters $\theta \leftarrow \theta - \beta \sum_i \nabla \mathcal{L}_{\text{query}}$.
7. **Deploy:** At test time, adapt $\theta$ on a new task using 1 to 5 gradient steps.

## 5. Limitations & Failure Modes
- **Inner-Loop Gradient Vanishing/Explosion:** Deep networks require gradient clipping and tuned inner learning rates $\alpha$.
- **Task Distribution Shift:** Fails when test tasks do not share the underlying inductive structure of training tasks.

## 6. Where This Fits
Cross-reference: In-context learning (ICL) in Transformers performs emergent meta-learning entirely within the forward pass activations without updating weights. MAML represents the explicit gradient-based precursor to this capability.
