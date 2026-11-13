# Deep Dive: Linear Attention & GLA

## 1. The Sub-Quadratic Dream
Standard Softmax Attention requires calculating the $QK^T$ matrix, which scales at $O(N^2)$. If a sequence is 100,000 tokens long, this matrix requires massive amounts of GPU memory.

Linear Attention attempts to solve this by removing the `softmax` entirely. If you remove softmax, you can change the order of matrix multiplication due to the associative property:
$$ (Q K^T) V = Q (K^T V) $$
If you multiply $K^T \times V$ first, you get a fixed-size state matrix that does not grow with the sequence length. You achieve $O(N)$ linear time!

## 2. The Kernel Trick
You can't just delete softmax, or the model loses its non-linear expressivity and fails. 
Instead of softmax, Linear Attention applies a mathematical "feature map" or "kernel function" $\phi(x)$ to the Queries and Keys individually before multiplying them.
$$ \text{Attention} \approx \phi(Q) (\phi(K)^T V) $$
Models like the **Performer** (FAVOR+) use random Fourier features as the kernel $\phi$ to approximate the softmax mathematically without actually calculating the $N \times N$ matrix.

## 3. Gated Linear Attention (GLA)
Early Linear Attention models were fast but stupid. They performed significantly worse than standard Transformers on language tasks. 

In 2023, **GLA (Gated Linear Attention)** fixed this. The researchers realized that pure linear attention lacked the ability to "forget" irrelevant tokens (it was just a giant running sum).
They introduced **Data-Dependent Gating** (similar to an LSTM or Mamba). They added a "forget gate" vector that decays older keys/values if the current token tells it to.

Because GLA retains the mathematical structure of attention but adds the selective forgetting of an RNN, it achieves $O(N)$ speed while matching the performance of $O(N^2)$ Transformers on many benchmarks.
