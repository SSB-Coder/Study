# Deep Dive: Positional Encodings & RoPE

## 1. The Bag of Words Problem
The standard Attention formula $\text{softmax}(QK^T)V$ has absolutely no concept of time or order. If you feed it "The dog bit the man" and "The man bit the dog", the mathematical output of the Attention matrix is identical. 
To fix this, we must inject position information into the embeddings.

## 2. Absolute Positional Encoding (The 2017 Way)
The original 2017 Transformer added a sine/cosine wave to the word embedding.
$$ \text{Embedding} = \text{Word\_Vector} + \text{Sine\_Wave\_Position} $$
**The Flaw:** If you train the model on sequences of 1,000 words, it literally has no mathematical idea what position 1,001 looks like. It completely collapses if you give it a document longer than its training data.

## 3. Rotary Position Embeddings (RoPE)
Almost every modern SOTA model (Llama-3, Mistral, Qwen) uses **RoPE**.
Instead of adding a number to the embedding, RoPE treats the Query and Key vectors as complex numbers on a 2D plane and **Rotates** them based on their position $m$.

$$ q_m = q e^{im\theta} $$
When the model calculates the dot product between Query $m$ and Key $n$, the math naturally simplifies to only care about the *relative distance* between them ($m - n$). 

### 3.1 Why RoPE is Brilliant (Context Extension)
Because RoPE relies on rotation angles, we can easily "trick" the model into reading 1-million-token contexts even if it was only trained on 8k tokens. 
If we simply divide the rotation angle by a factor of 10, the model thinks a 10,000-word document is only 1,000 words long. This technique is called **NTK-aware Interpolation** or **YaRN**, and it is how modern models magically support infinite context windows without retraining from scratch.

## 4. ALiBi (Attention with Linear Biases)
Used in models like MPT, ALiBi skips embedding rotation entirely. It just subtracts a penalty directly from the Attention score matrix before the softmax.
If word A is 10 spaces away from word B, ALiBi subtracts a massive penalty from their attention score. If they are right next to each other, it subtracts zero. This naturally forces the model to pay attention to local context and gives it the ability to extrapolate to infinite lengths gracefully.
