# M04: Probabilistic Streaming Sketches (Count-Min Sketch, HyperLogLog, Reservoir Sampling)

## 1. The Beginner's Analogy
Imagine you are standing beside a roaring river of data—billions of Google search queries or network packets rushing past every second. You have only a tiny sticky note (a few kilobytes of RAM) to record what is happening:
- If you try to remember every unique word in a dictionary or hash map, your memory explodes instantly ($O(N)$).
- **Count-Min Sketch:** Acts like a grid of tally counters with multiple color-coded lenses. Even if some items accidentally share the same counter (a collision), by looking at the *minimum* counter across all lenses, you get an accurate frequency count that never underestimates.
- **HyperLogLog:** Flips coins for every item. If you see a coin come up heads 20 times in a row, you can statistically guarantee you've seen roughly $2^{20} \approx 1,000,000$ unique items, using just a handful of 6-bit registers!
- **Reservoir Sampling:** Maintains a fair, unbiased sample of exactly $k$ fish from an infinite stream, giving every fish that swam past an equal $k/N$ chance of being in your bucket.

## 2. The Deep Architecture & Mechanics
Modern big-data streaming architectures and online ML pipelines cannot store full historical state. They rely on sub-linear **streaming sketches**:
1. **Count-Min Sketch (Frequency Estimation):**
   - Uses a 2D matrix of counters $C \in \mathbb{N}^{d \times w}$.
   - For each item $x$, updates $d$ independent hash locations: $C[i, h_i(x)] \leftarrow C[i, h_i(x)] + c$.
   - Frequency query returns the minimum: $\hat{a}_x = \min_{1 \le i \le d} C[i, h_i(x)]$.
2. **HyperLogLog (Cardinality Estimation):**
   - Hashes items to uniform 64-bit integers.
   - Uses the first $p$ bits to route to one of $m = 2^p$ registers.
   - Uses the remaining bits to count leading zeros $\rho(w) = \text{number of consecutive zeros} + 1$.
   - Combines registers using the **normalized harmonic mean** to neutralize outlier variance.
3. **Reservoir Sampling (Uniform Sampling):**
   - Keeps $k$ elements in a buffer.
   - For the $i$-th element in the stream ($i > k$), replaces a random element with probability $k/i$.

## 3. The Hardcore Mathematics

### Count-Min Sketch Error Bounds (Cormode & Muthukrishnan)
For parameters $\varepsilon$ (additive error factor) and $\delta$ (failure probability):
Set matrix width $w = \left\lceil \frac{e}{\varepsilon} \right\rceil$ and depth $d = \left\lceil \ln\left(\frac{1}{\delta}\right) \right\rceil$.
For any item $x$ with true frequency $a_x$ and stream norm $\|a\|_1 = \sum_y a_y$:
$$a_x \le \hat{a}_x \le a_x + \varepsilon \|a\|_1$$
with probability at least $1 - \delta$.

*Proof Sketch:*
Since counters only increment, $\hat{a}_x \ge a_x$.
Let $I_{i, x, y} = 1$ if $h_i(x) = h_i(y)$ and $0$ otherwise.
For independent pairwise universal hash functions, $\mathbb{E}[I_{i, x, y}] \le 1/w \le \varepsilon / e$.
The expected collision error in row $i$ is:
$$\mathbb{E}[C[i, h_i(x)] - a_x] \le \frac{\varepsilon}{e} \|a\|_1$$
By Markov's inequality:
$$P\left(C[i, h_i(x)] - a_x \ge \varepsilon \|a\|_1\right) \le \frac{1}{e}$$
Taking the minimum over $d = \ln(1/\delta)$ independent rows:
$$P\left(\hat{a}_x - a_x \ge \varepsilon \|a\|_1\right) \le \left(\frac{1}{e}\right)^d \le \delta$$

### HyperLogLog Harmonic Mean & Bias Correction
Given $m = 2^p$ registers $M[0 \dots m-1]$ tracking maximum leading zeros:
The raw Flajolet-Martin indicator is:
$$Z = \left( \sum_{j=0}^{m-1} 2^{-M[j]} \right)^{-1}$$
Raw estimate:
$$E = \alpha_m m^2 Z$$
where the normalization constant $\alpha_m$ corrects multiplicative bias:
$$\alpha_m = \left( m \int_0^\infty \left(\log_2\left(\frac{2 + u}{1 + u}\right)\right)^m du \right)^{-1}$$
For $m \ge 128$: $\alpha_m \approx \frac{0.7213}{1 + 1.079/m}$.
Standard error: $\text{SE} \approx \frac{1.04}{\sqrt{m}}$.

### Reservoir Sampling Invariant
To prove every element from $1 \dots N$ has probability $k/N$ of being in the reservoir of size $k$:
By induction on $N$:
For $N = k$, probability is $k/k = 1$.
Assume true for $N - 1$. The $N$-th element enters the reservoir with probability $k/N$.
An existing element survives if either:
1. The $N$-th element is rejected: $1 - k/N$.
2. The $N$-th element is accepted, but replaces a *different* element: $\frac{k}{N} \times \frac{k-1}{k} = \frac{k-1}{N}$.
Combining survival probability:
$$P(\text{survive}) = \left(1 - \frac{k}{N}\right) + \frac{k-1}{N} = \frac{N - 1}{N}$$
By inductive hypothesis, its prior probability was $k/(N-1)$.
Total probability:
$$P(\text{in reservoir at step } N) = \frac{k}{N - 1} \times \frac{N - 1}{N} = \frac{k}{N}$$

## 4. The Algorithm Step-by-Step
1. **Count-Min Sketch:** Hash item across $d$ rows with universal random linear hashes; increment $C[i, h_i(x)]$. To query, return $\min_i C[i, h_i(x)]$.
2. **HyperLogLog:** Hash element to 64-bit integer. Split into register index $j$ and remainder $w$. Count leading zeros of $w$ plus one; update $M[j] \leftarrow \max(M[j], \rho(w))$. Compute harmonic mean.
3. **Reservoir Sampling:** Fill buffer with first $k$ items. For item $i > k$, pick random integer $j \in [0, i-1]$. If $j < k$, buffer$[j] \leftarrow \text{item}$.

## 5. Limitations & Failure Modes
- **CMS Overestimation on Heavy Tails:** In highly skewed Zipfian distributions, frequent heavy hitters bleed into rare item counters.
- **HLL Small Cardinality Bias:** Raw HLL overestimates when many registers are 0; requires LinearCounting fallback ($m \ln(m/V)$ where $V$ is number of empty registers).

## 6. Where This Fits
Cross-reference: Streaming sketches provide the real-time feature extraction foundation for online learning (J05), drift detection, network anomaly detection (J08), and production feature stores handling millions of events per second.
