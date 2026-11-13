# Deep Dive: Test-Time Compute Scaling & Process Reward Models

## 1. The 2026 Paradigm Shift: Inference-Time Scaling
From 2020 to 2024, the AI industry worshipped a single law: **Pre-training Scaling Laws**. The belief was simple: if you want a smarter model, you must use a larger dataset and train a model with more parameters.

By 2025/2026, researchers hit a wall. Making a model larger yielded diminishing returns on complex reasoning tasks (like high-level math and coding).

The solution? **Test-Time Compute Scaling** (also known as Inference-Time Scaling).
Instead of investing all the compute into *training* the model, we invest compute into the *inference phase*. We force the model to "think" for 30, 60, or even 120 seconds before it outputs a final answer.

Research proved that taking a small, cheap model and giving it massive amounts of Test-Time compute allows it to definitively beat a massive, expensive model running a standard single-pass inference.

---

## 2. The Three Strategies of Test-Time Scaling

There are three primary ways to scale compute during inference.
*   **See Code:** [36_Test_Time_Scaling_and_PRMs.py](../../Code/Deep_Dives/36_Test_Time_Scaling_and_PRMs.py)

### 2.1 Parallel Scaling (Self-Consistency / Pass@K)
This is the brute-force approach.
If you ask an LLM a complex math question, it might get it right 40% of the time.
Instead of generating one answer, we instruct the LLM to generate **100 independent reasoning paths**. 
We then use **Majority Voting**. If 45 paths output "Answer: 42", 20 output "Answer: 17", and 35 output random noise, we mathematically assume 42 is the correct answer. This simple technique dramatically raises benchmark scores.

### 2.2 Sequential Scaling (Tree-of-Thoughts / MCTS)
This is the intelligent approach. Instead of generating 100 blind paths, the model generates a "Tree" of logic.
1. The model generates 3 possible opening logical steps.
2. An internal grader (see PRMs below) evaluates the 3 steps.
3. The model abandons the lowest-scoring step and expands the two highest-scoring steps.
4. This continues, performing a Monte Carlo Tree Search (MCTS) until the optimal logical chain is found.

### 2.3 Hybrid / Adaptive Scaling
Frontier reasoning models like OpenAI o1 / o3 and DeepSeek-R1 use Adaptive Test-Time Scaling. They have a mechanism that determines how hard a question is. For a simple greeting, it uses $O(1)$ test-time compute. For a complex mathematical proof or coding task, it automatically scales up its test-time compute, generating thousands of internal reasoning tokens to plan, explore alternatives, and self-correct before outputting the final answer.

---

## 3. Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)

To make Sequential Scaling work, the model needs to know if a thought is "good" or "bad."

*   **Outcome Reward Model (ORM):** This is standard RLHF. The model outputs a massive wall of text and a final answer. A human (or AI judge) gives it a Thumbs Up or Thumbs Down. The fatal flaw? If the model used terrible, hallucinated logic but accidentally arrived at the correct final answer, the ORM rewards the entire hallucinated chain!
*   **Process Reward Model (PRM):** The PRM grades *every single sentence* of the reasoning chain.

### The Mathematics of RLVR (Reinforcement Learning with Verifiable Rewards)
PRMs are trained on massive datasets of human-annotated step-by-step reasoning. 
During inference, when the main Policy Model generates a `<thought>` token block, the PRM evaluates it.

Let $s_t$ be the state (the context so far) and $a_t$ be the action (the newest logical step).
The PRM outputs a value $V(s_t, a_t) \in [-1, 1]$.
If $V < 0$, the logical step contains a math error, a logical fallacy, or a hallucination. The system immediately prunes this branch of the Tree-of-Thoughts, preventing the model from spiraling into a completely broken answer.

By grading the *process* rather than the *outcome*, PRMs force the model to remain rigorously aligned to mathematical and logical truth.
