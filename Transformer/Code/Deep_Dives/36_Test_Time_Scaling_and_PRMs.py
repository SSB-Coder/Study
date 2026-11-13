"""
36. Test-Time Compute Scaling & Process Reward Models (PRMs)
Deep Dive Implementation

This file demonstrates the 2026 bleeding-edge meta: Inference-Time Scaling.
Instead of making models larger, we force them to "think" longer.
It covers:
1. Parallel Scaling (Self-Consistency / Majority Voting)
2. Sequential Scaling (Tree-of-Thoughts)
3. Process Reward Models (Grading intermediate <thought> tokens, not just the final answer)
"""

import random

# ---------------------------------------------------------------------------
# 1. MOCK MODELS
# ---------------------------------------------------------------------------

class MockLLM:
    """A dummy LLM that generates reasoning steps."""
    def generate_step(self, context):
        # Nondeterministic generation to simulate multiple branching paths
        paths = [
            "<thought> Let's carry the 1... </thought>",
            "<thought> Wait, I need to use the quadratic formula. </thought>",
            "<thought> The derivative is 2x. </thought>",
            "<thought> Actually, this is a trick question. </thought>"
        ]
        return random.choice(paths)
        
    def generate_final_answer(self, context):
        answers = ["42", "73", "12", "42", "42"] # 42 is the most likely
        return random.choice(answers)

class MockProcessRewardModel:
    """
    A PRM grades *intermediate steps*, unlike an Outcome Reward Model (ORM) 
    which only grades the final answer. This prevents hallucination chains.
    """
    def grade_step(self, step_text):
        # A real PRM would use a neural network to output a score from -1.0 to 1.0.
        if "trick question" in step_text:
            return -1.0 # Penalize lazy thinking
        elif "derivative" in step_text or "carry the 1" in step_text:
            return 0.8  # Good, mathematically sound steps
        else:
            return 0.1  # Neutral

# ---------------------------------------------------------------------------
# 2. PARALLEL SCALING (Self-Consistency)
# ---------------------------------------------------------------------------
# The simplest way to scale test-time compute. Generate N independent answers.
# The most common answer is mathematically proven to be the most accurate.

def parallel_self_consistency(llm, prompt, num_samples=5):
    print("\n--- Strategy 1: Parallel Scaling (Self-Consistency) ---")
    answers = []
    
    for i in range(num_samples):
        # Simulate generating a full reasoning chain and answer
        ans = llm.generate_final_answer(prompt)
        answers.append(ans)
        print(f"Trajectory {i+1}: Generated Answer -> {ans}")
        
    # Majority Voting
    most_common = max(set(answers), key=answers.count)
    print(f"Final Output (Majority Vote): {most_common}")
    return most_common

# ---------------------------------------------------------------------------
# 3. SEQUENTIAL SCALING (Tree-of-Thoughts with PRM)
# ---------------------------------------------------------------------------
# The model branches its logic. A PRM grades each branch. We prune the bad
# branches and only continue expanding the good branches.

def tree_of_thoughts_search(llm, prm, prompt, max_depth=3, branches_per_step=3):
    print("\n--- Strategy 2: Sequential Scaling (Tree of Thoughts) ---")
    
    # State is a list of tuples: (context_string, cumulative_score)
    current_states = [(prompt, 0.0)]
    
    for depth in range(max_depth):
        print(f"\n[Depth {depth + 1}] Expanding reasoning tree...")
        next_states = []
        
        for context, score in current_states:
            # Generate multiple possible next steps from this state
            for b in range(branches_per_step):
                new_step = llm.generate_step(context)
                
                # The PRM grades the step immediately (Process Reward)
                step_reward = prm.grade_step(new_step)
                
                new_context = context + "\n" + new_step
                new_score = score + step_reward
                
                next_states.append((new_context, new_score))
                print(f"  Branch generated: {new_step} | PRM Score: {step_reward:.1f}")
                
        # Pruning: Keep only the Top-K best reasoning chains (Beam Search)
        # Let's keep the top 2 best chains to continue expanding
        next_states.sort(key=lambda x: x[1], reverse=True)
        current_states = next_states[:2]
        print(f"  Pruning: Kept top 2 branches (Scores: {current_states[0][1]:.1f}, {current_states[1][1]:.1f})")
        
    # Finally, generate the answer from the best reasoning chain
    best_context = current_states[0][0]
    final_answer = llm.generate_final_answer(best_context)
    print(f"\nFinal Chosen Answer from optimal reasoning path: {final_answer}")
    return final_answer

# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    llm = MockLLM()
    prm = MockProcessRewardModel()
    
    prompt = "User: Solve this complex physics equation."
    
    parallel_self_consistency(llm, prompt, num_samples=5)
    tree_of_thoughts_search(llm, prm, prompt)
