"""
G08: Proximal Policy Optimization (PPO) Clipped Objective
Pure NumPy implementation of the PPO-Clip objective and surrogate loss.
"""

import numpy as np

class PPOSurrogateLoss:
    """Computes the PPO clipped surrogate objective with analytical gradients."""
    def __init__(self, clip_eps: float = 0.2):
        self.clip_eps = clip_eps

    def compute_loss(self, log_probs: np.ndarray, old_log_probs: np.ndarray, advantages: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Computes the clipped surrogate loss:
        L_clip = - mean(min(r * A, clip(r, 1-eps, 1+eps) * A))
        Returns:
            loss: float scalar
            d_log_probs: gradient of loss w.r.t active log_probs
        """
        ratios = np.exp(log_probs - old_log_probs)
        clipped_ratios = np.clip(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        surr1 = ratios * advantages
        surr2 = clipped_ratios * advantages

        # Elementwise minimum: pessimistic bound
        chosen_surr = np.minimum(surr1, surr2)
        loss = - float(np.mean(chosen_surr))

        # Analytical gradient w.r.t log_probs
        # d/d(log_p) [ r * A ] = d/d(log_p) [ exp(log_p - old_log_p) * A ] = r * A
        # If surr2 < surr1 and ratio is outside [1-eps, 1+eps], gradient is 0.
        B = len(advantages)
        d_log_probs = np.zeros(B, dtype=np.float64)
        for i in range(B):
            if surr1[i] <= surr2[i]:
                # surr1 is active: unclipped gradient passes through
                d_log_probs[i] = - ratios[i] * advantages[i] / B
            else:
                # surr2 is strictly smaller: ratio was clipped, gradient is 0
                d_log_probs[i] = 0.0

        return loss, d_log_probs


if __name__ == "__main__":
    print("--- G08: PPO Clipped Objective Sanity Check ---")
    ppo = PPOSurrogateLoss(clip_eps=0.2)

    # Test Case 1: Positive advantage with large ratio (> 1 + eps)
    # The gradient should be zero because it is clipped!
    old_lp = np.array([0.0])
    new_lp = np.array([0.5])  # ratio = exp(0.5) = 1.648 > 1.2
    adv = np.array([1.0])
    loss, d_lp = ppo.compute_loss(new_lp, old_lp, adv)
    print(f"Clipped Positive Advantage -> Ratio: {np.exp(new_lp - old_lp)[0]:.4f}, Loss: {loss:.4f}, Grad: {d_lp[0]}")
    assert d_lp[0] == 0.0, "Gradient must be 0 when ratio is clipped beyond 1+eps!"

    # Test Case 2: Positive advantage within trust region (ratio = 1.1)
    new_lp_near = np.log(np.array([1.1]))
    loss_near, d_lp_near = ppo.compute_loss(new_lp_near, old_lp, adv)
    print(f"Unclipped Positive Advantage -> Ratio: {np.exp(new_lp_near - old_lp)[0]:.4f}, Loss: {loss_near:.4f}, Grad: {d_lp_near[0]:.4f}")
    assert d_lp_near[0] < 0.0, "Gradient must be non-zero (negative to minimize loss) when unclipped!"

    # Test Case 3: Negative advantage with small ratio (< 1 - eps)
    # Here surr2 = (1 - eps) * A is smaller than r * A, so ratio is clipped!
    adv_neg = np.array([-2.0])
    new_lp_small = np.log(np.array([0.5])) # ratio = 0.5 < 0.8
    loss_neg, d_lp_neg = ppo.compute_loss(new_lp_small, old_lp, adv_neg)
    print(f"Clipped Negative Advantage Small Ratio -> Loss: {loss_neg:.4f}, Grad: {d_lp_neg[0]:.4f}")
    assert d_lp_neg[0] == 0.0, "For negative advantage when ratio drops below 1-eps, gradient must be 0 (clipped)"

    # Test Case 4: Negative advantage within trust region (ratio = 0.95)
    new_lp_neg_near = np.log(np.array([0.95]))
    loss_neg_near, d_lp_neg_near = ppo.compute_loss(new_lp_neg_near, old_lp, adv_neg)
    print(f"Unclipped Negative Advantage -> Ratio: 0.95, Loss: {loss_neg_near:.4f}, Grad: {d_lp_neg_near[0]:.4f}")
    assert d_lp_neg_near[0] > 0.0, "Gradient must be positive (to push log-prob down) when unclipped"

    print("G08 Sanity Check: PASSED cleanly.")
