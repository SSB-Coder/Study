"""
X08. Forward & Inverse Kinematics — Jacobian-Based IK
From-scratch implementation in pure NumPy.
"""

import numpy as np

class PlanarArmIK:
    """
    Kinematics for an N-link planar robotic manipulator arm.
    Computes:
    1. Forward Kinematics (FK) via transformation matrices.
    2. Manipulator Jacobian J(theta) = dx / dtheta.
    3. Inverse Kinematics (IK) via Damped Least Squares (Levenberg-Marquardt).
    """
    
    def __init__(self, link_lengths: list):
        self.lengths = np.array(link_lengths, dtype=float)
        self.n_links = len(link_lengths)

    def forward_kinematics(self, thetas: np.ndarray):
        """
        Computes (x, y) coordinates of all joints and the end-effector <- eq. 3.1.
        thetas: array of relative joint angles in radians.
        """
        x, y = 0.0, 0.0
        angle = 0.0
        joint_positions = [(x, y)]
        
        for i in range(self.n_links):
            angle += thetas[i]
            x += self.lengths[i] * np.cos(angle)
            y += self.lengths[i] * np.sin(angle)
            joint_positions.append((x, y))
            
        end_effector = np.array([x, y])
        return end_effector, joint_positions

    def compute_jacobian(self, thetas: np.ndarray) -> np.ndarray:
        """
        Computes 2 x N analytic Jacobian matrix J = [dx/dtheta; dy/dtheta] <- eq. 3.2.
        Column i is the cross product z_i x (p_ee - p_i).
        """
        _, joints = self.forward_kinematics(thetas)
        p_ee = np.array(joints[-1])
        J = np.zeros((2, self.n_links))
        
        for i in range(self.n_links):
            p_i = np.array(joints[i])
            # In 2D, z-axis is out of plane [0, 0, 1]
            # Cross product: [0, 0, 1] x [dx, dy, 0] = [-dy, dx, 0]
            delta = p_ee - p_i
            J[0, i] = -delta[1] # -dy
            J[1, i] =  delta[0] #  dx
            
        return J

    def inverse_kinematics(self, target_xy: np.ndarray, init_thetas: np.ndarray = None,
                           damping: float = 0.1, max_iter: int = 200, tol: float = 1e-4):
        """
        Damped Least Squares (DLS) Inverse Kinematics <- eq. 3.3:
        Delta theta = J^T (J J^T + lambda^2 I)^{-1} (target - current)
        """
        if init_thetas is None:
            thetas = np.zeros(self.n_links, dtype=float)
        else:
            thetas = np.array(init_thetas, dtype=float).copy()
            
        target = np.array(target_xy, dtype=float)
        reach = np.sum(self.lengths)
        if np.linalg.norm(target) > reach:
            raise ValueError(f"Target {target} outside workspace reach {reach}")

        for it in range(max_iter):
            curr_pos, _ = self.forward_kinematics(thetas)
            error = target - curr_pos
            if np.linalg.norm(error) < tol:
                return thetas, it + 1, True
                
            J = self.compute_jacobian(thetas)
            # Damped pseudo-inverse: J^T * inv(J * J^T + lambda^2 * I)
            JJT_damp = J @ J.T + (damping**2) * np.eye(2)
            d_thetas = J.T @ np.linalg.solve(JJT_damp, error)
            
            # Update joint angles with step limit
            thetas += np.clip(d_thetas, -0.5, 0.5)
            # Wrap to [-pi, pi]
            thetas = (thetas + np.pi) % (2.0 * np.pi) - np.pi
            
        final_pos, _ = self.forward_kinematics(thetas)
        return thetas, max_iter, np.linalg.norm(target - final_pos) < tol

if __name__ == "__main__":
    print("--- X08: Jacobian Inverse Kinematics Sanity Check ---")
    # 3-link arm with link lengths 10, 8, 5 (Total reach = 23)
    arm = PlanarArmIK([10.0, 8.0, 5.0])
    
    # 1. Forward Kinematics verification
    thetas_test = np.array([np.pi/4, np.pi/6, -np.pi/3])
    ee_pos, joints = arm.forward_kinematics(thetas_test)
    print(f"Test Angles: {thetas_test}")
    print(f"FK End-Effector: {ee_pos}")
    
    # 2. Inverse Kinematics to reach target (12.0, 8.0)
    target = np.array([12.0, 8.0])
    solved_thetas, iters, success = arm.inverse_kinematics(target, init_thetas=[0.1, 0.1, 0.1])
    achieved_pos, _ = arm.forward_kinematics(solved_thetas)
    
    print(f"IK Target: {target}")
    print(f"Achieved Position: {achieved_pos} in {iters} iterations")
    pos_err = np.linalg.norm(achieved_pos - target)
    print(f"Position Error: {pos_err:.2e}")
    
    assert success, "IK failed to reach target"
    assert pos_err < 1e-3, "IK accuracy out of tolerance"
    print("X08 Sanity Check: PASSED cleanly.")
