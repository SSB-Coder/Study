"""
Y01: Lie Groups SO(3), SE(3) and Geometric Equivariance
Pure NumPy implementation of 3D rotation groups, rigid-body transformations,
hat/vee Lie algebra operators, exponential and logarithmic maps, adjoint actions,
and equivariant neural coordinate transformations.
"""

import numpy as np

class SO3:
    """Special Orthogonal Group SO(3) and Lie Algebra so(3)."""
    
    @staticmethod
    def hat(omega: np.ndarray) -> np.ndarray:
        """Map 3D vector omega to 3x3 skew-symmetric matrix [omega]_x in so(3)."""
        w1, w2, w3 = omega
        return np.array([
            [0.0, -w3, w2],
            [w3, 0.0, -w1],
            [-w2, w1, 0.0]
        ], dtype=float)

    @staticmethod
    def vee(omega_hat: np.ndarray) -> np.ndarray:
        """Map 3x3 skew-symmetric matrix back to 3D vector omega."""
        return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], dtype=float)

    @staticmethod
    def exp(omega: np.ndarray) -> np.ndarray:
        """
        Exponential map exp: so(3) -> SO(3) via Rodrigues' rotation formula.
        exp([omega]_x) = I + (sin theta / theta) [omega]_x + ((1 - cos theta) / theta^2) [omega]_x^2
        """
        theta = np.linalg.norm(omega)
        omega_hat = SO3.hat(omega)
        if theta < 1e-12:
            return np.eye(3) + omega_hat
        
        c = np.cos(theta)
        s = np.sin(theta)
        return np.eye(3) + (s / theta) * omega_hat + ((1.0 - c) / (theta**2)) * (omega_hat @ omega_hat)

    @staticmethod
    def log(R: np.ndarray) -> np.ndarray:
        """
        Logarithmic map log: SO(3) -> so(3). Returns 3D vector omega.
        theta = arccos((Tr(R) - 1) / 2)
        """
        tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(tr)
        if theta < 1e-12:
            return np.zeros(3, dtype=float)
        
        diff = (R - R.T) / (2.0 * np.sin(theta))
        return SO3.vee(diff) * theta

class SE3:
    """Special Euclidean Group SE(3) = SO(3) |>< R^3 for rigid-body transformations."""
    
    @staticmethod
    def hat(twist: np.ndarray) -> np.ndarray:
        """Map 6D twist xi = [v, omega] to 4x4 matrix in se(3)."""
        v = twist[:3]
        omega = twist[3:]
        mat = np.zeros((4, 4), dtype=float)
        mat[:3, :3] = SO3.hat(omega)
        mat[:3, 3] = v
        return mat

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """Map 4x4 matrix in se(3) back to 6D twist xi."""
        v = mat[:3, 3]
        omega = SO3.vee(mat[:3, :3])
        return np.concatenate([v, omega])

    @staticmethod
    def exp(twist: np.ndarray) -> np.ndarray:
        """
        Exponential map exp: se(3) -> SE(3).
        xi = [v, omega] -> [[R, V v], [0, 1]]
        """
        v = twist[:3]
        omega = twist[3:]
        theta = np.linalg.norm(omega)
        R = SO3.exp(omega)
        
        if theta < 1e-12:
            V = np.eye(3)
        else:
            omega_hat = SO3.hat(omega)
            s = np.sin(theta)
            c = np.cos(theta)
            V = np.eye(3) + ((1.0 - c) / (theta**2)) * omega_hat + ((theta - s) / (theta**3)) * (omega_hat @ omega_hat)
            
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = V @ v
        return T

    @staticmethod
    def log(T: np.ndarray) -> np.ndarray:
        """Logarithmic map log: SE(3) -> se(3). Returns 6D twist xi."""
        R = T[:3, :3]
        p = T[:3, 3]
        omega = SO3.log(R)
        theta = np.linalg.norm(omega)
        
        if theta < 1e-12:
            V_inv = np.eye(3)
        else:
            omega_hat = SO3.hat(omega)
            # V_inv = I - 0.5 [omega]_x + ((1 - theta*sin(theta)/(2*(1-cos(theta)))) / theta^2) [omega]_x^2
            coef = (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta)))) / (theta**2)
            V_inv = np.eye(3) - 0.5 * omega_hat + coef * (omega_hat @ omega_hat)
            
        v = V_inv @ p
        return np.concatenate([v, omega])

    @staticmethod
    def adjoint(T: np.ndarray) -> np.ndarray:
        """
        Adjoint representation matrix Ad_T in R^{6x6}:
        Ad_T = [[R, [p]_x R], [0, R]]
        """
        R = T[:3, :3]
        p = T[:3, 3]
        p_hat = SO3.hat(p)
        Ad = np.zeros((6, 6), dtype=float)
        Ad[:3, :3] = R
        Ad[:3, 3:] = p_hat @ R
        Ad[3:, 3:] = R
        return Ad

def test_equivariance(coords: np.ndarray, transform_T: np.ndarray) -> float:
    """
    Test SE(3)-equivariance of a geometric vector field f(X).
    Here f computes the geometric centroid-relative displacement field:
    f(X) = X - mean(X).
    Equivariance implies: f(T . X) = R . f(X).
    """
    R = transform_T[:3, :3]
    p = transform_T[:3, 3]
    
    # 1. Transform coordinates: T . X
    coords_transformed = (R @ coords.T).T + p
    
    # 2. Apply f to transformed coords: f(T . X)
    f_transformed = coords_transformed - np.mean(coords_transformed, axis=0)
    
    # 3. Apply R to original f(X): R . f(X)
    f_orig = coords - np.mean(coords, axis=0)
    expected_f = (R @ f_orig.T).T
    
    return float(np.max(np.abs(f_transformed - expected_f)))

if __name__ == "__main__":
    print("--- Y01: Lie Groups SO(3) & SE(3) Sanity Check ---")
    
    # 1. SO(3) Exp/Log round-trip
    rng = np.random.default_rng(42)
    omega_test = rng.uniform(-1.5, 1.5, size=3)
    R = SO3.exp(omega_test)
    assert np.allclose(R.T @ R, np.eye(3)), "Rotation matrix R must be orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "det(R) must equal 1.0"
    
    omega_recovered = SO3.log(R)
    assert np.allclose(omega_test, omega_recovered, atol=1e-12), "SO(3) exp/log round-trip failed"
    
    # 2. SE(3) Exp/Log round-trip
    twist_test = rng.uniform(-2.0, 2.0, size=6)
    T = SE3.exp(twist_test)
    twist_recovered = SE3.log(T)
    assert np.allclose(twist_test, twist_recovered, atol=1e-10), "SE(3) exp/log round-trip failed"
    
    # 3. Adjoint Representation Identity: T * hat(xi) * T^{-1} == hat(Ad_T * xi)
    T_inv = np.linalg.inv(T)
    xi_hat = SE3.hat(twist_test)
    lhs = T @ xi_hat @ T_inv
    rhs = SE3.hat(SE3.adjoint(T) @ twist_test)
    assert np.allclose(lhs, rhs, atol=1e-10), "Adjoint representation identity violated"
    
    # 4. SE(3) Equivariance Verification
    point_cloud = rng.normal(size=(50, 3))
    eq_error = test_equivariance(point_cloud, T)
    print(f"SE(3) Equivariance Discrepancy: {eq_error:.2e}")
    assert eq_error < 1e-12, "Geometric layer must be strictly SE(3)-equivariant"
    
    print("Y01 Sanity Check: PASSED cleanly.")
