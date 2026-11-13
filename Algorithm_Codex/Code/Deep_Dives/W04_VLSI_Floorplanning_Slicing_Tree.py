"""
W04. VLSI Floorplanning via Slicing Trees & Simulated Annealing
From-scratch implementation in pure NumPy.
"""

import numpy as np

class SlicingFloorplanner:
    """
    Floorplanning for rectangular macro blocks using Normalized Polish Expressions
    representing Slicing Trees.
    Operators: 'H' (horizontal cut), 'V' (vertical cut).
    Operands: Integer block indices.
    """
    
    def __init__(self, block_dims: list):
        """
        block_dims: list of (width, height) tuples for each block.
        """
        self.block_dims = block_dims
        self.n_blocks = len(block_dims)
        # Initial Polish expression: e.g. for 4 blocks: 0 1 V 2 H 3 V
        expr = [0]
        for i in range(1, self.n_blocks):
            expr.extend([i, 'V' if i % 2 == 1 else 'H'])
        self.expression = expr

    def evaluate_dimensions(self, polish_expr: list):
        """
        Evaluates bounding box (width, height) of floorplan using stack evaluation <- eq. 3.1.
        """
        stack = []
        for token in polish_expr:
            if isinstance(token, int):
                stack.append(self.block_dims[token])
            elif token == 'V':
                # Vertical cut: place side-by-side -> W = w1 + w2, H = max(h1, h2)
                w2, h2 = stack.pop()
                w1, h1 = stack.pop()
                stack.append((w1 + w2, max(h1, h2)))
            elif token == 'H':
                # Horizontal cut: stack on top -> W = max(w1, w2), H = h1 + h2
                w2, h2 = stack.pop()
                w1, h1 = stack.pop()
                stack.append((max(w1, w2), h1 + h2))
        return stack[0]

    def compute_cost(self, polish_expr: list, alpha: float = 0.7) -> float:
        """Cost = alpha * Area + (1 - alpha) * Aspect_Ratio_Penalty."""
        w, h = self.evaluate_dimensions(polish_expr)
        area = w * h
        aspect = max(w / h, h / w)
        return alpha * area + (1.0 - alpha) * 100.0 * (aspect - 1.0)**2

    def perturb(self, expr: list) -> list:
        """
        Wong-Liu moves on Normalized Polish Expressions:
        M1: Swap two adjacent operands.
        M2: Complement a chain of operators (V <-> H).
        M3: Swap adjacent operand and operator.
        """
        new_expr = list(expr)
        move_type = np.random.choice(['M1', 'M2', 'M3'])
        
        if move_type == 'M1':
            operand_indices = [i for i, t in enumerate(new_expr) if isinstance(t, int)]
            if len(operand_indices) >= 2:
                i1, i2 = np.random.choice(operand_indices, 2, replace=False)
                new_expr[i1], new_expr[i2] = new_expr[i2], new_expr[i1]
        elif move_type == 'M2':
            op_indices = [i for i, t in enumerate(new_expr) if t in ('V', 'H')]
            if op_indices:
                idx = np.random.choice(op_indices)
                new_expr[idx] = 'H' if new_expr[idx] == 'V' else 'V'
        else: # M3
            for i in range(len(new_expr) - 1):
                if (isinstance(new_expr[i], int) and new_expr[i+1] in ('V', 'H')) or \
                   (new_expr[i] in ('V', 'H') and isinstance(new_expr[i+1], int)):
                    new_expr[i], new_expr[i+1] = new_expr[i+1], new_expr[i]
                    # Verify validity
                    if self._is_valid(new_expr):
                        return new_expr
            # If invalid, revert
            return expr
            
        return new_expr if self._is_valid(new_expr) else expr

    def _is_valid(self, expr: list) -> bool:
        """Verifies Polish expression obeys balloting property (operands always lead operators)."""
        count = 0
        for t in expr:
            if isinstance(t, int):
                count += 1
            else:
                count -= 1
                if count < 1:
                    return False
        return count == 1

    def optimize(self, n_steps: int = 500, initial_temp: float = 1000.0, cooling: float = 0.96):
        """Simulated annealing optimization of slicing floorplan."""
        curr_expr = list(self.expression)
        curr_cost = self.compute_cost(curr_expr)
        best_expr = list(curr_expr)
        best_cost = curr_cost
        temp = initial_temp
        
        for _ in range(n_steps):
            cand_expr = self.perturb(curr_expr)
            cand_cost = self.compute_cost(cand_expr)
            delta = cand_cost - curr_cost
            
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-4)):
                curr_expr = cand_expr
                curr_cost = cand_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_expr = list(curr_expr)
                    
            temp *= cooling
            
        self.expression = best_expr
        w, h = self.evaluate_dimensions(best_expr)
        return best_cost, (w, h)

if __name__ == "__main__":
    print("--- W04: Slicing Floorplanner Sanity Check ---")
    np.random.seed(42)
    # 5 macro blocks with given (w, h)
    blocks = [(4, 6), (5, 3), (3, 7), (6, 4), (5, 5)]
    total_module_area = sum(w * h for w, h in blocks)
    
    floorplanner = SlicingFloorplanner(blocks)
    w_init, h_init = floorplanner.evaluate_dimensions(floorplanner.expression)
    init_area = w_init * h_init
    print(f"Total Block Area: {total_module_area}")
    print(f"Initial Floorplan Bounding Box: {w_init}x{h_init} = Area {init_area}")
    
    best_cost, (w_opt, h_opt) = floorplanner.optimize(n_steps=400)
    opt_area = w_opt * h_opt
    print(f"Optimized Floorplan Bounding Box: {w_opt}x{h_opt} = Area {opt_area}")
    
    assert opt_area <= init_area, "Optimization should not increase bounding box area"
    assert opt_area >= total_module_area, "Floorplan area cannot be smaller than block sum"
    print("W04 Sanity Check: PASSED cleanly.")
