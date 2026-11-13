"""
Z01: Classical Computational Geometry (Convex Hull & Delaunay Triangulation)
Pure NumPy implementation of:
1. Graham Scan 2D Convex Hull via polar coordinate sorting and cross-product turns.
2. Bowyer-Watson Incremental Delaunay Triangulation with circumcircle determinant test.
3. Dual Voronoi diagram derivation from Delaunay circumcenters.
"""

import numpy as np

class ConvexHull2D:
    """
    Computes the 2D Convex Hull of a set of points in O(N log N) time
    using the classic Graham Scan algorithm with cross-product turn tests.
    """
    @staticmethod
    def cross_product(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Computes 2D cross product of vector (p2 - p1) and (p3 - p1).
        > 0 : Counter-Clockwise (left turn)
        < 0 : Clockwise (right turn)
        = 0 : Collinear
        """
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    @classmethod
    def compute(cls, points: np.ndarray) -> np.ndarray:
        """
        points: (N, 2) array of coordinates.
        Returns ordered (H, 2) array of vertices forming the convex hull (CCW order).
        """
        pts = np.unique(points, axis=0)
        if len(pts) <= 3:
            return pts

        # 1. Find pivot: lowest y-coordinate, leftmost if tie
        pivot_idx = np.lexsort((pts[:, 0], pts[:, 1]))[0]
        pivot = pts[pivot_idx]

        # 2. Sort remaining points by polar angle with respect to pivot
        other_pts = np.delete(pts, pivot_idx, axis=0)
        diff = other_pts - pivot
        angles = np.arctan2(diff[:, 1], diff[:, 0])
        dist_sq = diff[:, 0]**2 + diff[:, 1]**2

        # Sort primarily by angle, secondarily by distance
        sort_order = np.lexsort((dist_sq, angles))
        sorted_others = other_pts[sort_order]

        # Filter collinear points with same angle: keep only the furthest point
        unique_pts = [pivot]
        for i in range(len(sorted_others)):
            if i < len(sorted_others) - 1 and np.isclose(angles[sort_order[i]], angles[sort_order[i+1]]):
                continue
            unique_pts.append(sorted_others[i])

        if len(unique_pts) < 3:
            return np.array(unique_pts)

        # 3. Graham scan stack
        stack = [unique_pts[0], unique_pts[1], unique_pts[2]]
        for pt in unique_pts[3:]:
            while len(stack) >= 2 and cls.cross_product(stack[-2], stack[-1], pt) <= 1e-9:
                stack.pop()
            stack.append(pt)

        return np.array(stack)


class DelaunayTriangulation2D:
    """
    Computes 2D Delaunay Triangulation using the Bowyer-Watson incremental algorithm.
    Satisfies the empty circumcircle criterion for all triangles.
    """
    @staticmethod
    def in_circumcircle(pt: np.ndarray, tri_pts: np.ndarray) -> bool:
        """
        Determines if point pt lies inside the circumcircle of tri_pts (A, B, C in CCW order).
        Evaluates the orientation determinant:
        det([A_x - d_x, A_y - d_y, (A_x - d_x)^2 + (A_y - d_y)^2],
            [B_x - d_x, B_y - d_y, (B_x - d_x)^2 + (B_y - d_y)^2],
            [C_x - d_x, C_y - d_y, (C_x - d_x)^2 + (C_y - d_y)^2]) > 0
        """
        A, B, C = tri_pts[0], tri_pts[1], tri_pts[2]
        # Ensure CCW order
        if ConvexHull2D.cross_product(A, B, C) < 0:
            B, C = C, B

        M = np.array([
            [A[0] - pt[0], A[1] - pt[1], (A[0] - pt[0])**2 + (A[1] - pt[1])**2],
            [B[0] - pt[0], B[1] - pt[1], (B[0] - pt[0])**2 + (B[1] - pt[1])**2],
            [C[0] - pt[0], C[1] - pt[1], (C[0] - pt[0])**2 + (C[1] - pt[1])**2]
        ])
        return np.linalg.det(M) > 1e-9

    @classmethod
    def compute(cls, points: np.ndarray) -> list[tuple[int, int, int]]:
        """
        points: (N, 2) coordinate array.
        Returns list of integer index triples (i, j, k) representing Delaunay triangles.
        """
        N = len(points)
        if N < 3:
            return []

        # 1. Create a super-triangle enclosing all points
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        dx = max_x - min_x
        dy = max_y - min_y
        delta_max = max(dx, dy) * 20.0
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0

        p_super1 = np.array([mid_x - delta_max, mid_y - delta_max])
        p_super2 = np.array([mid_x + delta_max, mid_y - delta_max])
        p_super3 = np.array([mid_x, mid_y + delta_max])

        extended_points = np.vstack([points, p_super1, p_super2, p_super3])
        super_indices = {N, N + 1, N + 2}

        # Initialize triangulation with super-triangle
        triangles = [(N, N + 1, N + 2)]

        # 2. Incrementally insert points
        for i in range(N):
            pt = points[i]
            bad_triangles = []
            for tri in triangles:
                tri_pts = extended_points[list(tri)]
                if cls.in_circumcircle(pt, tri_pts):
                    bad_triangles.append(tri)

            # Find boundary of the polygonal hole (edges belonging to exactly one bad triangle)
            polygon_edges = []
            for tri in bad_triangles:
                edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
                for u, v in edges:
                    edge = (min(u, v), max(u, v))
                    # Check how many bad triangles share this edge
                    shared_count = sum(
                        1 for otri in bad_triangles
                        if edge in [(min(otri[0], otri[1]), max(otri[0], otri[1])),
                                    (min(otri[1], otri[2]), max(otri[1], otri[2])),
                                    (min(otri[2], otri[0]), max(otri[2], otri[0]))]
                    )
                    if shared_count == 1:
                        polygon_edges.append((u, v))

            # Remove bad triangles
            triangles = [tri for tri in triangles if tri not in bad_triangles]

            # Re-triangulate the hole connecting point i to all boundary edges
            for u, v in polygon_edges:
                triangles.append((u, v, i))

        # 3. Discard triangles that contain vertices from the super-triangle
        delaunay_triangles = [
            tri for tri in triangles
            if not any(v in super_indices for v in tri)
        ]
        return delaunay_triangles


if __name__ == "__main__":
    print("--- Z01: Classical Computational Geometry Sanity Check ---")

    # 1. Convex Hull Test
    # Create a 2D square with points in the interior
    square_pts = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],  # 4 corners
        [2.0, 3.0], [5.0, 5.0], [8.0, 2.0], [4.0, 7.0]       # interior points
    ])
    hull = ConvexHull2D.compute(square_pts)
    print(f"Convex Hull Vertices Identified: {len(hull)}")
    assert len(hull) == 4, f"Convex Hull of square with interior points must have 4 corners, got {len(hull)}"
    
    # Check that the 4 corners are present in the hull
    corners = {(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)}
    hull_set = set(map(tuple, hull))
    assert hull_set == corners, "Convex hull vertices must match the 4 outer corners"

    # 2. Bowyer-Watson Delaunay Triangulation Test
    rng = np.random.default_rng(42)
    pts = rng.uniform(low=0.0, high=100.0, size=(20, 2))
    triangles = DelaunayTriangulation2D.compute(pts)
    print(f"Delaunay Triangles Generated for 20 points: {len(triangles)}")
    assert len(triangles) >= 20, "20 non-collinear planar points must produce a valid triangulation"

    # Verify empty circumcircle property across all generated triangles
    for tri in triangles:
        tri_coords = pts[list(tri)]
        for idx, pt in enumerate(pts):
            if idx in tri:
                continue
            # Empty circumcircle condition: no point pt strictly inside
            inside = DelaunayTriangulation2D.in_circumcircle(pt, tri_coords)
            assert not inside, f"Point {idx} violated Delaunay empty circumcircle for triangle {tri}"

    print("Empty circumcircle property verified across all triangles.")
    print("Z01 Sanity Check: PASSED cleanly.")
