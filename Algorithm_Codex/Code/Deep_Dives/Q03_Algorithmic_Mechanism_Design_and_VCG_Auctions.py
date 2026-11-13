"""
Q03: Algorithmic Mechanism Design & VCG Auctions
Pure NumPy implementation of reverse game theory and incentive compatibility:
1. Vickrey Single-Item Second-Price Auction
2. Combinatorial Vickrey-Clarke-Groves (VCG) Mechanism with Clarke Pivot Payments
3. Empirical proof of Dominant-Strategy Incentive Compatibility (DSIC / Truthfulness)
"""

import itertools
import numpy as np
from typing import List, Dict, Tuple


class VickreyAuction:
    """Single-item second-price sealed-bid auction."""
    @staticmethod
    def run(bids: Dict[int, float]) -> Tuple[int, float]:
        """Returns (winner_id, payment)."""
        sorted_bidders = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        winner_id, highest_bid = sorted_bidders[0]
        second_highest = sorted_bidders[1][1] if len(sorted_bidders) > 1 else 0.0
        return winner_id, second_highest


class CombinatorialVCGAuction:
    """
    Multi-item VCG auction with discrete bundle allocations.
    Allocation maximizes reported social welfare.
    Clarke Pivot Payment: p_i = max_{k} sum_{j!=i} b_j(k) - sum_{j!=i} b_j(k*)
    """
    def __init__(self, items: List[str], bidders: List[int]):
        self.items = items
        self.bidders = bidders
        self.n_items = len(items)
        self.n_bidders = len(bidders)

    def _all_partitions(self) -> List[Dict[int, Tuple[str, ...]]]:
        """Generates all possible disjoint item partition allocations among bidders."""
        allocations = []
        # Assign each item to one bidder or unallocated (None)
        b_choices = self.bidders + [None]
        for assignment in itertools.product(b_choices, repeat=self.n_items):
            alloc = {b: [] for b in self.bidders}
            for item_idx, b in enumerate(assignment):
                if b is not None:
                    alloc[b].append(self.items[item_idx])
            allocations.append({b: tuple(sorted(alloc[b])) for b in self.bidders})
        return allocations

    def run(self, bid_valuations: Dict[int, Dict[Tuple[str, ...], float]]) -> Tuple[Dict[int, Tuple[str, ...]], Dict[int, float]]:
        """
        bid_valuations: bidder_id -> {bundle_tuple: declared_bid}
        Returns: (optimal_allocation, payments)
        """
        all_allocs = self._all_partitions()

        def eval_welfare(alloc: Dict[int, Tuple[str, ...]], exclude_bidder: int = None) -> float:
            tot = 0.0
            for b, bundle in alloc.items():
                if b != exclude_bidder:
                    tot += bid_valuations[b].get(bundle, 0.0)
            return tot

        # 1. Global Social Welfare Maximization
        best_alloc = None
        best_welfare = -1.0
        for alloc in all_allocs:
            w = eval_welfare(alloc)
            if w > best_welfare:
                best_welfare = w
                best_alloc = alloc

        # 2. Clarke Pivot Payments
        payments = {}
        for i in self.bidders:
            # Optimal welfare of other bidders if bidder i were completely absent
            w_without_i = max(eval_welfare(alloc, exclude_bidder=i) for alloc in all_allocs)
            # Welfare of other bidders in the actual chosen allocation
            w_others_actual = eval_welfare(best_alloc, exclude_bidder=i)
            # Externality imposed by bidder i
            payments[i] = max(0.0, float(w_without_i - w_others_actual))

        return best_alloc, payments


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Single Item Vickrey Auction Verification
    bids = {1: 100.0, 2: 75.0, 3: 40.0}
    winner, payment = VickreyAuction.run(bids)
    print(f"Vickrey Auction: Winner={winner}, Pays={payment} (Second-Highest Bid)")
    assert winner == 1 and np.isclose(payment, 75.0), "Vickrey payment must equal 2nd highest bid"

    # 2. Combinatorial VCG Auction
    # Items: A (Compute), B (Storage)
    # Bidder 1 wants bundle (A, B) for 15.0, but only 4.0 for A alone
    # Bidder 2 wants A for 8.0
    # Bidder 3 wants B for 6.0
    items = ["Compute", "Storage"]
    bidders = [1, 2, 3]
    vcg = CombinatorialVCGAuction(items, bidders)

    valuations = {
        1: {(): 0.0, ("Compute",): 4.0, ("Storage",): 3.0, ("Compute", "Storage"): 15.0},
        2: {(): 0.0, ("Compute",): 8.0, ("Storage",): 0.0, ("Compute", "Storage"): 8.0},
        3: {(): 0.0, ("Compute",): 0.0, ("Storage",): 6.0, ("Compute", "Storage"): 6.0}
    }

    alloc, payments = vcg.run(valuations)
    print(f"VCG Allocations: {alloc}")
    print(f"VCG Payments:    {payments}")

    # Bidder 1 wins both items because 15 > (8 + 6 = 14)
    assert alloc[1] == ("Compute", "Storage"), "Bidder 1 must win both items"
    # Bidder 1 pays the externality: without Bidder 1, Bidders 2 and 3 would have gotten 8 + 6 = 14.
    # With Bidder 1 winning, others get 0. Thus, Bidder 1 pays 14.0!
    assert np.isclose(payments[1], 14.0), f"Expected Bidder 1 payment 14.0, got {payments[1]}"
    assert np.isclose(payments[2], 0.0) and np.isclose(payments[3], 0.0)

    # 3. Empirical Verification of Dominant Strategy Incentive Compatibility (DSIC / Truthfulness)
    # Test that deviating from truthful valuation never increases Bidder 1's utility
    true_v1 = valuations[1][("Compute", "Storage")]  # 15.0
    truthful_util = true_v1 - payments[1]  # 15.0 - 14.0 = 1.0

    shading_bids = [10.0, 13.0, 14.5, 16.0, 20.0, 30.0]
    for fake_bid in shading_bids:
        manipulated_vals = {
            1: {(): 0.0, ("Compute",): 4.0, ("Storage",): 3.0, ("Compute", "Storage"): fake_bid},
            2: valuations[2],
            3: valuations[3]
        }
        fake_alloc, fake_payments = vcg.run(manipulated_vals)
        if fake_alloc[1] == ("Compute", "Storage"):
            fake_util = true_v1 - fake_payments[1]
        else:
            fake_util = 0.0

        # Deviating must never strictly improve utility over truth-telling
        assert fake_util <= truthful_util + 1e-9, f"Truthfulness violated: fake bid {fake_bid} yielded higher utility!"

    print(f"Verified DSIC Truthfulness: Truthful Utility={truthful_util:.2f} >= all non-truthful deviations.")
    print("Q03 Sanity Check: PASSED cleanly.")
