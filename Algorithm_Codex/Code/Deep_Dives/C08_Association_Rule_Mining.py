"""
C08. Association Rule Mining (Apriori & Metric Calculations)
From-scratch implementation in pure NumPy/Python.
"""

from itertools import combinations

class AprioriScratch:
    """From-scratch Apriori algorithm for mining frequent itemsets and association rules."""
    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets_ = {} # frozenset -> support
        self.rules_ = [] # (antecedent, consequent, support, confidence, lift)

    def fit(self, transactions: list):
        N = len(transactions)
        # 1. Frequent 1-itemsets
        item_counts = {}
        for t in transactions:
            for item in set(t):
                item_counts[frozenset([item])] = item_counts.get(frozenset([item]), 0) + 1

        # Filter by min_support <- eq. 3.1
        current_L = {item: count / N for item, count in item_counts.items() if (count / N) >= self.min_support}
        self.frequent_itemsets_.update(current_L)

        k = 2
        while current_L:
            # Candidate generation: join L_{k-1} with L_{k-1}
            candidates = set()
            items_list = list(current_L.keys())
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    union_set = items_list[i] | items_list[j]
                    if len(union_set) == k:
                        candidates.add(union_set)

            # Count support for candidates
            candidate_counts = {c: 0 for c in candidates}
            for t in transactions:
                t_set = set(t)
                for c in candidates:
                    if c.issubset(t_set):
                        candidate_counts[c] += 1

            # Filter candidates by min_support
            current_L = {c: count / N for c, count in candidate_counts.items() if (count / N) >= self.min_support}
            self.frequent_itemsets_.update(current_L)
            k += 1

        # 2. Extract Association Rules <- eq. 3.2
        for itemset, supp_xy in self.frequent_itemsets_.items():
            if len(itemset) < 2:
                continue
            
            # Generate all proper subsets
            for r in range(1, len(itemset)):
                for antecedent in combinations(itemset, r):
                    ant_set = frozenset(antecedent)
                    conseq_set = itemset - ant_set
                    
                    supp_x = self.frequent_itemsets_[ant_set]
                    supp_y = self.frequent_itemsets_.get(conseq_set, 0.0)
                    
                    # Confidence = P(X, Y) / P(X) <- eq. 3.2
                    conf = supp_xy / supp_x
                    if conf >= self.min_confidence:
                        # Lift = P(X, Y) / (P(X) * P(Y)) <- eq. 3.2
                        lift = supp_xy / (supp_x * supp_y) if supp_y > 0 else 0.0
                        self.rules_.append((ant_set, conseq_set, supp_xy, conf, lift))

        return self

if __name__ == "__main__":
    print("--- C08: Association Rule Mining (Apriori) Sanity Check ---")
    # Toy supermarket receipts
    dataset = [
        ["Milk", "Bread", "Beer"],
        ["Bread", "Diapers", "Beer", "Eggs"],
        ["Milk", "Diapers", "Beer", "Cola"],
        ["Bread", "Milk", "Diapers", "Beer"],
        ["Bread", "Milk", "Diapers", "Cola"]
    ]

    apriori = AprioriScratch(min_support=0.4, min_confidence=0.6).fit(dataset)
    print(f"Frequent Itemsets discovered: {len(apriori.frequent_itemsets_)}")
    print(f"Association Rules generated: {len(apriori.rules_)}")
    
    # Verify that Beer and Diapers co-occur
    beer_diapers_rule = [r for r in apriori.rules_ if "Beer" in r[0] and "Diapers" in r[1]]
    print(f"Rules with Beer -> Diapers: {len(beer_diapers_rule)}")
    assert len(apriori.frequent_itemsets_) > 0
    assert len(apriori.rules_) > 0
    print("C08 Sanity Check: PASSED cleanly.")
