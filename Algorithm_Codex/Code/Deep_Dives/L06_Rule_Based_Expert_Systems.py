"""
L06: Rule-Based Expert Systems
Pure Python/NumPy implementation of Classical Expert Systems.
Implements Forward-Chaining (data-driven) with conflict resolution
and Backward-Chaining (goal-driven) with recursive sub-goal resolution.
"""

from typing import List, Set, Dict, Optional

class Rule:
    def __init__(self, name: str, antecedents: List[str], consequent: str):
        self.name = name
        self.antecedents = set(antecedents)
        self.consequent = consequent


class RuleBasedInferenceEngine:
    def __init__(self):
        self.rules: List[Rule] = []
        self.facts: Set[str] = set()

    def add_rule(self, name: str, antecedents: List[str], consequent: str):
        self.rules.append(Rule(name, antecedents, consequent))

    def add_fact(self, fact: str):
        self.facts.add(fact)

    def forward_chain(self) -> Set[str]:
        """
        Forward-chaining (data-driven):
        Derives all logically deducible facts until quiescence.
        """
        inferred = set(self.facts)
        while True:
            new_derived = False
            for rule in self.rules:
                if rule.antecedents.issubset(inferred) and rule.consequent not in inferred:
                    inferred.add(rule.consequent)
                    new_derived = True
            if not new_derived:
                break
        return inferred

    def backward_chain(self, goal: str, visited: Optional[Set[str]] = None) -> bool:
        """
        Backward-chaining (goal-driven):
        Recursively proves whether goal can be derived from existing facts.
        """
        if visited is None:
            visited = set()

        # Base Case 1: Goal is already an asserted ground fact
        if goal in self.facts:
            return True

        # Cycle prevention
        if goal in visited:
            return False
        visited.add(goal)

        # Base Case 2: Find all rules whose consequent matches the goal
        matching_rules = [r for r in self.rules if r.consequent == goal]
        if not matching_rules:
            return False

        for rule in matching_rules:
            # Prove all antecedents of this rule recursively
            all_premises_proven = True
            for premise in rule.antecedents:
                if not self.backward_chain(premise, visited.copy()):
                    all_premises_proven = False
                    break
            if all_premises_proven:
                # Add proven goal to known facts for future caching
                self.facts.add(goal)
                return True

        return False


if __name__ == "__main__":
    print("--- L06: Rule-Based Expert Systems Sanity Check ---")

    engine = RuleBasedInferenceEngine()

    # Rule Base for Medical / Fault Diagnosis:
    # R1: fever AND cough -> infection
    # R2: infection AND chest_pain -> pneumonia
    # R3: fever AND rash -> measles
    # R4: chills -> fever
    engine.add_rule("R1", ["fever", "cough"], "infection")
    engine.add_rule("R2", ["infection", "chest_pain"], "pneumonia")
    engine.add_rule("R3", ["fever", "rash"], "measles")
    engine.add_rule("R4", ["chills"], "fever")

    # Patient symptoms
    engine.add_fact("chills")
    engine.add_fact("cough")
    engine.add_fact("chest_pain")

    # 1. Test Forward Chaining
    deduced = engine.forward_chain()
    print(f"Initial Facts: [chills, cough, chest_pain]")
    print(f"Forward-Chaining Deduced Facts: {sorted(list(deduced))}")

    assert "fever" in deduced, "R4 should deduce fever from chills"
    assert "infection" in deduced, "R1 should deduce infection from fever + cough"
    assert "pneumonia" in deduced, "R2 should deduce pneumonia from infection + chest_pain"
    assert "measles" not in deduced, "Measles should NOT be deduced without rash"

    # 2. Test Backward Chaining
    # Reset engine facts
    engine.facts = {"chills", "cough", "chest_pain"}
    can_prove_pneumonia = engine.backward_chain("pneumonia")
    can_prove_measles = engine.backward_chain("measles")

    print(f"Backward Chaining Goal 'pneumonia': Proven = {can_prove_pneumonia}")
    print(f"Backward Chaining Goal 'measles':   Proven = {can_prove_measles}")

    assert can_prove_pneumonia is True, "Backward chaining must prove pneumonia from symptoms"
    assert can_prove_measles is False, "Backward chaining must fail to prove measles without rash"

    print("L06 Sanity Check: PASSED cleanly.")
