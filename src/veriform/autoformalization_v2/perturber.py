import random
from abc import ABC, abstractmethod
import re
from typing import List, Any, Optional, Protocol, Tuple, runtime_checkable
from .dag import DAGModel

class Perturber(Protocol):
    @abstractmethod
    def perturb(self, dag_input: DAGModel, step_id: Optional[int] = None) -> Any:
        ...

class StandardPerturber:
    def __init__(
        self,
        p: float = 0.1,
        operator_swap: bool = True,
        value_change: bool = True,
        logical_negation: bool = True,
    ):
        self.p = p
        self.operator_swap = operator_swap
        self.value_change = value_change
        self.logical_negation = logical_negation
    
    def operator_swap_str(self, content: str) -> Tuple[str, bool]:
        """Swap mathematical or logical operators in the content string."""
        # Enhanced operator map with correct inverses and comparison operators
        operators = {
            '+': '-', '-': '+',
            '*': '/', '/': '*',
            '>': '<=', '<': '>=',
            '>=': '<', '<=': '>',
            '==': '!=', '!=': '=='
        }
        swapping = list(operators.keys())
        random.shuffle(swapping) 
        for op in swapping:
            all_occurrences = [m.start() for m in re.finditer(re.escape(op), content)]
            if len(all_occurrences) > 0:
                idx = random.choice(all_occurrences)
                content = content[:idx] + operators[op] + content[idx + len(op):]
                return content, True
        return content, False
    
    def value_change_str(self, content: str) -> Tuple[str, bool]:
        """
        Change numerical values in the content string.
        e.g. 'Maryam raises 400+300=700 dollars.' => 'Maryam raises 400+250=650 dollars.'
        True for successful change, False otherwise.
        """
        # Try to find numbers on the RHS first
        if '=' in content:
            lhs, rhs = content.rsplit('=', 1)
            rhs_numbers = [m for m in re.finditer(r'\d+', rhs)]
            if rhs_numbers:
                match = random.choice(rhs_numbers)
                original_value = int(match.group())
                change = random.choice([-1, 1]) * random.randint(1, max(1, int(original_value * 0.1)) + 5)
                new_value = max(0, original_value + change)
                # Reconstruct content: lhs + '=' + rhs_before + new_val + rhs_after
                new_rhs = rhs[:match.start()] + str(new_value) + rhs[match.end():]
                return lhs + '=' + new_rhs, True

        # Fallback to original logic if no RHS numbers found
        numbers = [m for m in re.finditer(r'\d+', content)]
        if not numbers:
            return content, False
        
        # Pick a random number to change
        match = random.choice(numbers)
        original_value = int(match.group())
        
        # Generate a new value (e.g., +/- 1 to 10 or +/- 10%)
        change = random.choice([-1, 1]) * random.randint(1, max(1, int(original_value * 0.1)) + 5)
        new_value = max(0, original_value + change) # Ensure non-negative for simplicity, or allow negative
        
        # Replace only that specific occurrence
        content = content[:match.start()] + str(new_value) + content[match.end():]
        return content, True


    def logical_negation_str(self, content: str) -> Tuple[str, bool]:
        """Apply logical negation to the content string."""
        if "not" in content:
            content = content.replace("not", "")
            return content, True
        else:
            # Insert 'not' at a random position before a verb or at the start
            verbs = re.finditer(r'\b(is|are|was|were|has|have|do|does|did|can|will|shall|may|might|must)\b', content)
            verb_positions = [m.start() for m in verbs]
            if verb_positions:
                insert_pos = random.choice(verb_positions)
                content = content[:insert_pos] + "not " + content[insert_pos:]
                return content, True
            else:
                # If no verbs found, prepend 'not' to the content
                return content, False

    def perturb_str(self, content: str) -> str:
        """Apply perturbations to a string based on the initialized settings."""
        perturbed_content = content
        applying_fn = []

        if self.operator_swap:
            applying_fn.append(self.operator_swap_str)
        if self.value_change:
            applying_fn.append(self.value_change_str)
        if self.logical_negation:
            applying_fn.append(self.logical_negation_str)

        random.shuffle(applying_fn)

        changed = False

        for fn in applying_fn:
            perturbed_content, changed = fn(perturbed_content)
            if changed:
                break  # Apply only one perturbation
        
        if not changed:
            print(f"Warning: No perturbation was applied to the content {content}")

        return perturbed_content

    def perturb(self, dag_input: DAGModel, step_id: int) -> DAGModel:
        """Concrete implementation of perturbation."""
        # Implementation details would go here
        if step_id is None:
            step_id = random.randint(0, len(dag_input) - 1)        
        if random.random() < self.p:
            node = dag_input[step_id]
            node.content = self.perturb_str(node.content)
            dag_input[step_id] = node
        return dag_input


if __name__ == "__main__":
    from veriform.data_collection.dataset_loaders import GSM8KLoader
    from pprint import pprint
    from veriform.autoformalization_v2.dag import StandardDAGModel
    
    gsm8k = GSM8KLoader(split="test", num_samples=1)
    chains = gsm8k.load()
    dag_model = StandardDAGModel(chains[0])
    for chain in chains:
        pprint(chain)
        dag = dag_model.build_dag(chain)
        perturber = StandardPerturber(
            p=1.0, 
            operator_swap=False, 
            value_change=True, 
            logical_negation=False)
        print(dag)
        perturbed_dag = perturber.perturb(dag, step_id=2)
        print(perturbed_dag)

