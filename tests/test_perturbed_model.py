import unittest

from veriform.data_collection.dataset_loaders import GSM8KLoader
from veriform.autoformalization_v2.dag import StandardDAGModel
from veriform.autoformalization_v2.perturber import StandardPerturber
from pprint import pprint

class TestPerturbedModel(unittest.TestCase):
    def test_perturbation(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        dag = StandardDAGModel(chains[0])
        perturber = StandardPerturber(
            p=1.0, 
            operator_swap=False, 
            value_change=True, 
            logical_negation=False)
        
        for node in dag.nodes:
            print(f"Original node content: {node.content}")
        perturbed_dag = perturber.perturb(dag, step_id=2)
        for node in perturbed_dag.nodes:
            print(f"Perturbed node content: {node.content}")

# Run the tests
# python -m unittest tests.test_perturbed_model