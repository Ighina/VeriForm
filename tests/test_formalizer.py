import unittest

from veriform.data_collection.dataset_loaders import GSM8KLoader
from veriform.autoformalization_v2.dag import StandardDAGModel
from veriform.autoformalization_v2.perturber import StandardPerturber
from veriform.autoformalization_v2.formalizer import StepfunFormalizer
import pickle

class TestFormalizer(unittest.TestCase):
    def test_formalization(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        dag = StandardDAGModel(chains[0])
        
        formalizer = StepfunFormalizer()

        formalized_dag = formalizer.formalize(dag)

        
        for node_nl, node_fl in zip(dag.nodes, formalized_dag.nodes):
            print(f"Original node content: {node_nl.content}")
            print(f"Formalized node content: ```\n{node_fl.formalized_content}```")
        
        # Save to the pkl of formalized_dag for testing the next stage

        with open("tests/pkls/formalized_dag.pkl", "wb") as f:
            pickle.dump(formalized_dag, f)


    def test_perturbed_formalization(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        dag = StandardDAGModel(chains[0])
        perturber = StandardPerturber(
            p=1.0, 
            operator_swap=True, 
            value_change=True, 
            logical_negation=True)
        
        dag = perturber.perturb(dag, step_id=0)
        dag = perturber.perturb(dag, step_id=1)
        dag = perturber.perturb(dag, step_id=2)
        
        formalizer = StepfunFormalizer()

        formalized_dag = formalizer.formalize(dag)

        
        for node_nl, node_fl in zip(dag.nodes, formalized_dag.nodes):
            print(f"Original node content: {node_nl.content}")
            print(f"Formalized node content: ```\n{node_fl.formalized_content}```")
        
        with open("tests/pkls/formalized_perturbed_dag.pkl", "wb") as f:
            pickle.dump(formalized_dag, f)

# Run the tests
# python -m unittest tests.test_formalizer