import unittest


from veriform.autoformalization_v2.prover import DeepSeekProver
import pickle

class TestProver(unittest.TestCase):
    def test_prove(self):

        with open("tests/pkls/formalized_dag.pkl", "rb") as f:
            formalized_dag = pickle.load(f)

        prover = DeepSeekProver()
        proven_dag = prover.prove(formalized_dag)
        
        for node in proven_dag.nodes:
            print(f"Original node content: {node.content}")
            print(f"Formalized node content: \n```\n{node.formalized_content}\n```")
        
        del prover

    def test_perturbed_prove(self):
        prover = DeepSeekProver()

        with open("tests/pkls/formalized_perturbed_dag.pkl", "rb") as f:
            formalized_dag = pickle.load(f)

        proven_dag = prover.prove(formalized_dag)

        for node in proven_dag.nodes:
            print(f"Original node content: {node.content}")
            print(f"Formalized node content: \n```\n{node.formalized_content}```")
        
        del prover
        
# Run the tests
# To run the test_prove only, run
# python -m unittest tests.test_prover.TestProver.test_prove
# To run the test_perturbed_prove only, run
# python -m unittest tests.test_prover.TestProver.test_perturbed_prove
# Run the full test_prover suite:
# python -m unittest tests.test_prover: