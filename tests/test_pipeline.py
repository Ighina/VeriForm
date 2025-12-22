import unittest

from src.veriform.data_collection.dataset_loaders import GSM8KLoader, MATHLoader
from src.veriform.data_collection.reasoning_step import ReasoningChain

from veriform.autoformalization_v2.dag import StandardDAGModel
from veriform.autoformalization_v2.perturber import StandardPerturber
from veriform.autoformalization_v2.formalizer import StepfunFormalizer
from veriform.autoformalization_v2.prover import DeepSeekProver
from veriform.autoformalization_v2.pipeline import StandardPipeline

from pprint import pprint


class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        perturber = StandardPerturber(
            p=0.0, 
            operator_swap=False, 
            value_change=True, 
            logical_negation=False)
        formalizer = StepfunFormalizer()
        prover = DeepSeekProver()

        pipeline = StandardPipeline(perturber, formalizer, prover)

        lean_program = pipeline(chains[0])
        print(lean_program)

        del pipeline

    def test_perturbed_pipeline(self):
        gsm8k = GSM8KLoader(split="test", num_samples=1)
        chains = gsm8k.load()
        perturber = StandardPerturber(
            p=1.0, 
            operator_swap=False, 
            value_change=True, 
            logical_negation=False)
        formalizer = StepfunFormalizer()
        prover = DeepSeekProver()

        pipeline = StandardPipeline(perturber, formalizer, prover)

        lean_program = pipeline(chains[0])
        print(lean_program)

        del pipeline

        
# Run the tests
# To run the test_pipeline only, run
# python -m unittest tests.test_pipeline.TestPipeline.test_pipeline
# To run the test_perturbed_pipeline only, run
# python -m unittest tests.test_pipeline.TestPipeline.test_perturbed_pipeline
# Run the full test_pipeline suite:
# python -m unittest tests.test_pipeline