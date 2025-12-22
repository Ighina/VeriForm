import unittest
from src.veriform.data_collection.dataset_loaders import GSM8KLoader, MATHLoader
from src.veriform.data_collection.reasoning_step import ReasoningChain

class TestDatasetLoaders(unittest.TestCase):

    def test_gsm8k_loader(self):
        """Test GSM8KLoader with a small sample."""
        loader = GSM8KLoader(split="test", num_samples=2, seed=42)
        chains = loader.load()

        self.assertIsInstance(chains, list)
        self.assertGreater(len(chains), 0)
        self.assertIsInstance(chains[0], ReasoningChain)
        self.assertEqual(chains[0].source_dataset, "gsm8k")

    def test_math_loader(self):
        """Test MATHLoader with a small sample."""
        loader = MATHLoader(split="test", num_samples=2, seed=42)
        chains = loader.load()

        self.assertIsInstance(chains, list)
        self.assertGreater(len(chains), 0)
        self.assertIsInstance(chains[0], ReasoningChain)
        self.assertEqual(chains[0].source_dataset, "math")

if __name__ == "__main__":
    unittest.main()