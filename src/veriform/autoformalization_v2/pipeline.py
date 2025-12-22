from abc import ABC, abstractmethod
from ..data_collection.reasoning_step import ReasoningChain
from .dag import StandardDAGModel
from .perturber import Perturber
from .formalizer import Formalizer
from .prover import Prover
# --- Component Signatures (Protocols) ---
# These define what your helper objects must look like ("Duck Typing").
# It is 'spot on' for keeping dependencies loose.



# --- Main Pipeline Signature ---

class PerturbedFormalizationPipeline(ABC):
    """
    The Blueprint. It defines the 'what', not the 'how'.
    Any concrete pipeline must implement these methods.
    """

    def __init__(self,
                perturbation_model: Perturber, # Use the Protocol type!
                formalizer_model: Formalizer,
                prover_model: Prover):
        self.perturbation_model = perturbation_model
        self.formalizer_model = formalizer_model
        self.prover_model = prover_model

    @abstractmethod
    def __call__(self, reasoning_chain: ReasoningChain) -> str:
        ...

class StandardPipeline(PerturbedFormalizationPipeline):
    """
    The concrete implementation. This is where the rubber meets the road.
    """
    def __init__(self,
                perturbation_model: Perturber, # Use the Protocol type!
                formalizer_model: Formalizer,
                prover_model: Prover):
        self.perturbation_model = perturbation_model
        self.formalizer_model = formalizer_model
        self.prover_model = prover_model
    
    def __call__(self, reasoning_chain: ReasoningChain) -> str:
        """Execute the full pipeline."""
        dag = StandardDAGModel(reasoning_chain)
        
        for i in range(len(dag)):
            dag = self.perturbation_model.perturb(dag, step_id=i)
        
        dag = self.formalizer_model.formalize(dag)

        dag = self.prover_model.prove(dag)

        return dag.lean()