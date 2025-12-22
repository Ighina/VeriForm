


from typing import Protocol, Optional

from stepfun_playground.stepfun_formalizer import MODEL_DIR
from .dag import DAGModel
from abc import abstractmethod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import re

class Formalizer(Protocol):
    @abstractmethod
    def formalize(self, dag_input: DAGModel, step_id: Optional[int] = None) -> DAGModel:
        ...

class StepfunFormalizer:
    MODEL_DIR = "stepfun-ai/StepFun-Formalizer-7B"

    def __init__(self):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.model = LLM(
            self.MODEL_DIR,
            tensor_parallel_size=2, # 8 for 32B, 4 for 7B
            gpu_memory_utilization=0.25,
        )
        self.lean_pattern = re.compile(
            r"```lean4.*?(theorem.*?)```", 
            re.DOTALL | re.IGNORECASE
        )

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
            n=2
        )

    def initialize_dialog(self):
        self.dialog = [
            {"role": "system", "content": self.system_prompt}
        ]

    def get_formal_statement_prompt(self, informal_problem: str, header: str = "import Mathlib\n") -> str:
        prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
        prompt += informal_problem
        prompt += f"\n\nYour code should start with:\n```Lean4\n{header}\n```\n"
        return prompt
    
    def parse_lean_code(self, response_text: str) -> str:
        """
        Get the LAST Lean 4 code block from the response text.
        """
        matches = self.lean_pattern.findall(response_text)
        if matches:
            return matches[-1].strip()
        else:
            raise ValueError("No Lean 4 code block found in the response.")
    
    def _formalize_prompt(self, informal_problem: str) -> str:
        header = "import Mathlib\n\nopen Real\n"
        user_prompt = self.get_formal_statement_prompt(informal_problem, header)
        self.dialog.append({"role": "user", "content": user_prompt})
        prompt = self.tokenizer.apply_chat_template(self.dialog, tokenize=False, add_generation_prompt=True) + "<think>"
        response = self.model.generate(prompt, sampling_params=self.sampling_params)[0]
        self.dialog.append({"role": "assistant", "content": response.outputs[0].text}) 
        # Append model response to dialog
        return self.parse_lean_code(response.outputs[0].text)

    def formalize(self, dag: DAGModel, cleanup_dialog: bool = True) -> DAGModel:
        for i, node in enumerate(dag.nodes):
            if cleanup_dialog:
                self.initialize_dialog()
            informal_problem = node.content
            node.formalized_content = self._formalize_prompt(informal_problem)
            # theorem my_favorite_theorem => theorem step_i
            node.formalized_content = node.formalized_content.replace(
                "theorem my_favorite_theorem", 
                f"theorem step_{i}")
        return dag