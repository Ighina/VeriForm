


from typing import List, Protocol, Optional

from .deepseek.prover.lean.verifier import Lean4ServerScheduler
from .dag import DAGModel
from abc import abstractmethod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import re




class Prover(Protocol):
    @abstractmethod
    def prove(self, dag: DAGModel) -> DAGModel:
        ...


LEAN_TEMPLATE = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

-- {statement}
{lean_code}
""".strip()

LEAN_WRAPPER_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{lean_code}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()



class DeepSeekProver:
    MODEL_DIR = "deepseek-ai/DeepSeek-Prover-V2-7B"
    def __init__(self, batch_size: int = 1):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.model = LLM(
            self.MODEL_DIR,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.25,
        )
        self.lean_pattern = re.compile(r"```lean4(.*?)```", re.DOTALL | re.IGNORECASE)
        self.batch_size = batch_size

        # 'theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by ...' => 'theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10'
        self.theorem_pattern = re.compile(r"theorem\s+(\w+)\s*:\s*(.*?)\s*:=", re.DOTALL)

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=16384,
            n=self.batch_size
        )

        self.scheduler = Lean4ServerScheduler(
            max_concurrent_requests=self.batch_size,
        )
    
    def __del__(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.close()

    def parse_lean_code(self, response_text: str) -> str:
        """
        Get the LAST Lean 4 code block from the response text.
        """
        matches = self.lean_pattern.findall(response_text)
        if matches:
            return matches[-1].strip()
        else:
            raise ValueError("No Lean 4 code block found in the response.")
        
    def get_theorem_name_body(self, formal_problem: str) -> tuple[Optional[str], Optional[str]]:
        """
        IMPORTANT: Extract the theorem name and body for the last occurrence only.
        """
        matches = self.theorem_pattern.findall(formal_problem)
        if matches:
            name, body = matches[-1]
            return name.strip(), body.strip()
        else:
            return None, None

    def initialize_dialog(self):
        self.dialog = []

    def get_valid_proof(self, target_body: str, proof_candidates: List[str], comment: str = '') -> Optional[str]:
        valid_candidates = []
        for candidate in proof_candidates:
            try:
                lean_code = self.parse_lean_code(candidate)
            except ValueError:
                continue
            proof_name, proof_body = self.get_theorem_name_body(lean_code)
            if proof_body != target_body:
                continue
            valid_candidates.append(
                (
                    lean_code,
                    LEAN_TEMPLATE.format(
                        lean_code=lean_code,
                        statement=comment
                    )
                )
            )
        
        result_ids = self.scheduler.submit_all_request([
            dict(code=lean_code_full, ast=False, tactics=False)
            for _, lean_code_full in valid_candidates
        ])

        results = self.scheduler.get_all_request_outputs(result_ids)


        for (lean_code, full_lean_code), result in zip(valid_candidates, results):
            if result['complete']:
                return lean_code
        
        return None
    
    def prove_single(self, problem_nl: str, problem_fl: str) -> str:
        user_prompt = LEAN_WRAPPER_TEMPLATE.format(
            lean_code=LEAN_TEMPLATE.format(
                lean_code=problem_fl,
                statement=problem_nl
            )
        )
        name, body = self.get_theorem_name_body(problem_fl)
        assert name is not None and body is not None, "Failed to extract theorem name and body."
        self.dialog.append({"role": "user", "content": user_prompt})
        prompt = self.tokenizer.apply_chat_template(self.dialog, tokenize=False, add_generation_prompt=True)
        response = self.model.generate(prompt, sampling_params=self.sampling_params)[0]
        # Append model response to dialog

        valid_proof = self.get_valid_proof(
            target_body=body,
            proof_candidates=[response.outputs[i].text for i in range(self.batch_size)],
            comment=problem_nl
        )
        if valid_proof is not None:
            return valid_proof
        else:
            return problem_fl # Keep it unsolved

    def prove(self, dag: DAGModel) -> DAGModel:
        for node in dag.nodes:
            self.initialize_dialog()
            informal_problem = node.content
            formal_problem = node.formalized_content
            node.formalized_content = self.prove_single(
                informal_problem, formal_problem)
        return dag