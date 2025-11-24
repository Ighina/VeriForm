"""
Lean code verification utilities.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class VerificationResult:
    """Result of verifying Lean code."""

    success: bool
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    has_sorry: bool = False
    has_admit: bool = False

    @property
    def is_provable(self) -> bool:
        """Check if the code is provable (no sorry/admit and no errors)."""
        return self.success and not self.has_sorry and not self.has_admit


class LeanVerifier:
    """Verifier for Lean 4 code."""

    def __init__(self, lean_executable: str = "lean", timeout: int = 30):
        """
        Initialize the Lean verifier.

        Args:
            lean_executable: Path to Lean executable
            timeout: Timeout in seconds for verification
        """
        self.lean_executable = lean_executable
        self.timeout = timeout

    def verify(self, lean_code: str) -> VerificationResult:
        """
        Verify Lean code.

        Args:
            lean_code: The Lean code to verify

        Returns:
            VerificationResult indicating success or failure
        """
        # Check for sorry and admit
        has_sorry = "sorry" in lean_code
        has_admit = "admit" in lean_code

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            delete=False
        ) as f:
            f.write(lean_code)
            temp_file = f.name

        try:
            # Run Lean on the file
            result = subprocess.run(
                [self.lean_executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            success = result.returncode == 0

            return VerificationResult(
                success=success,
                error_message=None if success else "Lean verification failed",
                stdout=result.stdout,
                stderr=result.stderr,
                has_sorry=has_sorry,
                has_admit=has_admit
            )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                success=False,
                error_message=f"Verification timeout after {self.timeout}s",
                has_sorry=has_sorry,
                has_admit=has_admit
            )
        except FileNotFoundError:
            return VerificationResult(
                success=False,
                error_message=f"Lean executable not found: {self.lean_executable}",
                has_sorry=has_sorry,
                has_admit=has_admit
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                error_message=f"Verification error: {str(e)}",
                has_sorry=has_sorry,
                has_admit=has_admit
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    def verify_file(self, file_path: str) -> VerificationResult:
        """
        Verify a Lean file.

        Args:
            file_path: Path to the Lean file

        Returns:
            VerificationResult
        """
        with open(file_path, 'r') as f:
            lean_code = f.read()

        return self.verify(lean_code)


class MockLeanVerifier(LeanVerifier):
    """Mock verifier for testing without Lean installed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def verify(self, lean_code: str) -> VerificationResult:
        """
        Mock verification that checks for basic syntax issues.

        This is a simple heuristic-based checker for testing purposes.
        """
        has_sorry = "sorry" in lean_code
        has_admit = "admit" in lean_code

        # Simple checks
        errors = []

        if not any(keyword in lean_code for keyword in ["theorem", "lemma", "def", "example"]):
            errors.append("No theorem/lemma/def/example declaration found")

        if lean_code.count("(") != lean_code.count(")"):
            errors.append("Mismatched parentheses")

        if lean_code.count("{") != lean_code.count("}"):
            errors.append("Mismatched braces")

        success = len(errors) == 0

        return VerificationResult(
            success=success,
            error_message="; ".join(errors) if errors else None,
            stdout="",
            stderr="; ".join(errors) if errors else "",
            has_sorry=has_sorry,
            has_admit=has_admit
        )
