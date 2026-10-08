"""Verification result schema: what was checked, the outcome, and a precise evidence label."""

from typing import Literal

from pydantic import BaseModel, model_validator

CheckName = Literal["code_block", "syntax", "requested_names", "self_examples", "module_execution", "math"]
Evidence = Literal[
    "passed",
    "not_checkable",
    "syntax_failed",
    "function_missing",
    "function_signature_mismatch",
    "module_execution_failed",
    "assertion_failed",
    "self_example_failed",
    "math_mismatch",
]

_FAIL_EVIDENCE: dict[str, Evidence] = {
    "code_block": "function_missing",
    "syntax": "syntax_failed",
    "requested_names": "function_missing",
    "self_examples": "self_example_failed",
    "module_execution": "module_execution_failed",
    "math": "math_mismatch",
}


class CheckResult(BaseModel):
    """One answer check.

    status: "fail" is strong evidence the answer is wrong or internally inconsistent (escalates);
    "pass" is supporting evidence only; "not_checkable" means the check applied but could not decide.
    """

    name: CheckName
    status: Literal["pass", "fail", "not_checkable"]
    detail: str = ""
    evidence: Evidence | None = None

    @model_validator(mode="after")
    def _default_evidence(self) -> "CheckResult":
        if self.evidence is None:
            self.evidence = {"pass": "passed", "not_checkable": "not_checkable"}.get(self.status) or _FAIL_EVIDENCE[self.name]
        return self
