"""Bounded, actionable tool errors without reflecting private argument values."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pydantic import ValidationError

MAX_ISSUES = 8

# These are developer-authored constraints, never arbitrary exception text.
_CONSTRAINTS = {
    "missing": "Supply this required field.",
    "extra_forbidden": "Remove undeclared fields; describe_tool lists accepted fields.",
    "literal_error": "Choose one of the values in this field's schema enum.",
    "enum": "Choose one of the values in this field's schema enum.",
    "string_type": "Supply a string.",
    "string_too_short": "The string is shorter than the schema minimum.",
    "string_pattern_mismatch": "The string must match the schema pattern.",
    "int_type": "Supply an integer.",
    "int_parsing": "Supply an integer.",
    "int_from_float": "Supply an integer, not a fractional number.",
    "float_type": "Supply a number.",
    "float_parsing": "Supply a number.",
    "bool_type": "Supply a boolean.",
    "bool_parsing": "Supply a boolean.",
    "list_type": "Supply a JSON array.",
    "dict_type": "Supply a JSON object.",
    "model_type": "Supply an object matching this field's schema.",
    "too_short": "Supply at least the schema minimum number of items.",
    "too_long": "Supply no more than the schema maximum number of items.",
    "greater_than": "The number must exceed the schema lower bound.",
    "greater_than_equal": "The number must meet the schema lower bound.",
    "less_than": "The number must be below the schema upper bound.",
    "less_than_equal": "The number must meet the schema upper bound.",
    "finite_number": "Supply a finite number.",
}

# Model validators may add context to ValueErrors. Only exact known messages
# are public; input, arbitrary ctx, exception repr and source values stay out.
_CROSS_FIELD = {
    "split_ratios must contain train/development/test and sum to one",
    "action_ranges use nonnegative half-open assistant-turn indices",
    "max_per_label values must be nonnegative integers",
    "float16 SFT requires a loss-scaling backend; use float32 or bfloat16",
    "a separate tokenizer requires tokenizer_sha256",
    "required_stages must name defined stages without duplicates",
    "global changes cannot specify a target",
    "role and agent changes require a target",
    "unsupported evidence field; use a documented field or data./identity. path",
    "sequence supports only event selectors with min_count=1",
    "behavior rules support at most 12 nested operators",
}


class ToolInputError(ValueError):
    """An error whose public fields were constructed without private input text."""

    def __init__(self, code: str, message: str, *, issues: list[dict[str, Any]] | None = None,
                 error_type: str = "ValueError", issue_count: int | None = None):
        super().__init__(message)
        self.code = code
        self.issues = (issues or [])[:MAX_ISSUES]
        self.issue_count = len(issues or []) if issue_count is None else issue_count
        self.error_type = error_type


def validation_error(exc: ValidationError, *, known_fields: set[str] | None = None) -> ToolInputError:
    """Keep schema paths and constraints, masking user-defined dictionary keys."""
    allowed = known_fields or set()
    errors = exc.errors(include_url=False, include_input=False)
    issues = []
    for error in errors[:MAX_ISSUES]:
        path = [part if isinstance(part, int) or part in allowed else "<key>" for part in error["loc"]]
        code = error["type"]
        constraint = _CONSTRAINTS.get(code, "Check this field's schema and cross-field constraints in describe_tool.")
        cause = (error.get("ctx") or {}).get("error")
        if isinstance(cause, ValueError) and str(cause) in _CROSS_FIELD:
            constraint = str(cause)
        issue: dict[str, Any] = {"path": path, "code": code, "constraint": constraint}
        bounds = {key: value for key, value in (error.get("ctx") or {}).items()
                  if key in {"gt", "ge", "lt", "le", "min_length", "max_length"}
                  and type(value) in {int, float} and math.isfinite(value)}
        if bounds:
            issue["bounds"] = bounds
        issues.append(issue)
    return ToolInputError("invalid_arguments", "Correct the indicated fields before submitting again.",
                          issues=issues, error_type="ValidationError", issue_count=len(errors))


def error_payload(tool: str, exc: Exception) -> dict[str, Any]:
    """One public error object; a missing diagnostic is never advertised as stored."""
    tool = tool if isinstance(tool, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", tool) else "<invalid>"
    if isinstance(exc, ToolInputError):
        return {"ok": False, "tool": tool, "error_type": exc.error_type, "code": exc.code,
                "message": str(exc), "issues": exc.issues, "issue_count": exc.issue_count,
                "issues_truncated": exc.issue_count > len(exc.issues), "retryable": False}
    if isinstance(exc, ValidationError):
        return error_payload(tool, validation_error(exc))
    if isinstance(exc, json.JSONDecodeError):
        return {"ok": False, "tool": tool, "error_type": type(exc).__name__, "code": "invalid_json",
                "message": "Supply a valid JSON request object.", "line": exc.lineno, "column": exc.colno,
                "retryable": False}
    code, message = "request_failed", "Request failed; inspect the tool's constraints and any returned job status."
    if isinstance(exc, FileNotFoundError):
        code, message = "not_found", "A required local file or directory is missing. Check the supplied paths."
    elif isinstance(exc, PermissionError):
        code, message = "permission_denied", "The operation lacks access to a required local resource."
    elif isinstance(exc, ValueError):
        code, message = "request_rejected", "Request rejected. Check describe_tool and the referenced record state."
    # Mutating network calls can have succeeded before a transport error. Never
    # imply that a generic error authorizes a new submission or a blind retry.
    return {"ok": False, "tool": tool, "error_type": type(exc).__name__, "code": code,
            "message": message, "retryable": False}
