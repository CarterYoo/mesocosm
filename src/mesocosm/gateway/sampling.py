"""Fail-closed application of the opt-in per-model request contract."""

from __future__ import annotations

from typing import Any

from mesocosm.core.sampling import SamplingSpec

# This narrow contract covers single-choice chat/tool calls, not arbitrary vLLM extensions.
# Unknown fields could select another distribution, decoding mode or nested override.
INPUT_FIELDS = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
                "stream_options", "max_tokens", "max_completion_tokens", "user", "metadata",
                "logprobs", "top_logprobs", "return_token_ids", "n"}
MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id", "refusal"}


def apply_sampling(body: dict[str, Any], sampling: SamplingSpec) -> dict[str, Any]:
    parameters = sampling.model_dump()
    unsupported = set(body) - INPUT_FIELDS - parameters.keys()
    if unsupported:
        raise ValueError(f"unsupported sampling request fields: {sorted(unsupported)}")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise ValueError("model sampling requires n=1")
    if type(body.get("stream", False)) is not bool:
        raise ValueError("model sampling requires a boolean stream flag")
    for message in body.get("messages", []):
        if set(message) - MESSAGE_FIELDS:
            raise ValueError("unsupported sampling message fields")
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            not isinstance(part, dict) or set(part) != {"type", "text"}
            or part["type"] != "text" or not isinstance(part["text"], str) for part in content
        ):
            raise ValueError("model sampling supports text/tool messages only")
    supplied = {key: body[key] for key in parameters if key in body}
    # Presence is deliberate: explicit null, bool, nonfinite values and numeric strings fail.
    validated = SamplingSpec.model_validate(parameters | supplied)
    if validated != sampling:
        conflicts = [key for key in supplied if getattr(validated, key) != parameters[key]]
        raise ValueError(f"request conflicts with model sampling: {conflicts}")
    return {**body, **parameters}
