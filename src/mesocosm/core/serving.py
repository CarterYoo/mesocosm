"""Control-plane serving attribution, separate from numerical weight verification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ServingIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deployment_id: str = Field(pattern=r"^deployment-[a-f0-9]{64}$")
    policy_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


def response_identity(response: dict, expected: str) -> str:
    if "conflicting_model_ids" in response.get("stream_errors", []):
        return "conflicting"
    if not isinstance(response.get("model"), str) or not response["model"]:
        return "missing"
    return "matched" if response["model"] == expected else "mismatched"
