"""Explicit request sampling; this is not an engine or world replay seed."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)

    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    seed: int = Field(ge=0, le=2**63 - 1)
    top_k: int = -1
    min_p: float = Field(default=0.0, ge=0, le=1)

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, value: int) -> int:
        if value != -1 and value < 1:
            raise ValueError("top_k must be -1 (disabled) or a positive integer")
        return value
