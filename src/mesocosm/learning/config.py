"""Lightweight training contracts; importing these never loads a GPU library."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComputeDType = Literal["bfloat16", "float16"]


class TrainingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PolicyIdentity(TrainingModel):
    """Identity of the exact base snapshot and optional adapter used for sampling."""

    id: str = Field(min_length=1)
    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,80}$")
    base_model: str = Field(min_length=1)
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dtype: ComputeDType = "bfloat16"
    adapter_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    adapter_path: str | None = None

    @model_validator(mode="after")
    def adapter_identity(self) -> PolicyIdentity:
        if (self.adapter_sha256 is None) != (self.adapter_path is None):
            raise ValueError("adapter_path and adapter_sha256 must be supplied together")
        if self.id != policy_id(self.base_model, self.base_revision, self.base_sha256, self.adapter_sha256, self.dtype):
            raise ValueError("policy id does not match its checkpoint content identity")
        return self


def policy_id(base_model: str, base_revision: str, base_sha256: str, adapter_sha256: str | None,
              dtype: ComputeDType = "bfloat16") -> str:
    identity = {"base_model": base_model, "base_revision": base_revision,
                "base_sha256": base_sha256, "adapter_sha256": adapter_sha256, "dtype": dtype}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
