"""Exact worker relocation contract for lineage-qualified single-world jobs."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

from mesocosm.core.config import absolutize_manifest
from mesocosm.core.manifest import WorldManifest


def verify_runtime_manifest(packaged: dict[str, Any], actual: dict[str, Any], input_base: pathlib.Path) -> None:
    expected = copy.deepcopy(packaged)
    absolutize_manifest(expected, input_base)
    # These are the ONLY values replaced by DockerBackend._bind. In particular,
    # never strip model routes, agent budgets, task content or other gateway options.
    expected["name"] = actual["name"]
    for field in ("url", "api_key", "upstream_key", "run_label"):
        expected["gateway"][field] = actual["gateway"][field]
    if len(expected["agents"]) != len(actual["agents"]):
        raise ValueError("runtime agent count changed")
    for target, observed in zip(expected["agents"], actual["agents"], strict=True):
        target["url"] = observed["url"]
    if WorldManifest.model_validate(expected) != WorldManifest.model_validate(actual):
        raise ValueError("runtime scientific configuration differs from the frozen input package")
