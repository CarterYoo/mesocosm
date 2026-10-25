"""Validate retained native images before bypassing a heavyweight harness build."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from collections.abc import Awaitable, Callable
from typing import Any

PREFIX = "org.mesocosm.native."
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
Command = Callable[..., Awaitable[bytes]]


class PreparedImageError(ValueError):
    """A selected prepared image or its frozen import proof cannot be verified."""


def image_fingerprint(image: dict[str, Any]) -> str:
    """Exclude mutable local tags, while retaining image configuration and layer identities."""
    fields = {key: image.get(key) for key in ("Id", "Config", "RootFS", "Os", "Architecture", "Variant")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_import_receipt() -> tuple[dict[str, Any] | None, str | None]:
    path = os.environ.get("MESOCOSM_PREPARED_IMAGE_RECEIPT")
    expected = os.environ.get("MESOCOSM_PREPARED_IMAGE_SHA256")
    if not path and not expected:
        return None, None
    if not path or not expected:
        raise PreparedImageError("prepared-image import requires both receipt path and SHA-256")
    source = pathlib.Path(path)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PreparedImageError("prepared-image import receipt hash mismatch")
    data = json.loads(raw)
    if data.get("schema_version") != 1 or data.get("kind") != "mesocosm.native-image-import.v1":
        raise PreparedImageError("unsupported prepared-image import receipt")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise PreparedImageError("prepared-image import needs explicit entries")
    identities = [entry.get("environment_sha256") for entry in entries]
    if len(set(identities)) != len(identities):
        raise PreparedImageError("duplicate prepared-image import identity")
    for entry in entries:
        proof = entry.get("proof")
        if not isinstance(proof, dict) or not proof.get("path") or not proof.get("sha256"):
            raise PreparedImageError("prepared-image import needs a frozen operator build proof")
        relative = pathlib.Path(proof["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PreparedImageError("prepared-image proof must remain within its receipt directory")
        candidate = source.parent / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(source.parent.resolve()) or \
                hashlib.sha256(candidate.read_bytes()).hexdigest() != proof["sha256"]:
            raise PreparedImageError("prepared-image operator build proof hash mismatch")
    return data, expected


def expected_labels(environment_sha256: str, recipe_sha256: str, version: str,
                    platform: str | None) -> dict[str, str]:
    return {PREFIX + name: value for name, value in {
        "schema": "1", "environment-sha256": environment_sha256,
        "recipe-sha256": recipe_sha256, "opencode-version": version,
        "platform": platform or "native",
    }.items()}


async def _inspect(command: Command, reference: str) -> dict[str, Any]:
    rows = json.loads((await command("docker", "image", "inspect", reference)).decode())
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise PreparedImageError("prepared-image inspect must return exactly one image")
    return rows[0]


def validate_lineage(agent: dict[str, Any], base: dict[str, Any], platform: str | None) -> None:
    for image in (agent, base):
        if not IMAGE_ID.fullmatch(image.get("Id", "")):
            raise PreparedImageError("prepared-image identity must be an immutable SHA-256")
    layers = base.get("RootFS", {}).get("Layers")
    derived = agent.get("RootFS", {}).get("Layers")
    if not layers or not derived or len(derived) <= len(layers) or derived[:len(layers)] != layers:
        raise PreparedImageError("prepared harness does not extend the exact original image layers")
    for field in ("Os", "Architecture", "Variant"):
        if agent.get(field) != base.get(field):
            raise PreparedImageError("prepared-image platform differs from original base")
    actual = "/".join(str(base.get(key) or "") for key in ("Os", "Architecture"))
    if platform and platform != actual and platform != actual + "/" + str(base.get("Variant") or ""):
        raise PreparedImageError("prepared-image platform differs from task platform")


async def find_prepared_image(command: Command, *, tag: str, labels: dict[str, str],
                              platform: str | None, imported: dict[str, Any] | None,
                              import_sha256: str | None) -> dict[str, Any] | None:
    entry = None
    if imported is not None:
        matches = [item for item in imported["entries"]
                   if item.get("environment_sha256") == labels[PREFIX + "environment-sha256"]]
        if len(matches) != 1:
            raise PreparedImageError("frozen import does not contain this exact task environment")
        entry = matches[0]
        if entry.get("labels") != labels:
            raise PreparedImageError("prepared-image import source, recipe, version or platform drift")
        for field in ("agent_image_id", "base_image_id"):
            if not IMAGE_ID.fullmatch(entry.get(field, "")):
                raise PreparedImageError("prepared-image import must name immutable image IDs")
        agent = await _inspect(command, entry["agent_image_id"])
        base_id = entry["base_image_id"]
    else:
        try:
            agent = await _inspect(command, tag)
        except (RuntimeError, OSError):
            return None
        actual = agent.get("Config", {}).get("Labels") or {}
        if not any(key.startswith(PREFIX) for key in actual):
            return None  # Existing unlabeled tags are never adopted automatically.
        if any(actual.get(key) != value for key, value in labels.items()):
            raise PreparedImageError("prepared-image cache labels differ from source identity")
        base_id = actual.get(PREFIX + "base-image-id", "")
        if not IMAGE_ID.fullmatch(base_id):
            raise PreparedImageError("prepared-image cache has no immutable base identity")
    base = await _inspect(command, base_id)
    if base.get("Id") != base_id:
        raise PreparedImageError("prepared base image ID mismatch")
    if entry is not None:
        if agent.get("Id") != entry["agent_image_id"]:
            raise PreparedImageError("prepared agent image ID mismatch")
        if image_fingerprint(agent) != entry.get("agent_fingerprint") or \
                image_fingerprint(base) != entry.get("base_fingerprint"):
            raise PreparedImageError("prepared-image import configuration or layer fingerprint mismatch")
    validate_lineage(agent, base, platform)
    return {"agent": agent, "base": base, "evidence": {
        "kind": "frozen_import" if entry is not None else "validated_labels",
        "agent_image_id": agent["Id"], "base_image_id": base["Id"],
        "environment_sha256": labels[PREFIX + "environment-sha256"],
        "recipe_sha256": labels[PREFIX + "recipe-sha256"],
        "import_receipt_sha256": import_sha256,
        "operator_proof_sha256": entry["proof"]["sha256"] if entry is not None else None,
    }}
