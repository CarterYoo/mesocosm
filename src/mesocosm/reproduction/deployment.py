"""Consume owned SFT artifacts and existing vLLM APIs on a shared-filesystem venue."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from mesocosm.core.serving import ServingIdentity
from mesocosm.learning.checkpoints import checkpoint_sha256, resolve_base
from mesocosm.learning.config import PolicyIdentity
from mesocosm.reproduction.store import Store, digest, file_hash


def training_artifact(store: Store, job_id: str) -> dict[str, Any]:
    job = store.job(job_id)
    if job.get("kind") != "training" or job.get("status") != "completed":
        raise ValueError("deployment requires a completed owned SFT job")
    output = store.directory("jobs", job_id) / "output"
    if pathlib.Path(job["output_path"]).resolve() != output.resolve():
        raise ValueError("training output is outside its owned job")
    result = job.get("result")
    if not isinstance(result, dict) or result != json.loads((output / "result.json").read_text()):
        raise ValueError("training result differs from the durable job receipt")
    config = {key: value for key, value in job["config"].items() if key != "resume"}
    from mesocosm.reproduction.dataset import load_dataset

    manifest, _ = load_dataset(store.directory("datasets", job["dataset_id"]))
    if manifest != store.get(job["dataset_id"], "dataset") or result["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("training dataset differs from its owned immutable input")
    declaration = {"identity": result["identity"], "dataset_id": result["dataset_id"],
                   "config": config, "source_pins": job["training_sources"]}
    identity = digest({key: value for key, value in declaration.items() if key != "identity"})
    if (result.get("status") != "completed" or result.get("optimizer_steps", 0) < 1
            or result.get("config") != config or result.get("source_pins") != job["training_sources"]
            or declaration != json.loads((output / "declaration.json").read_text())
            or identity != result["identity"]):
        raise ValueError("training artifact has inconsistent configuration or update provenance")
    checkpoint = output / f"checkpoint-{result['optimizer_steps']:08d}"
    adapter = checkpoint / "adapter"
    if result.get("adapter_path") != str(adapter) or result.get("checkpoint") != str(checkpoint):
        raise ValueError("training adapter does not belong to the final committed step")
    state = json.loads((checkpoint / "checkpoint.json").read_text())
    if state.get("identity") != identity or state.get("step") != result["optimizer_steps"]:
        raise ValueError("checkpoint identity or step differs from the completed training receipt")
    for name, expected in state["files"].items():
        path = checkpoint / name
        if (path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve())
                or file_hash(path) != expected):
            raise ValueError("committed training checkpoint changed")
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("owned training output cannot contain symlinks")
    if checkpoint_sha256(adapter, adapter=True) != result["adapter_sha256"]:
        raise ValueError("adapter hash differs from the completed training receipt")
    if config.get("tokenizer") is not None:
        raise ValueError("separate SFT tokenizer deployment is unsupported")
    return {"training_job_id": job_id, "dataset_id": job["dataset_id"], "identity": identity,
            "config": config, "adapter_path": str(adapter), "adapter_sha256": result["adapter_sha256"],
            "result_sha256": digest(result), "checkpoint_sha256": file_hash(checkpoint / "checkpoint.json")}


def verify_artifacts(policy: PolicyIdentity) -> None:
    # Never download or substitute a checkpoint during deployment or reconciliation.
    if checkpoint_sha256(resolve_base(policy, local_files_only=True)) != policy.base_sha256:
        raise ValueError("deployment base checkpoint content mismatch")
    if policy.adapter_path and checkpoint_sha256(policy.adapter_path, adapter=True) != policy.adapter_sha256:
        raise ValueError("deployment adapter content mismatch")


def verify_receipt(policy: PolicyIdentity, receipt: dict[str, Any]) -> None:
    if receipt.get("policy_id") != policy.id or receipt.get("model") != policy.model:
        raise ValueError("serving receipt differs from the requested policy")
    probe = receipt.get("probe_response", {})
    choices, usage = probe.get("choices"), probe.get("usage", {})
    if (probe.get("error") is not None or probe.get("model") != policy.model
            or not isinstance(probe.get("id"), str) or not probe["id"]
            or not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") not in {"stop", "length"}
            or not isinstance(choices[0].get("text"), str)
            or type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != 1):
        raise ValueError("serving probe lacks a complete measured one-token response")
    if not policy.adapter_path:
        registration = receipt["registration"]
        if registration.get("parent") is not None or registration.get("root") != receipt["configuration"]["model"]:
            raise ValueError("baseline alias must expose the actual base, not a registered adapter")


def serving_identity(deployment_id: str, value: dict[str, Any]) -> dict[str, Any]:
    policy = PolicyIdentity.model_validate(value["policy"])
    return ServingIdentity(deployment_id=deployment_id, policy_id=policy.id,
                           base_sha256=policy.base_sha256, adapter_sha256=policy.adapter_sha256).model_dump()
