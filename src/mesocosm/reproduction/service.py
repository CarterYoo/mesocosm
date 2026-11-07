"""Research-agent tools over immutable evidence and the existing execution worker.

One trusted operator owns this store. This API is not a security boundary for an
agent that already has filesystem access. Agent tools never accept a shell command.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import pathlib
import re
import shutil
import tempfile
import time
from typing import Any, Literal

import yaml

from mesocosm.core.config import absolutize_manifest
from mesocosm.execution.client import ExecutionClient, worker_url
from mesocosm.execution.package import pack_submission, unpack_submission
from mesocosm.execution.protocol import TERMINAL
from mesocosm.reproduction import behavior, collective, evidence, factors
from mesocosm.reproduction.store import Store, digest, file_hash, write_json

TRANSCRIPT_MAX_FILE_BYTES = 256 * 1024 * 1024
TRANSCRIPT_MAX_BYTES = 512 * 1024 * 1024
TRANSCRIPT_MAX_FILES = 10_000

TOOLS = (
    "inspect_transcript", "define_behavior", "list_factors", "configure_experiment", "validate_experiment",
    "prepare_dataset", "submit_training", "run_experiment", "get_job_status", "snapshot_experiment",
    "fork_experiment", "evaluate_candidate", "compare_candidates", "export_reproducer",
    "deploy_policy", "evaluate_run", "plan_factorial", "advance_factorial", "report_factorial",
    "describe_tool", "read_record", "read_events", "list_records", "evaluate_collective",
)


def _held_out(value: Any) -> bool:
    if isinstance(value, dict):
        if any(value.get(key) == "test" for key in ("split", "partition")):
            return True
        return any(_held_out(item) for key, item in value.items()
                   if key in {"metadata", "source", "split_metadata", "lineage", "summary"})
    return False


class ResearchWorkbench:
    def __init__(self, root: str | pathlib.Path, *, allow_test_evaluation: bool = False,
                 allowed_workers: dict[str, str] | None = None,
                 allowed_servers: dict[str, str | None] | None = None):
        self.store = Store(root)
        self.allow_test_evaluation = allow_test_evaluation
        self.allowed_workers = {worker_url(url): variable for url, variable in (allowed_workers or {}).items()}
        self.allowed_servers = {worker_url(url): variable for url, variable in (allowed_servers or {}).items()}
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", value)
               for value in self.allowed_workers.values()):
            raise ValueError("worker credentials must name operator-approved environment variables")
        if any(value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", value))
               for value in self.allowed_servers.values()):
            raise ValueError("server credentials must name operator-approved environment variables")

    def _server(self, endpoint: str):
        from mesocosm.learning.serving import VLLMPolicyServer

        endpoint = worker_url(endpoint)
        if endpoint not in self.allowed_servers:
            raise ValueError("serving endpoint is not operator-approved for shared-filesystem deployment")
        variable = self.allowed_servers[endpoint]
        if variable and not os.environ.get(variable):
            raise ValueError("serving credential is unavailable")
        return VLLMPolicyServer(endpoint, os.environ[variable] if variable else None)

    async def deploy_policy(self, server: str, model: str, base_model: str, base_revision: str,
                            base_sha256: str, idempotency_key: str,
                            dtype: Literal["bfloat16", "float16"] = "bfloat16",
                            training_job_id: str | None = None,
                            external_adapter: dict[str, Any] | None = None) -> dict[str, Any]:
        """Verify a base or load a hashed local adapter; requires shared checkpoint paths with vLLM."""
        with self.store.lease("serving:" + worker_url(server) + ":" + model):
            return await self._deploy_policy(server, model, base_model, base_revision, base_sha256,
                                             idempotency_key, dtype, training_job_id, external_adapter)

    async def _deploy_policy(self, server: str, model: str, base_model: str, base_revision: str,
                             base_sha256: str, idempotency_key: str, dtype: Literal["bfloat16", "float16"],
                             training_job_id: str | None, external_adapter: dict[str, Any] | None) -> dict[str, Any]:
        from mesocosm.learning.config import PolicyIdentity, policy_id
        from mesocosm.reproduction.deployment import external_artifact, training_artifact, verify_artifacts

        endpoint = worker_url(server)
        if endpoint not in self.allowed_servers:
            raise ValueError("serving endpoint is not operator-approved")
        if training_job_id is not None and external_adapter is not None:
            raise ValueError("training_job_id and external_adapter are mutually exclusive")
        external, rank = (external_artifact(external_adapter, base_model)
                          if external_adapter is not None else (None, None))
        artifact = training_artifact(self.store, training_job_id) if training_job_id else None
        if artifact and (artifact["config"]["base_sha256"] != base_sha256 or artifact["config"]["dtype"] != dtype):
            raise ValueError("serving base/dtype differs from the actual SFT job")
        adapter_sha = artifact["adapter_sha256"] if artifact else external["sha256"] if external else None
        if dtype not in {"bfloat16", "float16"}:
            raise ValueError("serving currently requires bfloat16/float16; no implicit SFT dtype conversion")
        policy = PolicyIdentity.model_validate({"id": policy_id(base_model, base_revision, base_sha256,
                                                                adapter_sha, dtype),
            "model": model, "base_model": base_model, "base_revision": base_revision,
            "base_sha256": base_sha256, "dtype": dtype, "adapter_sha256": adapter_sha,
            "adapter_path": artifact["adapter_path"] if artifact else external["path"] if external else None})
        args = {"server": endpoint, "policy": policy.model_dump(mode="json"), "training_artifact": artifact,
                "server_token_env": self.allowed_servers[endpoint]}
        if external is not None:
            args["external_adapter"] = external
        identifier = "job-" + digest({"key": idempotency_key, "tool": "deploy_policy", "args": args})
        record = {"kind": "deployment", "status": "activating", **args, "created_at": time.time()}
        cached = self.store.begin(idempotency_key, "deploy_policy", args, job=(identifier, record))
        if cached is not None:
            return cached
        client = None
        activation_started = False
        try:
            await asyncio.to_thread(verify_artifacts, policy)
            client = self._server(endpoint)
            capacity = await client.verify_adapter_rank(policy, rank) if external else None
            activation_started = True
            receipt = await (client.load(policy) if policy.adapter_path else client.verify(policy))
            if external:
                receipt["external_adapter_capacity"] = capacity
            await asyncio.to_thread(verify_artifacts, policy)
            result = self._complete_deployment(identifier, record, receipt)
        except Exception as exc:
            if external and not activation_started:
                record.update(status="rejected", error_type=type(exc).__name__)
                self.store.save_job(identifier, record)
                self.store.finish(idempotency_key, None, error=type(exc).__name__)
                raise ValueError(str(exc)) from exc
            # A load may have succeeded before the connection failed. Reconcile by
            # verification only; never reload/replace an alias automatically.
            record.update(status="activation_uncertain", error_type=type(exc).__name__)
            self.store.save_job(identifier, record)
            result = {"job_id": identifier, "status": record["status"]}
        finally:
            if client is not None:
                await client.aclose()
        self.store.finish(idempotency_key, result)
        return result

    def _complete_deployment(self, job_id: str, job: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
        from mesocosm.learning.config import PolicyIdentity
        from mesocosm.reproduction.deployment import verify_receipt

        verify_receipt(PolicyIdentity.model_validate(job["policy"]), receipt)
        value = {"schema_version": 1, "server": job["server"], "server_token_env": job["server_token_env"],
                 "policy": job["policy"], "training_artifact": job["training_artifact"], "receipt": receipt,
                 "verified_at": time.time(), "deployment_job_id": job_id,
                 "evidence_level": "shared_filesystem_artifact_hash_and_serving_api",
                 "numerical_weight_verification": "unverified"}
        if job.get("external_adapter") is not None:
            value.update(external_adapter=job["external_adapter"],
                         evidence_level="external_adapter_hash_and_serving_api", training_provenance="unverified")
        identifier = self.store.put("deployment", value)
        job.update(status="ready", deployment_id=identifier)
        self.store.save_job(job_id, job)
        return {"job_id": job_id, "status": "ready", "deployment_id": identifier,
                "evidence_level": value["evidence_level"], "numerical_weight_verification": "unverified"}

    async def _verify_deployment(self, identifier: str) -> dict[str, Any]:
        from mesocosm.learning.config import PolicyIdentity
        from mesocosm.reproduction.deployment import (
            external_artifact,
            training_artifact,
            verify_artifacts,
            verify_receipt,
        )

        value = self.store.get(identifier, "deployment")
        if (value["server"] not in self.allowed_servers
                or self.allowed_servers[value["server"]] != value["server_token_env"]):
            raise ValueError("deployment server/credential approval changed")
        artifact = value["training_artifact"]
        if artifact and await asyncio.to_thread(training_artifact, self.store, artifact["training_job_id"]) != artifact:
            raise ValueError("deployed training artifact changed")
        policy = PolicyIdentity.model_validate(value["policy"])
        external = value.get("external_adapter")
        rank = (await asyncio.to_thread(external_artifact, external, policy.base_model))[1] if external else None
        await asyncio.to_thread(verify_artifacts, policy)
        client = self._server(value["server"])
        try:
            capacity = await client.verify_adapter_rank(policy, rank) if external else None
            receipt = await client.verify(policy)
            if external:
                receipt["external_adapter_capacity"] = capacity
            verify_receipt(policy, receipt)
            await asyncio.to_thread(verify_artifacts, policy)
            return {"deployment_id": identifier, "verified_at": time.time(), "receipt": receipt}
        finally:
            await client.aclose()

    async def _serving_preflight(self, experiment: dict[str, Any]) -> dict[str, Any]:
        from mesocosm.reproduction.deployment import agent_bindings

        receipts = {}
        for identifier in sorted(set(agent_bindings(experiment).values())):
            receipts[identifier] = await self._verify_deployment(identifier)
        return receipts

    def _worker(self, endpoint: str, token_env: str) -> str:
        endpoint = worker_url(endpoint)
        if self.allowed_workers.get(endpoint) != token_env:
            raise ValueError("worker endpoint and credential variable are not operator-approved")
        return endpoint

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from mesocosm.reproduction.contracts import validate_arguments
        from mesocosm.reproduction.views import present

        if tool not in TOOLS:
            raise ValueError("unknown research tool")
        function = getattr(self, tool)
        arguments = validate_arguments(getattr(type(self), tool), arguments)
        if inspect.iscoroutinefunction(function):
            result = await function(**arguments)
        else:
            # Synchronous imports must not block another caller's status request.
            result = await asyncio.to_thread(function, **arguments)
        return await asyncio.to_thread(present, self, tool, result)

    @classmethod
    def tools(cls) -> dict[str, Any]:
        return {"schema_version": 2, "detail_tool": "describe_tool", "response_limit_bytes": 24 * 1024, "tools": [
            {"name": name, "description": inspect.getdoc(getattr(cls, name)),
             "parameters": [{"name": key, "required": value.default is inspect.Parameter.empty,
                             "type": str(value.annotation),
                             **({"default": value.default} if value.default is not inspect.Parameter.empty else {})}
                            for key, value in inspect.signature(getattr(cls, name)).parameters.items()
                            if key != "self"]}
            for name in TOOLS], "trust_boundary": "single trusted operator; not a multi-tenant sandbox"}

    def describe_tool(self, name: str) -> dict[str, Any]:
        """Read one tool's detailed input schema, example and operational boundaries."""
        from mesocosm.reproduction.contracts import describe_tool
        if name not in TOOLS:
            raise ValueError("unknown research tool")
        return describe_tool(type(self), name)

    def read_record(self, record_id: str, pointer: str = "", offset: int = 0, limit: int = 20,
                    index_path: list[int] | None = None, expected_version: str | None = None) -> dict[str, Any]:
        """Navigate a saved object/job by JSON pointer; pages and long text are bounded, with explicit cursors."""
        from mesocosm.reproduction.views import read_record
        return read_record(self, record_id, pointer, offset, limit, index_path, expected_version)

    def read_events(self, transcript_id: str, agent_id: str | None = None, kind: str | None = None,
                    event_id: str | None = None, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Read a bounded filtered event page; use returned pointers to inspect oversized events in full."""
        from mesocosm.reproduction.views import read_events
        return read_events(self, transcript_id, agent_id, kind, event_id, offset, limit)

    def list_records(self, kind: str = "experiment", offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Rediscover persisted records or jobs after context loss without replaying prior tool responses."""
        from mesocosm.reproduction.views import list_records
        return list_records(self, kind, offset, limit)

    def inspect_transcript(self, path: str) -> dict[str, Any]:
        """Normalize recorded evidence; narration never establishes executed behavior."""
        supplied = pathlib.Path(path)
        if supplied.is_symlink():
            raise ValueError("transcript input cannot be a symlink")
        source = supplied.resolve(strict=True)
        if source.is_file() and source.stat().st_size > TRANSCRIPT_MAX_FILE_BYTES:
            raise ValueError("transcript input exceeds file-size limit")
        value = evidence.inspect_transcript(source)
        entries = value["source"]["files"]
        if len(entries) > TRANSCRIPT_MAX_FILES or sum(entry["bytes"] for entry in entries) > TRANSCRIPT_MAX_BYTES:
            raise ValueError("transcript sources exceed frozen input limits")
        if _held_out(value) and not self.allow_test_evaluation:
            raise ValueError("declared held-out evidence import requires the independent operator")
        identifier = self.store.put("transcript", value)
        destination = self.store.directory("transcripts", identifier)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
                staged = pathlib.Path(temporary) / "sources"
                staged.mkdir(mode=0o700)
                for entry in value["source"]["files"]:
                    original = source / entry["path"] if source.is_dir() else source
                    if (original.is_symlink() or not original.is_file()
                            or original.stat().st_size > TRANSCRIPT_MAX_FILE_BYTES
                            or original.stat().st_size != entry["bytes"]
                            or source.is_dir() and not original.resolve().is_relative_to(source)):
                        raise ValueError("transcript source is unsafe or exceeds its recorded size")
                    if source.is_dir() and any(parent.is_symlink() for parent in original.parents if parent != source):
                        raise ValueError("transcript source path cannot traverse symlinks")
                    content = original.read_bytes()
                    if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                        raise ValueError("transcript original bytes changed during import")
                    target = staged / entry["path"]
                    if not target.resolve().is_relative_to(staged.resolve()):
                        raise ValueError("transcript source path escapes its frozen directory")
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target.write_bytes(content)
                    target.chmod(0o600)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                staged.rename(destination)
        self._transcript_sources(identifier)
        self.store.audit("transcript.imported", {"transcript_id": identifier})
        return {"transcript_id": identifier, "transcript": value}

    def _transcript_sources(self, identifier: str) -> pathlib.Path:
        value = self.store.get(identifier, "transcript")
        destination = self.store.directory("transcripts", identifier)
        expected = {entry["path"]: entry["sha256"] for entry in value["source"]["files"]}
        actual: dict[str, str] = {}
        total = 0
        for path in destination.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("frozen transcript contains unsafe paths")
            if path.is_file():
                size = path.stat().st_size
                total += size
                if (size > TRANSCRIPT_MAX_FILE_BYTES or total > TRANSCRIPT_MAX_BYTES
                        or len(actual) >= TRANSCRIPT_MAX_FILES):
                    raise ValueError("frozen transcript exceeds input limits")
                actual[path.relative_to(destination).as_posix()] = file_hash(path)
        if actual != expected:
            raise ValueError("frozen transcript source bytes changed or are missing")
        return destination

    def define_behavior(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Freeze a versioned stage rubric or closed-domain collective detector before candidate access."""
        value = collective.freeze(spec) if spec.get("kind") == "collective" else behavior.define_behavior(spec)
        identifier = self.store.put("behavior", value)
        self.store.audit("behavior.defined", {"behavior_id": identifier})
        return {"behavior_id": identifier, "definition": value}

    def list_factors(self, experiment_id: str | None = None, names: list[str] | None = None,
                     prefix: str | None = None, detail: bool = False, offset: int = 0,
                     limit: int = 20) -> dict[str, Any]:
        """Search a bounded factor index; request detail for selected fields and their exact schemas."""
        from mesocosm.reproduction.views import limits, page, record
        limits(offset, limit)
        world = record(self, experiment_id)["configuration"]["manifest"] if experiment_id else None
        catalog = factors.list_factors(world)
        if names is not None and set(names) - {item["name"] for item in catalog}:
            raise ValueError("unknown factor name; search the factor index first")
        selected = [item for item in catalog if (names is None or item["name"] in names)
                    and (prefix is None or item["name"].startswith(prefix))]
        if not detail:
            fields = {"name", "status", "category", "scopes", "change_timing", "native_acceptance"}
            selected = [{key: value for key, value in item.items() if key in fields} for item in selected]
        # Detailed schemas can be larger than a page; the dispatcher's retained-result view bounds those.
        if detail:
            stop = min(offset + limit, len(selected))
            return {"factors": selected[offset:stop], "total": len(selected), "offset": offset,
                    "next_offset": stop if stop < len(selected) else None, "eof": stop >= len(selected)}
        result = page(selected, offset, limit)
        return {"factors": result.pop("items"), **result}

    def configure_experiment(self, base_world: str, behavior_id: str, changes: list[dict[str, Any]],
                             roles: dict[str, list[str]] | None = None,
                             split: str = "development", deployment_id: str | None = None,
                             agent_deployments: dict[str, str] | None = None) -> dict[str, Any]:
        """Freeze a rubric, scoped factors and all portable world inputs into one candidate."""
        self.store.get(behavior_id, "behavior")
        if split not in {"train", "development", "test"}:
            raise ValueError("split must be train, development or test")
        source = pathlib.Path(base_world).resolve(strict=True)
        original = source.read_bytes()
        raw = yaml.safe_load(original)
        if deployment_id is not None and agent_deployments is not None:
            raise ValueError("deployment_id and agent_deployments are mutually exclusive")
        if deployment_id or agent_deployments is not None:
            from mesocosm.core.manifest import WorldManifest
            from mesocosm.reproduction.deployment import serving_identity

            world = WorldManifest.model_validate(raw)
            if deployment_id and any(world.model_for(agent).id != world.model.id for agent in world.agents):
                raise ValueError("deployment binding currently requires a shared policy for all agents")
            if any(change.get("factor", "").startswith(("model.", "agent.model"))
                   or change.get("factor") == "models" for change in changes):
                raise ValueError("model factors cannot override a bound deployment")
            if agent_deployments is not None:
                bindings = agent_deployments
            else:
                assert deployment_id is not None
                bindings = {agent.id: deployment_id for agent in world.agents}
            if set(bindings) != {agent.id for agent in world.agents}:
                raise ValueError("agent_deployments must name every agent exactly once; missing or unknown agent id")
            deployments = {identifier: self.store.get(identifier, "deployment")
                           for identifier in set(bindings.values())}
            bases = {(worker_url(value["server"]), *(value["policy"][key] for key in
                      ("base_model", "base_revision", "base_sha256", "dtype"))) for value in deployments.values()}
            if len(bases) != 1:
                raise ValueError("agent deployments must share endpoint, base model, revision, base hash and dtype")
            aliases = [value["policy"]["model"] for value in deployments.values()]
            if len(set(aliases)) != len(aliases):
                raise ValueError("distinct deployments must have distinct model aliases")
            raw = world.model_dump(mode="json")
            old_model = copy.deepcopy(raw["model"])
            bound_models = {}
            for identifier, deployed in sorted(deployments.items()):
                policy = deployed["policy"]
                bound_models[policy["model"]] = {**old_model, "id": policy["model"], "hf_repo": policy["base_model"],
                    "revision": policy["base_revision"], "upstream": deployed["server"],
                    "api_key_env": deployed["server_token_env"], "kind": "vllm",
                    "serving_identity": serving_identity(identifier, deployed)}
            if deployment_id:
                raw["model"] = bound_models.pop(deployments[deployment_id]["policy"]["model"])
            elif raw["model"]["id"] in bound_models:
                raw["model"] = bound_models.pop(raw["model"]["id"])
            raw["models"] = [model for model in raw["models"] if model["id"] not in aliases
                             and (deployment_id or model["id"] == raw["judge"]["monitor"]["model"])]
            raw["models"].extend(bound_models.values())
            if (raw["judge"]["monitor"]["model"] == old_model["id"] and old_model["id"] != raw["model"]["id"]
                    and old_model["id"] not in {model["id"] for model in raw["models"]}):
                raw["models"].append(old_model)
            for agent in raw["agents"]:
                agent["model"] = deployments[bindings[agent["id"]]]["policy"]["model"]
        configuration = factors.configure_experiment(raw, changes, roles)
        runtime_manifest = copy.deepcopy(configuration["manifest"])
        absolutize_manifest(runtime_manifest, source.parent)
        # The immutable configuration hashes its original normalized relative
        # paths. Packaging relocates a separate copy and has its own digest.
        with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
            directory = pathlib.Path(temporary)
            manifest = directory / "world.json"
            write_json(manifest, runtime_manifest)
            archive = directory / "world.zip"
            package = pack_submission(manifest, archive)
            value = {"schema_version": 1, "configuration": configuration, "behavior_id": behavior_id,
                     "split": split, "package": package, "archive_sha256": file_hash(archive),
                     "source_manifest_sha256": hashlib.sha256(original).hexdigest(), "parent_experiment_id": None,
                     "state_mode": "fresh_world", "source_root": str(source.parent)}
            if deployment_id:
                value["deployment_id"] = deployment_id
            if agent_deployments is not None:
                value["agent_deployments"] = dict(agent_deployments)
            identifier = self.store.put("experiment", value)
            target = self.store.directory("experiments", identifier)
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            if (target / "world.zip").exists():
                if file_hash(target / "world.zip") != value["archive_sha256"]:
                    raise ValueError("stored experiment archive changed")
            else:
                shutil.copyfile(archive, target / "world.zip")
                (target / "world.zip").chmod(0o600)
        self.store.audit("experiment.configured", {"experiment_id": identifier, "behavior_id": behavior_id})
        return {"experiment_id": identifier, "configuration": configuration, "split": split,
                "archive_sha256": value["archive_sha256"], "native_acceptance": "unverified"}

    def _archive(self, experiment_id: str) -> pathlib.Path:
        experiment = self.store.get(experiment_id, "experiment")
        archive = self.store.directory("experiments", experiment_id) / "world.zip"
        if archive.is_symlink() or file_hash(archive) != experiment["archive_sha256"]:
            raise ValueError("experiment archive identity mismatch")
        return archive

    def validate_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Validate the frozen inputs; this does not claim Linux/model readiness."""
        with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
            root = pathlib.Path(temporary) / "inputs"
            package = unpack_submission(self._archive(experiment_id), root)
            path = root / package["entrypoint"]
            result = factors.validate_experiment(yaml.safe_load(path.read_text()), root)
        return {"experiment_id": experiment_id, **result}

    def prepare_dataset(self, sources: list[str], config: dict[str, Any]) -> dict[str, Any]:
        """Freeze observation/action prefixes and group-disjoint dataset partitions."""
        from mesocosm.reproduction.dataset import prepare_dataset

        with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
            output = pathlib.Path(temporary) / "dataset"
            result = prepare_dataset([pathlib.Path(path) for path in sources], output, config)
            manifest = {key: value for key, value in result.items() if key != "dataset_path"}
            identifier = self.store.put("dataset", manifest)
            destination = self.store.directory("datasets", identifier)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.move(str(output), destination)
            from mesocosm.reproduction.dataset import load_dataset
            if load_dataset(destination)[0] != manifest:
                raise ValueError("existing dataset content does not match its immutable identity")
        self.store.audit("dataset.prepared", {"dataset_id": identifier})
        return {"dataset_id": identifier, "counts": manifest["counts"], "manifest": manifest}

    def submit_training(self, dataset_id: str, config: dict[str, Any], idempotency_key: str,
                        start: bool = True, resume_job_id: str | None = None) -> dict[str, Any]:
        """Queue real local SFT/LoRA; local pinned models only, no cloud allocation."""
        from mesocosm.reproduction.dataset import load_dataset
        from mesocosm.reproduction.jobs import start_worker
        from mesocosm.reproduction.training import SFTConfig, training_source_pins

        manifest = self.store.get(dataset_id, "dataset")
        dataset_path = self.store.directory("datasets", dataset_id)
        if load_dataset(dataset_path)[0] != manifest:
            raise ValueError("stored dataset identity mismatch")
        specification = SFTConfig.model_validate(config).model_dump(mode="json")
        if bool(specification["resume"]) != bool(resume_job_id):
            raise ValueError("resume requires both resume=true and a prior job ID")
        sources = training_source_pins()
        args = {"dataset_id": dataset_id, "config": specification, "start": start,
                "resume_job_id": resume_job_id, "training_sources": sources}
        cached = self.store.begin(idempotency_key, "submit_training", args)
        if cached is not None:
            return cached
        identifier = "job-" + digest({"key": idempotency_key, "tool": "submit_training", "args": args})
        directory = self.store.directory("jobs", identifier)
        output = directory / "output"
        try:
            directory.mkdir(parents=True, mode=0o700)
            elapsed_floor = 0.0
            if resume_job_id:
                prior = self.store.job(resume_job_id)
                if (prior["kind"] != "training" or prior["dataset_id"] != dataset_id
                        or prior["status"] not in {"cancelled", "failed", "interrupted"}
                        or not prior.get("elapsed_seconds")):
                    raise ValueError("resume requires a stopped training job with measured elapsed time")
                if prior.get("training_sources") != sources:
                    raise ValueError("resume training source code changed")
                previous_config = {**prior["config"], "resume": True}
                if previous_config != specification:
                    raise ValueError("resume configuration must match the original job")
                shutil.copytree(pathlib.Path(prior["output_path"]), output, symlinks=False)
                elapsed_floor = prior.get("elapsed_floor_seconds", 0.0) + prior["elapsed_seconds"]
            record = {"kind": "training", "status": "queued", "dataset_id": dataset_id,
                      "dataset_path": str(dataset_path), "config": specification, "output_path": str(output),
                      "training_sources": sources,
                      "resume_job_id": resume_job_id, "elapsed_floor_seconds": elapsed_floor,
                      "created_at": time.time()}
            self.store.save_job(identifier, record, new=True)
            result = {"job_id": identifier, "status": "queued", "automatic_cloud_allocation": False}
            self.store.finish(idempotency_key, result)
        except Exception as exc:
            self.store.finish(idempotency_key, None, type(exc).__name__)
            raise
        if start:
            start_worker(self.store)
        return result

    async def run_experiment(self, experiment_id: str, idempotency_key: str, worker: str,
                             token_env: str = "MESOCOSM_WORKER_TOKEN") -> dict[str, Any]:
        """Submit the frozen package exactly once to an existing Linux execution worker."""
        args = {"experiment_id": experiment_id, "worker": worker_url(worker), "token_env": token_env}
        identifier = "job-" + digest({"key": idempotency_key, "tool": "run_experiment", "args": args})
        with self.store.lease(identifier):
            return await self._run_experiment(experiment_id, idempotency_key, worker, token_env)

    async def _run_experiment(self, experiment_id: str, idempotency_key: str, worker: str,
                              token_env: str) -> dict[str, Any]:
        experiment = self.store.get(experiment_id, "experiment")
        if experiment["state_mode"] != "fresh_world":
            raise ValueError("partial-state forks require the local fork executor; remote fresh runs would lose state")
        archive = self._archive(experiment_id)
        endpoint = self._worker(worker, token_env)
        args = {"experiment_id": experiment_id, "worker": endpoint, "token_env": token_env}
        identifier = "job-" + digest({"key": idempotency_key, "tool": "run_experiment", "args": args})
        record: dict[str, Any] = {"kind": "world", "status": "submitting", "experiment_id": experiment_id,
                  "worker": endpoint, "token_env": token_env, "remote_job_id": identifier[-32:],
                  "created_at": time.time()}
        cached = self.store.begin(idempotency_key, "run_experiment", args, job=(identifier, record))
        if cached is not None:
            return cached
        try:
            if experiment.get("deployment_id") or experiment.get("agent_deployments"):
                record["serving_preflight"] = await self._serving_preflight(experiment)
                self.store.save_job(identifier, record)
            record["submission_started"] = True
            self.store.save_job(identifier, record)
            async with ExecutionClient(endpoint, os.environ.get(token_env, "")) as client:
                remote = await client.submit(archive, record["remote_job_id"], digest=experiment["package"]["digest"])
            record.update(status=remote["status"], remote=remote)
        except Exception as exc:
            # An HTTP failure may occur after acceptance. Keep the same remote identity.
            record.update(status="submission_uncertain" if record.get("submission_started") else "preflight_failed",
                          error_type=type(exc).__name__)
        self.store.save_job(identifier, record)
        result = {"job_id": identifier, "status": record["status"], "remote_job_id": record["remote_job_id"]}
        self.store.finish(idempotency_key, result)
        return result

    async def get_job_status(self, job_id: str, action: str = "inspect",
                             since_version: str | None = None) -> dict[str, Any]:
        """Inspect or act on an owned job; since_version returns compact unchanged polling responses."""
        from mesocosm.reproduction.progress import status_view

        if since_version is not None and (action != "inspect" or not re.fullmatch(r"[a-f0-9]{64}", since_version)):
            raise ValueError("since_version requires inspect and a returned state_version")
        job = self.store.job(job_id)
        resource = ("serving:" + job["server"] + ":" + job["policy"]["model"]
                    if job["kind"] == "deployment" else job_id)
        with self.store.lease(resource):
            result = await self._get_job_status(job_id, action)
            return (status_view(self.store, self.store.job(job_id), result, since_version)
                    if action == "inspect" else result)

    async def _get_job_status(self, job_id: str, action: str) -> dict[str, Any]:
        if action not in {"inspect", "fetch", "cancel", "start", "resubmit", "reconcile"}:
            raise ValueError("unsupported job action")
        job = self.store.job(job_id)
        directory = self.store.directory("jobs", job_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if job["kind"] == "factorial":
            if action != "inspect":
                raise ValueError("inspect the study or cancel its current world job explicitly")
            return self.report_factorial(job["study_id"])
        if job["kind"] == "deployment":
            if action == "inspect":
                return job
            if action == "reconcile" and job["status"] == "ready":
                deployed = self.store.get(job["deployment_id"], "deployment")
                return {"job_id": job_id, "status": "ready", "deployment_id": job["deployment_id"],
                        "evidence_level": deployed["evidence_level"],
                        "numerical_weight_verification": "unverified"}
            if action != "reconcile" or job["status"] not in {"activation_uncertain", "activating"}:
                raise ValueError("deployment only supports inspect or explicit uncertain-load reconciliation")
            from mesocosm.learning.config import PolicyIdentity
            from mesocosm.reproduction.deployment import external_artifact, training_artifact, verify_artifacts

            artifact = job["training_artifact"]
            if artifact and training_artifact(self.store, artifact["training_job_id"]) != artifact:
                raise ValueError("training artifact changed before reconciliation")
            if (job["server"] not in self.allowed_servers
                    or self.allowed_servers[job["server"]] != job["server_token_env"]):
                raise ValueError("deployment server/credential approval changed")
            policy = PolicyIdentity.model_validate(job["policy"])
            external = job.get("external_adapter")
            rank = (await asyncio.to_thread(external_artifact, external, policy.base_model))[1] if external else None
            await asyncio.to_thread(verify_artifacts, policy)
            client = self._server(job["server"])
            try:
                capacity = await client.verify_adapter_rank(policy, rank) if external else None
                receipt = await client.verify(policy)
                if external:
                    receipt["external_adapter_capacity"] = capacity
                await asyncio.to_thread(verify_artifacts, policy)
                return self._complete_deployment(job_id, job, receipt)
            finally:
                await client.aclose()
        if action == "reconcile":
            raise ValueError("reconcile is only for an uncertain deployment")
        if job["kind"] == "world" and job["status"] == "preflight_failed":
            if action != "inspect":
                raise ValueError("preflight failed before submission; declare a new attempt after repair")
            return job
        if job["kind"] == "training":
            if action in {"fetch", "resubmit"}:
                raise ValueError("training results are local; use explicit resume for another attempt")
            if action == "start":
                from mesocosm.reproduction.jobs import start_worker
                start_worker(self.store)
            if action == "cancel":
                # The trainer checks this sentinel between batches. No PID is killed.
                (directory / "cancel.requested").touch(mode=0o600, exist_ok=True)
                with self.store.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
                    current = json.loads(row[0])
                    if current["status"] == "queued":
                        current.update(status="cancelled", finished_at=time.time())
                        db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(current), job_id))
                self.store.audit("job.cancel_requested", {"job_id": job_id, "mode": "cooperative"})
            return self.store.job(job_id)
        if action == "start":
            raise ValueError("remote worker owns its durable queue")
        if job["status"] == "submitting" and not job.get("submission_started") and action == "inspect":
            return job  # Interrupted preflight; explicit same-ID resubmit can resume it.
        endpoint = self._worker(job["worker"], job["token_env"])
        async with ExecutionClient(endpoint, os.environ.get(job["token_env"], "")) as client:
            if action == "resubmit":
                if job["status"] not in {"submitting", "submission_uncertain"}:
                    raise ValueError("resubmit only reconciles an uncertain submission using the SAME job ID")
                experiment = self.store.get(job["experiment_id"], "experiment")
                if experiment.get("deployment_id") or experiment.get("agent_deployments"):
                    job["serving_preflight"] = await self._serving_preflight(experiment)
                job.update(status="submission_uncertain", submission_started=True)
                self.store.save_job(job_id, job)
                remote = await client.submit(self._archive(job["experiment_id"]), job["remote_job_id"],
                                             digest=experiment["package"]["digest"])
            elif action == "cancel":
                remote = await client.stop(job["remote_job_id"])
            else:
                remote = await client.status(job["remote_job_id"])
            expected = self.store.get(job["experiment_id"], "experiment")["package"]["digest"]
            if remote.get("digest") != expected:
                raise ValueError("remote job input package identity changed")
            job.update(status=remote["status"], remote=remote)
            if action == "fetch":
                if job["status"] not in TERMINAL:
                    raise ValueError("fetch requires a terminal job")
                from mesocosm.reproduction.retrieval import fetch_results

                await fetch_results(self.store, job, client, self._verified_results)
        self.store.save_job(job_id, job)
        return self.store.job(job_id)

    def _verified_results(self, job: dict[str, Any]) -> pathlib.Path:
        """Recheck the complete fetched result inventory against the retained receipt."""
        from mesocosm.execution.package import MAX_INDEX_BYTES

        if not isinstance(job.get("retrieval"), dict) or not job.get("results_path"):
            raise ValueError("results lack a durable retrieval receipt; fetch the owned job to recover")
        root = self.store.directory("jobs", job["job_id"]) / "results"
        if root.is_symlink() or root.resolve() != pathlib.Path(job["results_path"]).resolve():
            raise ValueError("result directory differs from its owned job path")
        metadata_path = root / "package.json"
        if metadata_path.is_symlink():
            raise ValueError("result package metadata cannot be a symlink")
        with metadata_path.open("rb") as stream:
            content = stream.read(MAX_INDEX_BYTES + 1)
        if len(content) > MAX_INDEX_BYTES:
            raise ValueError("result package metadata exceeds size limit")
        metadata = json.loads(content)
        if not isinstance(metadata, dict) or not isinstance(metadata.get("files"), dict):
            raise ValueError("result package metadata is invalid")
        fingerprint = hashlib.sha256(json.dumps({key: value for key, value in metadata.items() if key != "digest"},
                                                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if metadata.get("digest") != fingerprint or fingerprint != job["retrieval"].get("digest"):
            raise ValueError("fetched result package identity mismatch")
        entries = metadata["files"]
        if {path.relative_to(root).as_posix() for path in root.rglob("*")} != {*entries, "package.json"}:
            raise ValueError("fetched result inventory changed")
        for relative, entry in entries.items():
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("fetched results contain unsafe paths")
            if entry["type"] == "file":
                if not path.is_file() or path.stat().st_size != entry["size"] or file_hash(path) != entry["sha256"]:
                    raise ValueError("fetched result bytes changed")
            elif entry["type"] != "directory" or not path.is_dir():
                raise ValueError("fetched result inventory type mismatch")
        original = json.loads((root / "job.json").read_text())
        experiment = self.store.get(job["experiment_id"], "experiment")
        if (original["job_id"] != job["remote_job_id"]
                or original["digest"] != experiment["package"]["digest"]):
            raise ValueError("fetched job does not bind this candidate input package")
        return root

    def _snapshot_parent(self, experiment_id: str, parent_bundle: str) -> str:
        parent = pathlib.Path(parent_bundle).resolve(strict=True)
        with self.store.connection() as db:
            jobs = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
        for job in jobs:
            if (job.get("kind") == "world" and job.get("experiment_id") == experiment_id
                    and job.get("results_path") and job.get("retrieval")):
                root = self._verified_results(job)
                if parent.parent == (root / "runs").resolve():
                    return job["job_id"]
        raise ValueError("snapshot parent must be a verified fetched run of this candidate")

    def snapshot_experiment(self, experiment_id: str, source_snapshot: str, parent_bundle: str) -> dict[str, Any]:
        """Freeze and verify an existing closed-run filesystem snapshot; exact live rewind is unsupported."""
        from mesocosm.reproduction.snapshots import freeze_snapshot

        experiment = self.store.get(experiment_id, "experiment")
        parent_job_id = self._snapshot_parent(experiment_id, parent_bundle)
        with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
            output = pathlib.Path(temporary) / "snapshot"
            result = freeze_snapshot(pathlib.Path(source_snapshot), output, parent_bundle=pathlib.Path(parent_bundle))
            result = {key: item for key, item in result.items() if key != "path"}
            value = {"experiment_id": experiment_id, "receipt": result, "parent_job_id": parent_job_id,
                     "behavior_id": experiment["behavior_id"]}
            identifier = self.store.put("snapshot", value)
            destination = self.store.directory("snapshots", identifier)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.move(str(output), destination)
        self.store.audit("snapshot.frozen", {"snapshot_id": identifier})
        return {"snapshot_id": identifier, "path": str(destination), **result}

    def fork_experiment(self, snapshot_id: str, changes: list[dict[str, Any]]) -> dict[str, Any]:
        """Restore declared files into a new isolated fork; return a local fork recipe, not a completed run."""
        from mesocosm.reproduction.snapshots import prepare_fork_recipe, verify_fork_recipe

        snapshot = self.store.get(snapshot_id, "snapshot")
        parent = self.store.get(snapshot["experiment_id"], "experiment")
        if (parent.get("deployment_id") or parent.get("agent_deployments")) and any(
                change.get("factor", "").startswith(("model.", "agent.model"))
                or change.get("factor") == "models" for change in changes):
            raise ValueError("model factors cannot override a bound deployment")
        configuration = factors.configure_experiment(parent["configuration"]["manifest"], changes,
                                                     parent["configuration"]["roles"])
        value = {**parent, "configuration": configuration, "parent_experiment_id": snapshot["experiment_id"],
                 "snapshot_id": snapshot_id, "state_mode": "partial_filesystem_fork"}
        identifier = self.store.put("experiment", value)
        target = self.store.directory("experiments", identifier)
        recipe_id_path = target / "recipe-id.json"
        if not target.exists():
            target.mkdir(parents=True, mode=0o700)
            receipt = prepare_fork_recipe(
                self.store.directory("snapshots", snapshot_id), target / "recipe",
                expected_index_sha256=snapshot["receipt"]["index_sha256"],
                world_archive=self._archive(snapshot["experiment_id"]),
                expected_archive_sha256=parent["archive_sha256"], changes=changes,
                roles=parent["configuration"]["roles"])
            recipe_id = self.store.put("fork_recipe", receipt)
            write_json(recipe_id_path, {"recipe_id": recipe_id})
        if not recipe_id_path.is_file():
            raise ValueError("fork preparation was interrupted; no executable recipe was admitted")
        reference = json.loads(recipe_id_path.read_text())
        frozen = self.store.get(reference["recipe_id"], "fork_recipe")
        receipt = verify_fork_recipe(target / "recipe", expected_receipt_sha256=frozen["receipt_sha256"])
        self.store.audit("experiment.fork_prepared", {"experiment_id": identifier, "snapshot_id": snapshot_id})
        return {"experiment_id": identifier, "restoration": receipt, "execution_status": "not_started",
                "remote_execution": "unsupported_for_partial_state", "configuration": configuration}

    def evaluate_candidate(self, experiment_id: str, transcript_id: str) -> dict[str, Any]:
        """Apply the candidate's predeclared rubric; test-set judgments require an operator-only gate."""
        experiment = self.store.get(experiment_id, "experiment")
        if experiment["split"] == "test" and not self.allow_test_evaluation:
            raise ValueError("held-out test evaluation is reserved for the independent operator")
        definition = self.store.get(experiment["behavior_id"], "behavior")
        transcript = self.store.get(transcript_id, "transcript")
        self._transcript_sources(transcript_id)
        if _held_out(transcript) and not self.allow_test_evaluation:
            raise ValueError("declared held-out transcript evaluation requires the independent operator")
        result = behavior.evaluate_candidate(definition, transcript)
        # The evidence content stays unchanged; candidate assignment is an explicit operator annotation.
        result["candidate_id"] = experiment_id
        result["assignment"] = {"experiment_id": experiment_id, "transcript_id": transcript_id,
                                "binding": "operator_supplied; not independently proven runtime lineage"}
        result["evaluation_sha256"] = digest({k: v for k, v in result.items() if k != "evaluation_sha256"})
        identifier = self.store.put("evaluation", result)
        self.store.audit("candidate.evaluated", {"evaluation_id": identifier, "experiment_id": experiment_id})
        return {"evaluation_id": identifier, "evaluation": result}

    def _run_binding(self, job_id: str, run_id: str) -> tuple[pathlib.Path, dict[str, Any]]:
        from mesocosm.core.manifest import load_manifest
        from mesocosm.reproduction.deployment import agent_bindings
        from mesocosm.reproduction.lineage import verify_run

        if not re.fullmatch(r"run_[A-Za-z0-9_-]+", run_id):
            raise ValueError("invalid runtime run_id")
        job = self.store.job(job_id)
        if job.get("kind") != "world" or job.get("status") != "completed":
            raise ValueError("only completed native jobs can receive a lineage-qualified evaluation")
        experiment = self.store.get(job["experiment_id"], "experiment")
        if experiment["split"] == "test" and not self.allow_test_evaluation:
            raise ValueError("held-out test evaluation is reserved for the independent operator")
        bindings = agent_bindings(experiment)
        preflight = job.get("serving_preflight", {})
        # Existing single-deployment jobs retain their original receipt shape.
        if experiment.get("deployment_id") and preflight.get("deployment_id") == experiment["deployment_id"]:
            preflight = {experiment["deployment_id"]: preflight}
        if not bindings or set(preflight) != set(bindings.values()) or any(
                preflight[identifier].get("deployment_id") != identifier for identifier in bindings.values()):
            raise ValueError("candidate/job lacks its verified deployment preflight")
        deployments = {identifier: self.store.get(identifier, "deployment") for identifier in set(bindings.values())}
        root = self._verified_results(job)
        original_job = json.loads((root / "job.json").read_text())
        if original_job.get("status") != "completed" or original_job.get("cleanup", {}).get("status") != "complete":
            raise ValueError("original worker did not confirm completion and cleanup")
        original_result = json.loads((root / "result.json").read_text())
        if (original_result.get("run_id") != run_id or original_result.get("status") != "ok"
                or original_result.get("run_status") != "ok"):
            raise ValueError("worker result does not identify this completed native run")
        run = root / "runs" / run_id
        if not run.is_dir() or run.is_symlink():
            raise ValueError("run is absent from this candidate's fetched job")
        with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
            inputs = pathlib.Path(temporary) / "inputs"
            package = unpack_submission(self._archive(job["experiment_id"]), inputs)
            path = inputs / package["entrypoint"]
            binding = verify_run(run, load_manifest(path).model_dump(mode="json"), package["digest"],
                                 job["remote_job_id"], experiment.get("deployment_id"),
                                 deployments, path, agent_deployments=bindings)
        binding.update(job_id=job_id, remote_job_id=job["remote_job_id"], experiment_id=job["experiment_id"],
                       input_package_digest=experiment["package"]["digest"],
                       result_package_digest=job["retrieval"]["digest"],
                       serving_preflight_sha256=digest(job["serving_preflight"]))
        return run, binding

    def evaluate_run(self, job_id: str, run_id: str) -> dict[str, Any]:
        """Evaluate original fetched evidence after strict candidate, model, call and verifier reconciliation."""
        run, binding = self._run_binding(job_id, run_id)
        imported = self.inspect_transcript(str(run))
        experiment = self.store.get(binding["experiment_id"], "experiment")
        result = behavior.evaluate_candidate(self.store.get(experiment["behavior_id"], "behavior"),
                                             imported["transcript"])
        binding["transcript_id"] = imported["transcript_id"]
        binding_id = self.store.put("runtime_binding", binding)
        result.update(candidate_id=binding["experiment_id"], assignment={**binding, "runtime_binding_id": binding_id})
        result.update(incident_id=binding["task_set_sha256"], costs=binding["costs"],
                      clustering_basis="frozen_task_set; runs are not independent task samples")
        result["evaluation_sha256"] = digest({key: value for key, value in result.items()
                                              if key != "evaluation_sha256"})
        identifier = self.store.put("evaluation", result)
        self.store.audit("run.evaluated", {"evaluation_id": identifier, "runtime_binding_id": binding_id})
        return {"evaluation_id": identifier, "evaluation": result}

    def evaluate_collective(self, behavior_id: str, transcript_id: str | None = None,
                            job_id: str | None = None, run_id: str | None = None,
                            evaluation_ids: list[str] | None = None) -> dict[str, Any]:
        """Measure frozen collective signals in a qualified run or imported evidence; aggregate without CIs."""
        definition = self.store.get(behavior_id, "behavior")
        collective.validate_definition(definition)
        modes = int(transcript_id is not None) + int(job_id is not None or run_id is not None)
        modes += int(evaluation_ids is not None)
        if modes != 1 or (job_id is None) != (run_id is None):
            raise ValueError("choose transcript_id, job_id plus run_id, or evaluation_ids")
        if evaluation_ids is not None:
            if not 1 <= len(evaluation_ids) <= 1000:
                raise ValueError("aggregate 1..1000 collective evaluation IDs")
            rows = [self._collective_evaluation(identifier) for identifier in evaluation_ids]
            if any(row["behavior_id"] != behavior_id for row in rows):
                raise ValueError("collective evaluations must use the same frozen behavior")
            result = collective.aggregate(rows)
            result.update(behavior_id=behavior_id, evaluation_ids=evaluation_ids,
                          operator_only=any(row.get("operator_only", False) for row in rows))
            identifier = self.store.put("collective_comparison", result)
            self.store.audit("collective.compared", {"comparison_id": identifier})
            return {"comparison_id": identifier, "comparison": result}
        binding = None
        if job_id is not None and run_id is not None:
            run, binding = self._run_binding(job_id, run_id)
            if self.store.get(binding["experiment_id"], "experiment")["behavior_id"] != behavior_id:
                raise ValueError("qualified collective evaluation must use the candidate's frozen behavior")
            transcript_id = self.inspect_transcript(str(run))["transcript_id"]
        assert transcript_id is not None
        transcript = self.store.get(transcript_id, "transcript")
        self._transcript_sources(transcript_id)
        if _held_out(transcript) and not self.allow_test_evaluation:
            raise ValueError("held-out collective evidence requires the independent operator")
        result = collective.evaluate(definition, transcript)
        result.update(behavior_id=behavior_id, association="verified_runtime_lineage" if binding else "unverified",
                      assignment={"transcript_id": transcript_id, **({"binding": binding,
                                  "experiment_id": binding["experiment_id"]} if binding else {})},
                      operator_only=_held_out(transcript) or bool(binding and self.store.get(
                          binding["experiment_id"], "experiment")["split"] == "test"))
        identifier = self.store.put("collective_evaluation", result)
        self.store.audit("collective.evaluated", {"evaluation_id": identifier, "behavior_id": behavior_id})
        return {"evaluation_id": identifier, "evaluation": result}

    def _collective_evaluation(self, identifier: str) -> dict[str, Any]:
        result = self.store.get(identifier, "collective_evaluation")
        transcript_id = result["assignment"]["transcript_id"]
        transcript = self.store.get(transcript_id, "transcript")
        self._transcript_sources(transcript_id)
        if _held_out(transcript) and not self.allow_test_evaluation:
            raise ValueError("held-out collective evidence requires the independent operator")
        binding = result["assignment"].get("binding")
        if binding and self._run_binding(binding["job_id"], binding["run_id"])[1] != binding:
            raise ValueError("collective runtime binding changed")
        return result

    def _evaluation(self, identifier: str) -> dict[str, Any]:
        value = self.store.get(identifier, "evaluation")
        experiment = self.store.get(value["candidate_id"], "experiment")
        transcript = self.store.get(value["assignment"]["transcript_id"], "transcript")
        self._transcript_sources(value["assignment"]["transcript_id"])
        if value["assignment"].get("runtime_binding_id"):
            binding = self.store.get(value["assignment"]["runtime_binding_id"], "runtime_binding")
            _, current = self._run_binding(binding["job_id"], binding["run_id"])
            if current != {key: item for key, item in binding.items() if key != "transcript_id"}:
                raise ValueError("runtime binding changed since evaluation")
        if (experiment["split"] == "test" or _held_out(transcript)) and not self.allow_test_evaluation:
            raise ValueError("held-out evaluation access requires the independent operator")
        return value

    def compare_candidates(self, evaluation_ids: list[str]) -> dict[str, Any]:
        """Compare source-clustered outcomes with unknowns and failed attempts retained."""
        results = [self._evaluation(identifier) for identifier in evaluation_ids]
        result = behavior.compare_candidates(results)
        result["assignment_limits"] = sorted({item.get("assignment", {}).get("binding", "unknown")
                                              for item in results})
        result["prospective_runtime_binding"] = ("verified" if results and all(
            item["assignment"].get("runtime_binding_id") for item in results) else "unverified")
        result["causal_inference"] = "requires_prospective_comparison_design"
        identifier = self.store.put("comparison", result)
        return {"comparison_id": identifier, "comparison": result}

    def plan_factorial(self, baseline_experiment_id: str, base_deployment_id: str, trained_deployment_id: str,
                       environment_changes: list[dict[str, Any]], idempotency_key: str,
                       repetitions: int = 3, order_seed: int = 0) -> dict[str, Any]:
        """Freeze base/trained × baseline/designed cells, paired repetitions and unchanged task/model budgets."""
        from mesocosm.reproduction.factorial import plan

        return plan(self, baseline_experiment_id, base_deployment_id, trained_deployment_id,
                    environment_changes, repetitions, order_seed, idempotency_key)

    async def advance_factorial(self, study_id: str, worker: str,
                                token_env: str = "MESOCOSM_WORKER_TOKEN") -> dict[str, Any]:
        """Reconcile the current trial and submit at most one next trial; never retry based on outcomes."""
        from mesocosm.reproduction.factorial import advance

        return await advance(self, study_id, worker, token_env)

    def report_factorial(self, study_id: str) -> dict[str, Any]:
        """Report every planned trial, keeping absent, unknown, failed and unrun observations separate."""
        from mesocosm.reproduction.factorial import report

        return report(self, study_id)

    def export_reproducer(self, experiment_id: str, evaluation_ids: list[str] | None = None) -> dict[str, Any]:
        """Export portable original inputs plus rubric and evaluation evidence; no model weights are invented."""
        experiment = self.store.get(experiment_id, "experiment")
        if experiment["state_mode"] != "fresh_world":
            raise ValueError("portable partial-state fork execution is not implemented")
        evaluations = [self._evaluation(identifier) for identifier in evaluation_ids or []]
        if any(item.get("candidate_id") != experiment_id for item in evaluations):
            raise ValueError("export evaluations must belong to this candidate")
        archive = self._archive(experiment_id)
        receipt = {"schema_version": 1, "experiment_id": experiment_id, "archive_sha256": file_hash(archive),
                   "package_digest": experiment["package"]["digest"], "evaluation_ids": evaluation_ids or [],
                   "reproduction_mode": "new_model_calls_and_original_evaluators; not deterministic replay",
                   "required_external_resources": ["compatible installed Mesocosm integrations", "Linux Docker worker",
                                                   "pinned model serving and credentials supplied by the operator"],
                   "native_acceptance": "unverified"}
        with self.store.connection() as db:
            jobs = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
        related_jobs = [job for job in jobs if job.get("experiment_id") == experiment_id]
        # Freeze a complete directory before admitting its ID; repeat calls verify
        # the stored inventory rather than trusting existence of a partial export.
        with tempfile.TemporaryDirectory(dir=self.store.root) as temporary:
            staged = pathlib.Path(temporary) / "export"
            staged.mkdir(mode=0o700)
            shutil.copyfile(archive, staged / "world.zip")
            write_json(staged / "receipt.json", receipt)
            write_json(staged / "behavior.json", self.store.get(experiment["behavior_id"], "behavior"))
            if experiment.get("deployment_id"):
                write_json(staged / "deployment.json", self.store.get(experiment["deployment_id"], "deployment"))
            if experiment.get("agent_deployments"):
                write_json(staged / "deployments.json", {identifier: self.store.get(identifier, "deployment")
                           for identifier in sorted(set(experiment["agent_deployments"].values()))})
            for evaluation in evaluations:
                assignment = evaluation["assignment"]
                if assignment.get("runtime_binding_id"):
                    source = self._verified_results(self.store.job(assignment["job_id"]))
                    destination = staged / "runtime-results" / assignment["job_id"]
                    if not destination.exists():
                        shutil.copytree(source, destination)
            write_json(staged / "experiment.json", {
                "experiment_id": experiment_id, "behavior_id": experiment["behavior_id"], "split": experiment["split"],
                "configuration_sha256": experiment["configuration"]["configuration_sha256"],
                "manifest_sha256": experiment["configuration"]["manifest_sha256"],
                "roles": experiment["configuration"]["roles"], "package": experiment["package"],
                "source_manifest_sha256": experiment["source_manifest_sha256"],
                "agent_deployments": experiment.get("agent_deployments"),
                "configuration_source": "world.zip; operator connection credentials are excluded"})
            write_json(staged / "evaluations.json", evaluations)
            write_json(staged / "factor-changes.json", experiment["configuration"]["effective_changes"])
            write_json(staged / "attempts.json", related_jobs)
            # Preserve original source files, not just normalized event projections.
            transcripts = sorted({item["assignment"]["transcript_id"] for item in evaluations})
            write_json(staged / "evidence.json", {key: self.store.get(key, "transcript") for key in transcripts})
            for transcript_id in transcripts:
                shutil.copytree(self._transcript_sources(transcript_id), staged / "sources" / transcript_id)
            readme = ("# Mesocosm reproducer\n\nPrivate research artifact: review before sharing.\n"
                      "Run from this directory with a compatible Mesocosm installation. Supply an operator-approved\n"
                      "worker and credential through MESOCOSM_WORKER_URL and MESOCOSM_WORKER_TOKEN.\n"
                      "`python rerun.py` first persists a fresh submission identity, then submits world.zip.\n"
                      "Repeated calls use that same identity; an uncertain network result never creates another job.\n"
                      "For another independent run, copy the export to a new directory without submission.json.\n"
                      "Models/serving are external; no deterministic replay or successful behavior is claimed.\n")
            (staged / "README.md").write_text(readme)
            (staged / "rerun.py").write_text(_RERUN_SCRIPT)
            inventory = {path.relative_to(staged).as_posix(): file_hash(path)
                         for path in sorted(staged.rglob("*")) if path.is_file()}
            write_json(staged / "files.json", inventory)
            frozen = {**receipt, "files": inventory, "files_index_sha256": file_hash(staged / "files.json")}
            identifier = self.store.put("export", frozen)
            destination = self.store.directory("exports", identifier)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                staged.rename(destination)
        if destination.is_symlink() or (destination / "files.json").is_symlink():
            raise ValueError("export paths cannot be symlinks")
        if file_hash(destination / "files.json") != frozen["files_index_sha256"]:
            raise ValueError("export inventory changed")
        actual = {}
        for path in destination.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("export contains unsafe paths")
            if path.is_file() and path != destination / "files.json":
                actual[path.relative_to(destination).as_posix()] = file_hash(path)
        if actual != frozen["files"]:
            raise ValueError("export bytes are missing, changed or unexpected")
        self.store.audit("reproducer.exported", {"export_id": identifier, "experiment_id": experiment_id})
        return {"export_id": identifier, "path": str(destination), **receipt}


_RERUN_SCRIPT = r'''"""Submit a frozen world with a persisted idempotent identity; no resource provisioning."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import uuid
from mesocosm.execution.client import ExecutionClient, worker_url

root = Path(__file__).resolve().parent
receipt = json.loads((root / "receipt.json").read_text())
archive = root / "world.zip"
if hashlib.sha256(archive.read_bytes()).hexdigest() != receipt["archive_sha256"]:
    raise ValueError("world.zip content changed")
worker = worker_url(os.environ["MESOCOSM_WORKER_URL"])
identity = root / "submission.json"
if not identity.exists():
    descriptor = os.open(identity, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"worker": worker, "job_id": uuid.uuid4().hex,
                   "package_digest": receipt["package_digest"]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
submission = json.loads(identity.read_text())
if submission["worker"] != worker or submission["package_digest"] != receipt["package_digest"]:
    raise ValueError("persisted submission identity conflicts with this worker/package")
async def main():
    async with ExecutionClient(worker, os.environ["MESOCOSM_WORKER_TOKEN"]) as client:
        result = await client.submit(archive, submission["job_id"], digest=submission["package_digest"])
        print(json.dumps(result))
asyncio.run(main())
'''
