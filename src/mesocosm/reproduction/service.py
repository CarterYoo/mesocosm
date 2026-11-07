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
from typing import Any

import yaml

from mesocosm.core.config import absolutize_manifest
from mesocosm.execution.client import ExecutionClient, worker_url
from mesocosm.execution.package import pack_submission, unpack_submission
from mesocosm.execution.protocol import TERMINAL
from mesocosm.reproduction import behavior, evidence, factors
from mesocosm.reproduction.store import Store, digest, file_hash, write_json

TRANSCRIPT_MAX_FILE_BYTES = 256 * 1024 * 1024
TRANSCRIPT_MAX_BYTES = 512 * 1024 * 1024
TRANSCRIPT_MAX_FILES = 10_000

TOOLS = (
    "inspect_transcript", "define_behavior", "list_factors", "configure_experiment", "validate_experiment",
    "prepare_dataset", "submit_training", "run_experiment", "get_job_status", "snapshot_experiment",
    "fork_experiment", "evaluate_candidate", "compare_candidates", "export_reproducer",
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
                 allowed_workers: dict[str, str] | None = None):
        self.store = Store(root)
        self.allow_test_evaluation = allow_test_evaluation
        self.allowed_workers = {worker_url(url): variable for url, variable in (allowed_workers or {}).items()}
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", value)
               for value in self.allowed_workers.values()):
            raise ValueError("worker credentials must name operator-approved environment variables")

    def _worker(self, endpoint: str, token_env: str) -> str:
        endpoint = worker_url(endpoint)
        if self.allowed_workers.get(endpoint) != token_env:
            raise ValueError("worker endpoint and credential variable are not operator-approved")
        return endpoint

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool not in TOOLS:
            raise ValueError("unknown research tool")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        function = getattr(self, tool)
        inspect.signature(function).bind(**arguments)
        if inspect.iscoroutinefunction(function):
            return await function(**arguments)
        # Snapshot helpers and synchronous imports must not nest asyncio.run in
        # this dispatcher event loop or block another caller's status request.
        return await asyncio.to_thread(function, **arguments)

    @classmethod
    def tools(cls) -> dict[str, Any]:
        return {"schema_version": 1, "tools": [
            {"name": name, "description": inspect.getdoc(getattr(cls, name)),
             "parameters": [{"name": key, "required": value.default is inspect.Parameter.empty,
                             "type": str(value.annotation),
                             **({"default": value.default} if value.default is not inspect.Parameter.empty else {})}
                            for key, value in inspect.signature(getattr(cls, name)).parameters.items()
                            if key != "self"]}
            for name in TOOLS], "trust_boundary": "single trusted operator; not a multi-tenant sandbox"}

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
        """Freeze a versioned stage rubric before configuring candidates."""
        value = behavior.define_behavior(spec)
        identifier = self.store.put("behavior", value)
        self.store.audit("behavior.defined", {"behavior_id": identifier})
        return {"behavior_id": identifier, "definition": value}

    def list_factors(self, experiment_id: str | None = None) -> dict[str, Any]:
        """List enforced fields, types, defaults, scopes and unsupported live changes."""
        world = self.store.get(experiment_id, "experiment")["configuration"]["manifest"] if experiment_id else None
        return {"factors": factors.list_factors(world)}

    def configure_experiment(self, base_world: str, behavior_id: str, changes: list[dict[str, Any]],
                             roles: dict[str, list[str]] | None = None,
                             split: str = "development") -> dict[str, Any]:
        """Freeze a rubric, scoped factors and all portable world inputs into one candidate."""
        self.store.get(behavior_id, "behavior")
        if split not in {"train", "development", "test"}:
            raise ValueError("split must be train, development or test")
        source = pathlib.Path(base_world).resolve(strict=True)
        original = source.read_bytes()
        raw = yaml.safe_load(original)
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
        experiment = self.store.get(experiment_id, "experiment")
        if experiment["state_mode"] != "fresh_world":
            raise ValueError("partial-state forks require the local fork executor; remote fresh runs would lose state")
        archive = self._archive(experiment_id)
        endpoint = self._worker(worker, token_env)
        args = {"experiment_id": experiment_id, "worker": endpoint, "token_env": token_env}
        cached = self.store.begin(idempotency_key, "run_experiment", args)
        if cached is not None:
            return cached
        identifier = "job-" + digest({"key": idempotency_key, "tool": "run_experiment", "args": args})
        record: dict[str, Any] = {"kind": "world", "status": "submitting", "experiment_id": experiment_id,
                  "worker": endpoint, "token_env": token_env, "remote_job_id": identifier[-32:],
                  "created_at": time.time()}
        self.store.save_job(identifier, record, new=True)
        try:
            async with ExecutionClient(endpoint, os.environ.get(token_env, "")) as client:
                remote = await client.submit(archive, record["remote_job_id"], digest=experiment["package"]["digest"])
            record.update(status=remote["status"], remote=remote)
        except Exception as exc:
            # An HTTP failure may occur after acceptance. Keep the same remote identity.
            record.update(status="submission_uncertain", error_type=type(exc).__name__)
        self.store.save_job(identifier, record)
        result = {"job_id": identifier, "status": record["status"], "remote_job_id": record["remote_job_id"]}
        self.store.finish(idempotency_key, result)
        return result

    async def get_job_status(self, job_id: str, action: str = "inspect") -> dict[str, Any]:
        """Inspect, fetch, cancel, start a queued trainer or reconcile an uncertain submission."""
        if action not in {"inspect", "fetch", "cancel", "start", "resubmit"}:
            raise ValueError("unsupported job action")
        job = self.store.job(job_id)
        directory = self.store.directory("jobs", job_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        endpoint = self._worker(job["worker"], job["token_env"])
        async with ExecutionClient(endpoint, os.environ.get(job["token_env"], "")) as client:
            if action == "resubmit":
                if job["status"] != "submission_uncertain":
                    raise ValueError("resubmit only reconciles an uncertain submission using the SAME job ID")
                experiment = self.store.get(job["experiment_id"], "experiment")
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
                destination = directory / "results"
                if not destination.exists():
                    job["retrieval"] = await client.fetch(job["remote_job_id"], destination)
                job["results_path"] = str(destination)
                self._verified_results(job)
        self.store.save_job(job_id, job)
        return self.store.job(job_id)

    def _verified_results(self, job: dict[str, Any]) -> pathlib.Path:
        """Recheck the complete fetched result inventory against the retained receipt."""
        root = self.store.directory("jobs", job["job_id"]) / "results"
        if root.is_symlink() or root.resolve() != pathlib.Path(job["results_path"]).resolve():
            raise ValueError("result directory differs from its owned job path")
        metadata_path = root / "package.json"
        if metadata_path.is_symlink():
            raise ValueError("result package metadata cannot be a symlink")
        metadata = json.loads(metadata_path.read_text())
        fingerprint = hashlib.sha256(json.dumps({key: value for key, value in metadata.items() if key != "digest"},
                                                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if metadata.get("digest") != fingerprint or fingerprint != job["retrieval"]["digest"]:
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

    def _evaluation(self, identifier: str) -> dict[str, Any]:
        value = self.store.get(identifier, "evaluation")
        experiment = self.store.get(value["candidate_id"], "experiment")
        transcript = self.store.get(value["assignment"]["transcript_id"], "transcript")
        self._transcript_sources(value["assignment"]["transcript_id"])
        if (experiment["split"] == "test" or _held_out(transcript)) and not self.allow_test_evaluation:
            raise ValueError("held-out evaluation access requires the independent operator")
        return value

    def compare_candidates(self, evaluation_ids: list[str]) -> dict[str, Any]:
        """Compare source-clustered outcomes with unknowns and failed attempts retained."""
        results = [self._evaluation(identifier) for identifier in evaluation_ids]
        result = behavior.compare_candidates(results)
        result["assignment_limits"] = sorted({item.get("assignment", {}).get("binding", "unknown")
                                              for item in results})
        result["prospective_runtime_binding"] = "unverified"
        result["causal_inference"] = "unsupported_from_imported_evidence_alone"
        identifier = self.store.put("comparison", result)
        return {"comparison_id": identifier, "comparison": result}

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
            write_json(staged / "experiment.json", {
                "experiment_id": experiment_id, "behavior_id": experiment["behavior_id"], "split": experiment["split"],
                "configuration_sha256": experiment["configuration"]["configuration_sha256"],
                "manifest_sha256": experiment["configuration"]["manifest_sha256"],
                "roles": experiment["configuration"]["roles"], "package": experiment["package"],
                "source_manifest_sha256": experiment["source_manifest_sha256"],
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
