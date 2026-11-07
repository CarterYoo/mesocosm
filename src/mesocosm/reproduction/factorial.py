"""Prospective four-cell studies with durable trials and no outcome-selected retries."""

from __future__ import annotations

import copy
import fcntl
import os
import pathlib
import random
import tempfile
from collections import Counter
from typing import TYPE_CHECKING, Any

from mesocosm.core.manifest import load_manifest
from mesocosm.execution.package import unpack_submission
from mesocosm.execution.protocol import TERMINAL
from mesocosm.reproduction.store import digest

if TYPE_CHECKING:
    from mesocosm.reproduction.service import ResearchWorkbench

CELLS = ("base_baseline", "trained_baseline", "base_designed", "trained_designed")
PROTECTED = ("model", "models", "agent.model", "agent.budget", "resources.model_", "resources.completion_tokens",
             "tasks", "task_assignment", "mode", "observe", "agent.persona", "agent.skills")


def plan(bench: ResearchWorkbench, baseline_experiment_id: str, base_deployment_id: str,
         trained_deployment_id: str, environment_changes: list[dict[str, Any]], repetitions: int,
         order_seed: int, idempotency_key: str) -> dict[str, Any]:
    if type(repetitions) is not int or not 1 <= repetitions <= 100 or type(order_seed) is not int:
        raise ValueError("factorial studies require 1..100 repetitions and an integer scheduling seed")
    if not environment_changes or any(change.get("factor", "").startswith(PROTECTED) for change in environment_changes):
        raise ValueError("environment changes must preserve tasks, policies, prompts, sampling, "
                         "recording and model budgets")
    baseline = bench.store.get(baseline_experiment_id, "experiment")
    if baseline["state_mode"] != "fresh_world":
        raise ValueError("factorial trials require fresh worlds")
    if baseline["split"] == "test" and not bench.allow_test_evaluation:
        raise ValueError("held-out factorial planning requires the independent operator")
    deployments = {"base": base_deployment_id, "trained": trained_deployment_id}
    base, trained = [bench.store.get(identifier, "deployment") for identifier in deployments.values()]
    fields = ("base_model", "base_revision", "base_sha256", "dtype")
    if (any(base["policy"][key] != trained["policy"][key] for key in fields)
            or base["policy"]["adapter_sha256"] is not None or not trained["training_artifact"]
            or not trained["policy"]["adapter_sha256"] or base["policy"]["model"] == trained["policy"]["model"]
            or base["server"] != trained["server"]):
        raise ValueError("factorial models require one base and its trained adapter on the same serving endpoint/dtype")
    args = {"baseline_experiment_id": baseline_experiment_id, "deployments": deployments,
            "environment_changes": environment_changes, "repetitions": repetitions, "order_seed": order_seed}
    cached = bench.store.begin(idempotency_key, "plan_factorial", args)
    if cached is not None:
        return cached
    try:
        with tempfile.TemporaryDirectory(dir=bench.store.root) as temporary:
            root = pathlib.Path(temporary) / "inputs"
            package = unpack_submission(bench._archive(baseline_experiment_id), root)
            path = root / package["entrypoint"]
            world = load_manifest(path)
            if world.judge.monitor.enabled and world.judge.monitor.model in {
                    base["policy"]["model"], trained["policy"]["model"]}:
                raise ValueError("factorial monitoring requires a separate fixed model alias outside the policy axis")
            if (world.execution.backend != "docker" or world.task_assignment.policy != "all"
                    or world.mode != "comparison" or world.memory != "reset"):
                raise ValueError("factorial studies currently require native comparison/all and reset memory")
            if any(agent.budget.requests is None or agent.budget.completion_tokens is None for agent in world.agents):
                raise ValueError("freeze finite per-agent request and completion-token budgets before planning")
            cells = {}
            for cell in CELLS:
                model, environment = cell.split("_")
                configured = bench.configure_experiment(str(path), baseline["behavior_id"],
                    environment_changes if environment == "designed" else [], baseline["configuration"]["roles"],
                    baseline["split"], deployments[model])
                identifier = configured["experiment_id"]
                report = bench.validate_experiment(identifier)
                if not report.get("valid"):
                    raise ValueError("factorial candidate failed validation")
                cells[cell] = identifier
            # Compare packaged forms, not source paths or operator assertions.
            manifests = {}
            input_files = {}
            for cell, identifier in cells.items():
                unpacked = pathlib.Path(temporary) / cell
                value = unpack_submission(bench._archive(identifier), unpacked)
                manifests[cell] = load_manifest(unpacked / value["entrypoint"]).model_dump(mode="json")
                input_files[cell] = {key: entry for key, entry in value["files"].items() if key.startswith("inputs/")}
            for environment in ("baseline", "designed"):
                left, right = (copy.deepcopy(manifests[f"{model}_{environment}"]) for model in ("base", "trained"))
                for data in (left, right):
                    for key in ("id", "hf_repo", "revision", "upstream", "api_key_env", "serving_identity"):
                        data["model"].pop(key, None)
                    for agent in data["agents"]:
                        agent["model"] = "POLICY_AXIS"
                if left != right or input_files[f"base_{environment}"] != input_files[f"trained_{environment}"]:
                    raise ValueError("model cells differ outside the declared policy axis")
            if manifests["base_baseline"] == manifests["base_designed"]:
                raise ValueError("environment changes have no effective difference from baseline")
            # World tasks/verifiers must have identical bytes in both environments.
            for task in manifests["base_baseline"]["tasks"]:
                prefix = task["path"].rstrip("/") + "/"
                trees = [{key: value for key, value in input_files[cell].items() if key.startswith(prefix)}
                         for cell in CELLS]
                if not trees[0] or any(tree != trees[0] for tree in trees):
                    raise ValueError("task or original verifier bytes changed across factorial cells")
            # Dataset group IDs detect declared overlap, not hidden semantic duplication.
            from mesocosm.reproduction.dataset import load_dataset
            artifact = trained["training_artifact"]
            _, examples = load_dataset(bench.store.directory("datasets", artifact["dataset_id"]))
            trained_tasks = {row.get("task_id") for row in examples
                             if row.get("split") == "train" and row.get("task_id")}
            evaluation_tasks = {task.name for task in world.tasks}
            if trained_tasks & evaluation_tasks:
                raise ValueError("declared training/evaluation task identities overlap")
            trials: list[dict[str, Any]] = []
            rng = random.Random(order_seed)
            for repetition in range(repetitions):
                order = list(CELLS)
                rng.shuffle(order)
                trials.extend({"cell": cell, "repetition": repetition, "experiment_id": cells[cell]} for cell in order)
            value = {"schema_version": 1, **args, "cells": cells, "trials": trials,
                     "behavior_id": baseline["behavior_id"], "split": baseline["split"],
                     "request_budget_upper_bound": 4 * repetitions * sum(a.budget.requests or 0 for a in world.agents),
                     "completion_token_budget_upper_bound": 4 * repetitions * sum(
                         a.budget.completion_tokens or 0 for a in world.agents),
                     "control_plane_probe_cost": "one token per deployment verification; separate from agent budgets",
                     "disjointness": "declared task IDs checked; semantic/content disjointness unverified",
                     "order_seed_scope": "scheduling_only; no deterministic replay guarantee",
                     "failed_trials": "retained_without_replacement", "causal_generalization": "unverified"}
            identifier = bench.store.put("factorial", value)
        result = {"study_id": identifier, "study": value}
        bench.store.finish(idempotency_key, result)
        return result
    except Exception as exc:
        bench.store.finish(idempotency_key, None, type(exc).__name__)
        raise


def state_id(study_id: str) -> str:
    return "job-" + study_id.split("-", 1)[1]


async def advance(bench: ResearchWorkbench, study_id: str, worker: str, token_env: str) -> dict[str, Any]:
    study = bench.store.get(study_id, "factorial")
    if study["split"] == "test" and not bench.allow_test_evaluation:
        raise ValueError("held-out factorial execution requires the independent operator")
    endpoint = bench._worker(worker, token_env)
    identifier = state_id(study_id)
    directory = bench.store.directory("jobs", identifier)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / "advance.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another caller is advancing this study") from None
        for experiment_id in study["cells"].values():
            bench._archive(experiment_id)
        try:
            state = bench.store.job(identifier)
        except KeyError:
            state = {"kind": "factorial", "status": "running", "study_id": study_id,
                     "worker": endpoint, "token_env": token_env,
                     "trials": [{**trial, "status": "not_run"} for trial in study["trials"]]}
            bench.store.save_job(identifier, state, new=True)
        if state["worker"] != endpoint or state["token_env"] != token_env:
            raise ValueError("study worker/credential cannot change after execution begins")
        if state["status"] in {"completed", "halted"}:
            return report(bench, study_id)
        for index, trial in enumerate(state["trials"]):
            if trial["status"] in {"evaluated", "failed", "quarantined"}:
                continue
            args = {"experiment_id": trial["experiment_id"], "worker": endpoint, "token_env": token_env}
            key = f"{study_id}:{index}"
            job_id = "job-" + digest({"key": key, "tool": "run_experiment", "args": args})
            trial["job_id"] = job_id
            bench.store.save_job(identifier, state)
            try:
                job = bench.store.job(job_id)
            except KeyError:
                result = await bench.run_experiment(**args, idempotency_key=key)
                trial.update(status="submitted", remote_status=result["status"])
                bench.store.save_job(identifier, state)
                return report(bench, study_id)
            if job["status"] == "preflight_failed":
                trial.update(status="failed", failure="serving_preflight_failed")
                state.update(status="halted", reason="serving_preflight_failed")
            else:
                # Uncertain acceptance is inspected at the SAME remote identity.
                # No blind resubmission; explicit get_job_status(resubmit) is available.
                job = await bench.get_job_status(job_id)
                trial.update(status="submitted", remote_status=job["status"])
                if job["status"] not in TERMINAL:
                    trial["status"] = job["status"]
                    bench.store.save_job(identifier, state)
                    return report(bench, study_id)
                job = await bench.get_job_status(job_id, "fetch")
                trial["results_path"] = job["results_path"]
                if job["status"] != "completed":
                    trial.update(status="failed", failure=job["status"])
                    if job["status"] == "failed_cleanup":
                        state.update(status="halted", reason="failed_cleanup")
                else:
                    try:
                        runs = list((pathlib.Path(job["results_path"]) / "runs").iterdir())
                        if len(runs) != 1:
                            raise ValueError("single-world trial must return exactly one native run")
                        evaluated = bench.evaluate_run(job_id, runs[0].name)
                        trial.update(status="evaluated", evaluation_id=evaluated["evaluation_id"])
                    except (ValueError, KeyError, OSError) as exc:
                        trial.update(status="quarantined", failure_type=type(exc).__name__)
                        state.update(status="halted", reason="runtime_lineage_rejected")
            bench.store.save_job(identifier, state)
            if state["status"] == "halted":
                return report(bench, study_id)
        state["status"] = "completed"
        bench.store.save_job(identifier, state)
        return report(bench, study_id)


def report(bench: ResearchWorkbench, study_id: str) -> dict[str, Any]:
    study = bench.store.get(study_id, "factorial")
    if study["split"] == "test" and not bench.allow_test_evaluation:
        raise ValueError("held-out factorial reporting requires the independent operator")
    try:
        state = bench.store.job(state_id(study_id))
    except KeyError:
        state = {"status": "not_started", "trials": [{**trial, "status": "not_run"} for trial in study["trials"]]}
    trials = copy.deepcopy(state["trials"])
    cells: dict[str, Any] = {}
    for cell in CELLS:
        selected = [trial for trial in trials if trial["cell"] == cell]
        counts: Counter[str] = Counter()
        for trial in selected:
            if trial.get("evaluation_id"):
                value = bench._evaluation(trial["evaluation_id"])
                trial["evaluation"] = value
                counts[value["status"]] += 1
            else:
                counts["unknown"] += 1
        n = len(selected)
        cells[cell] = {"planned": n, "trial_statuses": dict(Counter(trial["status"] for trial in selected)),
                       "behavior": {key: counts[key] for key in ("present", "absent", "unknown")},
                       "presence_bounds": [counts["present"] / n, (counts["present"] + counts["unknown"]) / n]}
    effects = {}
    for environment in ("baseline", "designed"):
        base = cells[f"base_{environment}"]["presence_bounds"]
        trained = cells[f"trained_{environment}"]["presence_bounds"]
        effects[f"trained_minus_base_{environment}"] = [trained[0] - base[1], trained[1] - base[0]]
    left, right = effects["trained_minus_base_baseline"], effects["trained_minus_base_designed"]
    effects["interaction"] = [right[0] - left[1], right[1] - left[0]]
    return {"study_id": study_id, "status": state["status"], "halt_reason": state.get("reason"),
            "cells": cells, "trials": trials, "denominator": len(study["trials"]),
            "presence_difference_bounds": effects, "bounds_are_confidence_intervals": False,
            "interpretation": "descriptive within this frozen task set; repeated runs are not independent tasks",
            "behavioral_improvement_demonstrated": False, "numerical_weight_verification": "unverified"}
