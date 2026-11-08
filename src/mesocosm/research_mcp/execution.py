"""Bounded development trials over the real execution worker, with private lineage."""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from mesocosm.core.evaluation import input_snapshot
from mesocosm.core.manifest import load_manifest
from mesocosm.execution.client import worker_url
from mesocosm.execution.package import unpack_submission
from mesocosm.execution.protocol import TERMINAL
from mesocosm.integrations.tasks import load_task
from mesocosm.reproduction.deployment import agent_bindings
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import Store, digest

_REQUEST = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_DONE = {"evaluated", "failed", "quarantined", "not_run_cancelled"}
Action = Literal["inspect", "advance", "fetch", "cancel", "resubmit"]


def _fail(code: str, message: str) -> NoReturn:
    raise ToolInputError(code, message)


class ExecutionSpace(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    worker: str
    token_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    max_jobs: int = Field(default=4, ge=1, le=64)
    max_repetitions: int = Field(default=3, ge=1, le=32)
    max_total_agent_seconds: float = Field(default=3600, gt=0, le=604_800)
    max_total_requests: int = Field(default=1000, ge=1, le=1_000_000)
    max_total_completion_tokens: int = Field(default=1_000_000, ge=1, le=1_000_000_000)

    def checked(self, workbench: ResearchWorkbench) -> ExecutionSpace:
        endpoint = worker_url(self.worker)
        if endpoint in workbench.allowed_workers and workbench.allowed_workers[endpoint] != self.token_env:
            raise ValueError("execution worker credential approval conflicts")
        workbench.allowed_workers[endpoint] = self.token_env
        return self


class ExecutionManager:
    def __init__(self, space: ExecutionSpace, store: Store, workbench: ResearchWorkbench,
                 session_id: str, context_sha256: str):
        self.space, self.store, self.workbench = space, store, workbench
        self.session_id, self.context_sha256 = session_id, context_sha256

    def capabilities(self) -> dict[str, Any]:
        return {"split": "development", "max_jobs": self.space.max_jobs,
                "max_repetitions": self.space.max_repetitions,
                "reservation_limits": {"agent_seconds": self.space.max_total_agent_seconds,
                                       "requests": self.space.max_total_requests,
                                       "completion_tokens": self.space.max_total_completion_tokens},
                "wall_clock_limit": "unsupported; original task agent timeouts exclude setup and verification",
                "execution": "existing_operator_approved_worker; no_resource_allocation",
                "lifecycle": "advance_submits_one_trial; fetch_cancel_resubmit_are_explicit",
                "runtime_acceptance": "unverified"}

    def _owned(self, identifier: str, kind: str, *, job: bool = False) -> dict[str, Any]:
        try:
            value = self.store.job(identifier) if job else self.store.get(identifier, kind)
            if (value.get("session_id") != self.session_id or value.get("context_sha256") != self.context_sha256
                    or (job and value.get("kind") != kind)):
                raise ValueError("ownership")
            return value
        except (ValueError, KeyError, TypeError):
            _fail("unknown_development_handle", "Choose a development handle returned in this session.")

    def _identity(self, experiment_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        experiment = self.workbench.store.get(experiment_id, "experiment")
        if experiment.get("split") != "development" or experiment.get("state_mode") != "fresh_world":
            _fail("development_only", "Development execution requires a fresh development candidate.")
        if not agent_bindings(experiment):
            _fail("serving_policy_unbound", "Choose a ready approved policy for every agent before execution; "
                  "the unchanged base manifest alone has no verified serving binding. "
                  "Structural check and final submission remain available without serving.")
        validation = self.workbench.validate_experiment(experiment_id)
        if not validation.get("valid"):
            _fail("candidate_invalid", "The frozen candidate no longer passes native validation; prepare it again.")
        with tempfile.TemporaryDirectory(dir=self.workbench.store.root) as temporary:
            root = Path(temporary) / "inputs"
            package = unpack_submission(self.workbench._archive(experiment_id), root)
            path = root / package["entrypoint"]
            world = load_manifest(path)
            snapshot = input_snapshot(path)
            if not snapshot["tasks"] or snapshot["unresolved_tasks"] or world.stream.dir:
                _fail("unbounded_development_tasks", "Use a frozen nonempty task collection with no task stream.")
            agent_seconds = 0.0
            for task_spec in world.tasks:
                task = load_task(task_spec.source, path.parent / str(task_spec.path),
                                 dataset=task_spec.dataset, name=task_spec.name)
                timeout = (task.config.get("agent") or {}).get("timeout_sec")
                if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                        or not math.isfinite(timeout) or timeout <= 0):
                    _fail("unbounded_agent_time", "Every original task must declare a finite positive agent timeout.")
                agent_seconds += timeout * len(world.agents)
            if any(agent.budget.requests is None or agent.budget.completion_tokens is None for agent in world.agents):
                _fail("unbounded_model_budget", "The operator must freeze finite agent request and completion budgets.")
            identity = {"tasks": [{key: value for key, value in task.items() if key != "path"}
                                  for task in snapshot["tasks"]],
                        "integrations": snapshot["integrations"], "behavior_id": experiment["behavior_id"],
                        "judge": world.judge.model_dump(mode="json"), "observe": world.observe.model_dump(mode="json"),
                        "budgets": [{"slot": index, "budget": agent.budget.model_dump(mode="json")}
                                    for index, agent in enumerate(world.agents)],
                        "resources": world.resources.model_dump(mode="json")}
            budgets = {"agent_seconds": agent_seconds,
                       "requests": sum(agent.budget.requests or 0 for agent in world.agents),
                       "completion_tokens": sum(agent.budget.completion_tokens or 0 for agent in world.agents)}
        return identity, budgets

    @staticmethod
    def _grading_scope(candidate: dict[str, Any]) -> dict[str, Any]:
        scope = candidate["grading_scope"]
        slots = scope["graded_agent_slots"]
        return {**scope, "graded_agent_slots": slots[:20], "graded_agent_slots_total": len(slots),
                "graded_agent_slots_truncated": len(slots) > 20}

    def prepare_candidate(self, validation_id: str) -> dict[str, Any]:
        validation = self.store.get(validation_id, "designer_validation")
        if validation.get("context_sha256") != self.context_sha256 or not validation["validation"].get("valid"):
            _fail("candidate_invalid", "Prepare a successfully checked candidate in this session.")
        experiment_id = validation["configuration"]["experiment_id"]
        identity, budgets = self._identity(experiment_id)
        value = {"session_id": self.session_id, "context_sha256": self.context_sha256,
                 "validation_id": validation_id, "experiment_id": experiment_id,
                 "spec_sha256": validation["spec_sha256"], "comparison_identity": identity,
                 "comparison_sha256": digest(identity), "budgets": budgets}
        manifest = self.workbench.store.get(experiment_id, "experiment")["configuration"]["manifest"]
        finals = ((manifest.get("structure") or {}).get("execution") or {}).get("finalizers")
        slots = [f"agent-{index + 1}" for index, agent in enumerate(manifest["agents"])
                 if finals is None or agent["id"] in finals]
        value["grading_scope"] = {"graded_agent_slots": slots, "agents": len(manifest["agents"]),
                                  "original_verifier_units_per_trial": len(slots) * len(identity["tasks"]),
                                  "ungraded_agent_task_units_per_trial": (len(manifest["agents"]) - len(slots))
                                  * len(identity["tasks"])}
        identifier = self.store.put("designer_candidate", value)
        return {"candidate_id": identifier, "spec_sha256": value["spec_sha256"], "validated": True,
                "split": "development", "per_trial_reservation": budgets, "grading_scope": self._grading_scope(value),
                "runtime_acceptance": "unverified"}

    def _candidate(self, identifier: str) -> dict[str, Any]:
        value = self._owned(identifier, "designer_candidate")
        identity, budgets = self._identity(value["experiment_id"])
        if identity != value["comparison_identity"] or budgets != value["budgets"]:
            _fail("development_context_changed", "Frozen development inputs changed; restore the operator setup.")
        return value

    def _runs(self) -> list[dict[str, Any]]:
        with self.store.connection() as db:
            values = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
        return [value for value in values if value.get("kind") == "designer_execution"
                and value.get("session_id") == self.session_id and value.get("context_sha256") == self.context_sha256]

    async def run_candidate(self, candidate_id: str, request_id: str, repetitions: int = 1) -> dict[str, Any]:
        if not isinstance(request_id, str) or not _REQUEST.fullmatch(request_id):
            _fail("invalid_request_id", "Use 1 to 64 letters, digits, underscores or hyphens for request_id.")
        if type(repetitions) is not int or not 1 <= repetitions <= self.space.max_repetitions:
            _fail("execution_budget", "Choose repetitions within the advertised execution bound.")
        candidate = self._candidate(candidate_id)
        args = {"candidate_id": candidate_id, "repetitions": repetitions, "context_sha256": self.context_sha256}
        key = "designer-execution:" + self.session_id + ":" + request_id
        identifier = "designer_run-" + digest({"key": key, "arguments": args})
        with self.store.lease("designer-execution:" + self.session_id):
            with self.store.connection() as db:
                existing = db.execute("SELECT digest FROM requests WHERE key=?", (key,)).fetchone()
            if existing is not None:
                if existing[0] != digest({"tool": "designer_execution", "arguments": args}):
                    _fail("request_conflict", "This request_id is already bound to different execution inputs.")
                return self._view(self._owned(identifier, "designer_execution", job=True))
            reservation = {name: amount * repetitions for name, amount in candidate["budgets"].items()}
            previous = self._runs()
            limits = {"agent_seconds": self.space.max_total_agent_seconds, "requests": self.space.max_total_requests,
                      "completion_tokens": self.space.max_total_completion_tokens}
            if (sum(len(value["trials"]) for value in previous) + repetitions > self.space.max_jobs
                    or any(sum(value["reservation"][name] for value in previous) + reservation[name] > limit
                           for name, limit in limits.items())):
                _fail("execution_budget", "The session execution reservation is exhausted; "
                      "cancelled trials stay reserved.")
            state = {"kind": "designer_execution", "session_id": self.session_id,
                     "context_sha256": self.context_sha256, "candidate_id": candidate_id, "status": "planned",
                     "reservation": reservation, "trials": [{"index": index, "status": "not_run"}
                                                              for index in range(repetitions)]}
            self.store.begin(key, "designer_execution", args, job=(identifier, state))
            self.store.finish(key, {"run_id": identifier})
        return await self.run_status(identifier, "advance")

    def _native_job(self, state: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
        job = self.workbench.store.job(trial["native_job_id"])
        candidate = self._candidate(state["candidate_id"])
        if (job.get("kind") != "world" or job.get("experiment_id") != candidate["experiment_id"]
                or worker_url(job["worker"]) != worker_url(self.space.worker)
                or job.get("token_env") != self.space.token_env):
            _fail("development_context_changed", "The execution job no longer matches its frozen "
                  "development candidate.")
        return job

    async def _submit(self, state: dict[str, Any], trial: dict[str, Any]) -> None:
        candidate = self._candidate(state["candidate_id"])
        args = {"experiment_id": candidate["experiment_id"], "worker": worker_url(self.space.worker),
                "token_env": self.space.token_env}
        key = state["job_id"] + ":" + str(trial["index"])
        trial.update(native_job_id="job-" + digest({"key": key, "tool": "run_experiment", "args": args}),
                     status="submitting")
        state["status"] = "running"
        self.store.save_job(state["job_id"], state)
        try:
            await self.workbench.run_experiment(**args, idempotency_key=key)
            trial["status"] = self._native_job(state, trial)["status"]
        except Exception as exc:
            # Durable native admission, if present, owns recovery at the same identity.
            try:
                trial["status"] = self._native_job(state, trial)["status"]
            except KeyError:
                trial.update(status="submission_not_created", failure="submission_not_created")
            self.store.put("designer_execution_diagnostic", {"run_id": state["job_id"], "type": type(exc).__name__,
                                                             "message": str(exc)})

    def _evaluate(self, state: dict[str, Any], trial: dict[str, Any], job: dict[str, Any]) -> None:
        candidate = self._candidate(state["candidate_id"])
        try:
            root = self.workbench._verified_results(job)
            runs = list((root / "runs").iterdir())
            if len(runs) != 1:
                raise ValueError("expected one original world run")
            run, binding = self.workbench._run_binding(job["job_id"], runs[0].name)
            snapshot = json.loads((run / "worker-binding.json").read_text())["input_snapshot"]
            identity = candidate["comparison_identity"]
            tasks = [{key: value for key, value in task.items() if key != "path"} for task in snapshot["tasks"]]
            if tasks != identity["tasks"] or snapshot["integrations"] != identity["integrations"]:
                raise ValueError("actual task/verifier or runtime source identity changed")
            definition = self.workbench.store.get(identity["behavior_id"], "behavior")
            if definition.get("kind") == "collective":
                evaluated = self.workbench.evaluate_collective(identity["behavior_id"],
                                                               job_id=job["job_id"], run_id=runs[0].name)
            else:
                evaluated = self.workbench.evaluate_run(job["job_id"], runs[0].name)
            exported = self.workbench.export_reproducer(candidate["experiment_id"])
            rows = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
            trial.update(status="evaluated", evaluation_id=evaluated["evaluation_id"],
                         evaluation_kind="collective" if definition.get("kind") == "collective" else "behavior",
                         binding=binding, export=exported,
                         original_verifier_units=sum(row["kind"] == "verifier.reward" for row in rows))
        except (KeyError, ValueError, OSError) as exc:
            trial.update(status="quarantined", failure="runtime_lineage_or_evaluation_rejected")
            self.store.put("designer_execution_diagnostic", {"run_id": state["job_id"], "type": type(exc).__name__,
                                                             "message": str(exc)})

    async def run_status(self, run_id: str, action: Action = "inspect",
                         since_version: str | None = None) -> dict[str, Any]:
        if action not in {"inspect", "advance", "fetch", "cancel", "resubmit"}:
            _fail("invalid_execution_action", "Choose an advertised execution action.")
        if since_version is not None and (action != "inspect" or not re.fullmatch(r"[a-f0-9]{64}", since_version)):
            _fail("invalid_state_version", "Use a returned state_version only with inspect.")
        with self.store.lease("designer-run:" + str(run_id)):
            state = self._owned(run_id, "designer_execution", job=True)
            self._candidate(state["candidate_id"])
            if state["status"] == "halted" and action in {"advance", "resubmit"}:
                _fail("execution_halted", "This run halted on integrity or cleanup evidence; "
                      "the operator must inspect it.")
            trial = next((value for value in state["trials"] if value["status"] not in _DONE), None)
            if trial is not None:
                if action == "cancel":
                    state["cancel_requested"] = True
                    for pending in state["trials"]:
                        if pending["status"] == "not_run":
                            pending["status"] = "not_run_cancelled"
                if trial.get("native_job_id") and trial["status"] not in _DONE:
                    try:
                        native = self._native_job(state, trial)
                    except KeyError:
                        native = None
                    if native is None:
                        if action == "resubmit":
                            await self._submit(state, trial)
                        elif action == "cancel":
                            trial["status"] = "not_run_cancelled"
                    elif native["status"] == "preflight_failed":
                        trial.update(status="failed", failure="serving_preflight_failed")
                    elif action in {"inspect", "fetch", "cancel", "resubmit"}:
                        try:
                            native = await self.workbench.get_job_status(native["job_id"], action)
                            trial["status"] = native["status"]
                            if native["status"] in TERMINAL and native["status"] != "completed":
                                trial.update(status="failed", failure=native["status"])
                            cleanup = (native.get("remote") or {}).get("cleanup") or {}
                            if native["status"] == "failed_cleanup":
                                state.update(status="halted", halt_reason="failed_cleanup")
                            elif native["status"] in TERMINAL and cleanup.get("status") != "complete":
                                state.update(status="halted", halt_reason="cleanup_not_confirmed")
                            elif action == "fetch" and native["status"] == "completed":
                                self._evaluate(state, trial, native)
                        except Exception as exc:
                            self.store.put("designer_execution_diagnostic", {"run_id": run_id,
                                             "type": type(exc).__name__, "message": str(exc)})
                            self.store.save_job(run_id, state)
                            _fail("execution_action_failed", "The worker action failed. Inspect this run; "
                                  "fetch, cancellation and same-identity resubmission require explicit actions.")
                elif action == "advance" and trial["status"] == "not_run" and not state.get("cancel_requested"):
                    await self._submit(state, trial)
            if all(value["status"] in _DONE for value in state["trials"]) and state["status"] != "halted":
                state["status"] = "cancelled" if state.get("cancel_requested") else "completed"
            if any(value["status"] == "quarantined" for value in state["trials"]):
                state.update(status="halted", halt_reason="runtime_lineage_rejected")
            self.store.save_job(run_id, state)
            return self._view(state, since_version)

    def _measurements(self, state: dict[str, Any]) -> dict[str, Any]:
        behavior: Counter[str] = Counter()
        collective: list[dict[str, Any]] = []
        for trial in state["trials"]:
            if trial["status"] != "evaluated":
                behavior["unknown"] += 1
            elif trial["evaluation_kind"] == "behavior":
                value = self.workbench._evaluation(trial["evaluation_id"])
                behavior[value["status"]] += 1
            else:
                collective.append(self.workbench._collective_evaluation(trial["evaluation_id"]))
        n = len(state["trials"])
        candidate = self._owned(state["candidate_id"], "designer_candidate")
        definition = self.workbench.store.get(candidate["comparison_identity"]["behavior_id"], "behavior")
        if definition.get("kind") == "collective":
            result: dict[str, Any] = {"kind": "collective", "evaluated_trials": len(collective),
                                      "unknown_trials": n - len(collective), "measurements": {}}
            for key in ("prevalence", "review_acceptance"):
                rows = [value[key] for value in collective]
                numerator, denominator = (sum(value[field] for value in rows)
                                          for field in ("numerator", "denominator"))
                known = len(collective) == n and all(value["status"] == "measured" for value in rows)
                result["measurements"][key] = {"numerator": numerator, "observed_denominator": denominator,
                                              "unknown_trials": n - sum(v["status"] == "measured" for v in rows),
                                              "value": numerator / denominator if known and denominator else None}
            return result
        return {"kind": "behavior", "counts": {key: behavior[key] for key in ("present", "absent", "unknown")},
                "presence_bounds": [behavior["present"] / n, (behavior["present"] + behavior["unknown"]) / n]}

    def _view(self, state: dict[str, Any], since_version: str | None = None) -> dict[str, Any]:
        counts = dict(Counter(value["status"] for value in state["trials"]))
        candidate = self._owned(state["candidate_id"], "designer_candidate")
        result = {"run_id": state["job_id"], "candidate_id": state["candidate_id"], "status": state["status"],
                  "planned_trials": len(state["trials"]), "trial_statuses": counts,
                  "reservation": state["reservation"], "measurements": self._measurements(state),
                  "grading_scope": self._grading_scope(candidate),
                  "original_verifier_units_observed": sum(trial.get("original_verifier_units", 0)
                                                          for trial in state["trials"]),
                  "bounds_are_confidence_intervals": False, "failed_trials": "retained_without_replacement"}
        if state.get("halt_reason"):
            result["halt_reason"] = state["halt_reason"]
        version = digest(result)
        envelope = {"run_id": state["job_id"], "status": state["status"], "state_version": version,
                    "unchanged": since_version == version,
                    "retry_after_seconds": 5 if state["status"] == "running" else None}
        return envelope if since_version == version else {**result, **envelope}

    def compare_runs(self, baseline_run_id: str, candidate_run_id: str) -> dict[str, Any]:
        states = [self._owned(identifier, "designer_execution", job=True)
                  for identifier in (baseline_run_id, candidate_run_id)]
        candidates = [self._candidate(state["candidate_id"]) for state in states]
        if candidates[0]["comparison_identity"] != candidates[1]["comparison_identity"]:
            _fail("incomparable_development_runs", "Both runs must preserve the same original tasks, verifier, "
                  "evaluation, runtime source and model budget identities.")
        # Every admitted measurement is reverified, including fetched bytes and native bindings.
        views = [self._view(state) for state in states]
        evaluations = [trial["evaluation_id"] for state in states for trial in state["trials"]
                       if trial["status"] == "evaluated"]
        kind = self.workbench.store.get(candidates[0]["comparison_identity"]["behavior_id"], "behavior").get("kind")
        native = None
        if evaluations:
            if kind == "collective":
                native = self.workbench.evaluate_collective(candidates[0]["comparison_identity"]["behavior_id"],
                                                            evaluation_ids=evaluations)["comparison_id"]
            else:
                native = self.workbench.compare_candidates(evaluations)["comparison_id"]
        private = {"session_id": self.session_id, "context_sha256": self.context_sha256,
                   "runs": [baseline_run_id, candidate_run_id], "native_comparison_id": native,
                   "comparison_identity": candidates[0]["comparison_identity"], "views": views}
        identifier = self.store.put("designer_comparison", private)
        result = {"comparison_id": identifier, "baseline": views[0], "candidate": views[1],
                  "same_task_verifier_and_evaluation": True,
                  "outcome_comparability": ("same_grading_scope" if candidates[0]["grading_scope"]
                                            == candidates[1]["grading_scope"] else "partial; different_grading_scopes"),
                  "interpretation": "descriptive within the frozen task set; repeated runs are not independent tasks",
                  "behavioral_improvement_demonstrated": False}
        if kind != "collective" and candidates[0]["grading_scope"] == candidates[1]["grading_scope"]:
            left, right = [view["measurements"]["presence_bounds"] for view in views]
            result["candidate_minus_baseline_presence_bounds"] = [right[0] - left[1], right[1] - left[0]]
        return result
