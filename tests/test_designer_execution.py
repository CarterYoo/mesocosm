"""Execution boundary software contracts; no real worker or agent acceptance claim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from test_reproduction_runtime import fetched
from test_reproduction_runtime import loop as native_setup

from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.store import Store, digest
from mesocosm.research_mcp.execution import ExecutionManager, ExecutionSpace


@pytest.fixture
def execution(tmp_path):
    bench, _, _, deployments = native_setup.__wrapped__(tmp_path)
    source = tmp_path / "world-source/world.yaml"
    (source.parent / "task/task.toml").write_text("[agent]\ntimeout_sec = 10\n")
    behavior = bench.define_behavior({"name": "HIDDEN-rubric", "stages": {
        "onset": {"event": {"kind": "model_call"}}}})["behavior_id"]
    configuration = bench.configure_experiment(str(source), behavior, [], deployment_id=deployments["base"])
    store = Store(tmp_path / "designer-private")
    context = digest("execution-contract-context")
    validation_id = store.put("designer_validation", {"context_sha256": context,
                              "spec_sha256": digest({"changes": []}), "configuration": configuration,
                              "validation": bench.validate_experiment(configuration["experiment_id"])})
    space = ExecutionSpace(worker="http://127.0.0.1:8088", token_env="WORKER_KEY", max_jobs=8,
                           max_repetitions=3, max_total_agent_seconds=300.0,
                           max_total_requests=100, max_total_completion_tokens=10_000).checked(bench)
    manager = ExecutionManager(space, store, bench, "session-a", context)
    prepared = manager.prepare_candidate(validation_id)
    return manager, prepared, source, validation_id


@pytest.fixture
def transport(monkeypatch):
    calls = []
    state = {"status": "queued", "error": False, "cleanup": {"status": "complete"}}

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, archive, job_id, **kwargs):
            calls.append(("submit", job_id))
            if state["error"]:
                raise ConnectionError("HIDDEN /private/path secret-token")
            return {"job_id": job_id, "status": state["status"], "digest": kwargs["digest"]}

        async def status(self, job_id):
            calls.append(("status", job_id))
            return {"job_id": job_id, "status": state["status"], "digest": state["digest"], "cleanup": state["cleanup"]}

        async def stop(self, job_id):
            calls.append(("cancel", job_id))
            state["status"] = "cancelled"
            return {"job_id": job_id, "status": "cancelled", "digest": state["digest"], "cleanup": state["cleanup"]}

    async def preflight(experiment):
        return {identifier: {"deployment_id": identifier}
                for identifier in set((experiment.get("agent_deployments") or {}).values())
                | ({experiment["deployment_id"]} if experiment.get("deployment_id") else set())}

    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    monkeypatch.setattr("mesocosm.reproduction.service.ResearchWorkbench._serving_preflight",
                        lambda self, experiment: preflight(experiment))
    monkeypatch.setenv("WORKER_KEY", "private-software-fixture-token")
    return state, calls


def native_candidate(manager, public):
    return manager._candidate(public["candidate_id"])["experiment_id"]


def private_run(manager, public):
    return manager.store.job(public["run_id"])


def bind_digest(manager, public, transport):
    transport[0]["digest"] = manager.workbench.store.get(native_candidate(manager, public), "experiment")["package"][
        "digest"]


def test_prepare_uses_real_native_validation_and_closed_public_identity(execution):
    manager, prepared, source, validation_id = execution
    assert prepared["validated"] and prepared["split"] == "development"
    assert prepared["per_trial_reservation"] == {"agent_seconds": 30.0, "requests": 6, "completion_tokens": 768}
    assert manager.prepare_candidate(validation_id) == prepared
    public = json.dumps([prepared, manager.capabilities()])
    assert "HIDDEN" not in public and str(source.parent) not in public and "WORKER_KEY" not in public
    assert native_candidate(manager, prepared) not in public
    assert "unsupported" in manager.capabilities()["wall_clock_limit"]
    other = ExecutionManager(manager.space, manager.store, manager.workbench, "session-b", digest("other"))
    with pytest.raises(ToolInputError, match="this session"):
        other._candidate(prepared["candidate_id"])
    with pytest.raises(ToolInputError):
        manager._candidate(native_candidate(manager, prepared))


@pytest.mark.parametrize("mutation,code", [
    ("test", "development_only"), ("no_timeout", "unbounded_agent_time"),
    ("no_budgets", "unbounded_model_budget"),
])
def test_original_finite_task_and_budget_contracts_are_required(execution, mutation, code):
    manager, _, source, validation_id = execution
    previous = manager.store.get(validation_id, "designer_validation")
    if mutation == "no_timeout":
        (source.parent / "task/task.toml").unlink()
    if mutation == "no_budgets":
        raw = yaml.safe_load(source.read_text())
        for agent in raw["agents"]:
            agent.pop("budget")
        source.write_text(yaml.safe_dump(raw))
    experiment = manager.workbench.store.get(previous["configuration"]["experiment_id"], "experiment")
    configuration = manager.workbench.configure_experiment(str(source), experiment["behavior_id"], [],
                         deployment_id=experiment["deployment_id"], split="test" if mutation == "test" else "development")
    changed = manager.store.put("designer_validation", {**previous, "configuration": configuration,
                              "validation": manager.workbench.validate_experiment(configuration["experiment_id"])})
    with pytest.raises(ToolInputError) as error:
        manager.prepare_candidate(changed)
    assert error.value.code == code


@pytest.mark.asyncio
async def test_trial_reservations_request_identity_and_compact_polling(execution, transport):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    first = await manager.run_candidate(prepared["candidate_id"], "once", repetitions=2)
    assert first["planned_trials"] == 2 and first["trial_statuses"] == {"queued": 1, "not_run": 1}
    assert first["measurements"]["counts"] == {"present": 0, "absent": 0, "unknown": 2}
    assert await manager.run_candidate(prepared["candidate_id"], "once", repetitions=2) == first
    assert len(transport[1]) == 1
    with pytest.raises(ToolInputError) as conflict:
        await manager.run_candidate(prepared["candidate_id"], "once", repetitions=1)
    assert conflict.value.code == "request_conflict"
    polled = await manager.run_status(first["run_id"], since_version=first["state_version"])
    assert polled["unchanged"] and "measurements" not in polled and "trial_statuses" not in polled
    assert len(transport[1]) == 2
    assert private_run(manager, first)["trials"][0]["native_job_id"] not in json.dumps(first)


@pytest.mark.asyncio
async def test_same_native_identity_recovers_only_on_explicit_resubmit(execution, transport):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    transport[0]["error"] = True
    first = await manager.run_candidate(prepared["candidate_id"], "lost-response")
    assert first["trial_statuses"] == {"submission_uncertain": 1}
    await manager.run_candidate(prepared["candidate_id"], "lost-response")
    assert len(transport[1]) == 1
    transport[0]["error"] = False
    recovered = await manager.run_status(first["run_id"], "resubmit")
    assert recovered["trial_statuses"] == {"queued": 1}
    assert transport[1] == [("submit", transport[1][0][1]), ("submit", transport[1][0][1])]
    assert "HIDDEN" not in json.dumps([first, recovered])


@pytest.mark.asyncio
async def test_failure_before_native_job_creation_recovers_without_another_trial(execution, transport, monkeypatch):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    submit = manager.workbench.run_experiment

    async def fail_before_creation(**kwargs):
        raise ConnectionError("HIDDEN admission temporarily unavailable")

    monkeypatch.setattr(manager.workbench, "run_experiment", fail_before_creation)
    first = await manager.run_candidate(prepared["candidate_id"], "before-admission")
    assert first["trial_statuses"] == {"submission_not_created": 1}
    job_id = private_run(manager, first)["trials"][0]["native_job_id"]
    assert await manager.run_candidate(prepared["candidate_id"], "before-admission") == first
    monkeypatch.setattr(manager.workbench, "run_experiment", submit)
    recovered = await manager.run_status(first["run_id"], "resubmit")
    assert recovered["trial_statuses"] == {"queued": 1} and recovered["planned_trials"] == 1
    assert private_run(manager, first)["trials"][0]["native_job_id"] == job_id
    assert transport[1] == [("submit", job_id[-32:])]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("max_total_agent_seconds", 29.0), ("max_total_requests", 5),
                                         ("max_total_completion_tokens", 767)])
async def test_each_execution_reservation_bound_blocks_before_worker_admission(execution, transport, field, value):
    manager, prepared, _, _ = execution
    setattr(manager.space, field, value)
    with pytest.raises(ToolInputError) as exhausted:
        await manager.run_candidate(prepared["candidate_id"], "over-budget")
    assert exhausted.value.code == "execution_budget" and not transport[1]
    assert not manager._runs()


@pytest.mark.asyncio
async def test_cancel_preserves_failed_and_unrun_denominators_and_reservations(execution, transport):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    first = await manager.run_candidate(prepared["candidate_id"], "cancel", repetitions=3)
    cancelled = await manager.run_status(first["run_id"], "cancel")
    assert cancelled["status"] == "cancelled" and cancelled["planned_trials"] == 3
    assert cancelled["trial_statuses"] == {"failed": 1, "not_run_cancelled": 2}
    assert cancelled["measurements"]["counts"]["unknown"] == 3
    assert cancelled["reservation"]["requests"] == 18
    manager.space.max_jobs = 3
    with pytest.raises(ToolInputError) as exhausted:
        await manager.run_candidate(prepared["candidate_id"], "after-cancel")
    assert exhausted.value.code == "execution_budget"


@pytest.mark.asyncio
async def test_inspect_never_fetches_or_evaluates_and_failed_trial_needs_explicit_advance(execution, transport):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    first = await manager.run_candidate(prepared["candidate_id"], "failed", repetitions=2)
    transport[0]["status"] = "failed"
    status = await manager.run_status(first["run_id"])
    assert status["trial_statuses"] == {"failed": 1, "not_run": 1}
    assert len([call for call in transport[1] if call[0] == "submit"]) == 1
    assert not any(trial.get("evaluation_id") for trial in private_run(manager, first)["trials"])
    transport[0]["status"] = "queued"
    advanced = await manager.run_status(first["run_id"], "advance")
    assert advanced["trial_statuses"] == {"failed": 1, "queued": 1}
    assert len([call for call in transport[1] if call[0] == "submit"]) == 2


@pytest.mark.asyncio
async def test_cleanup_failure_halts_without_submitting_later_trials(execution, transport):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    first = await manager.run_candidate(prepared["candidate_id"], "cleanup", repetitions=2)
    transport[0]["status"] = "failed_cleanup"
    status = await manager.run_status(first["run_id"])
    assert status["status"] == "halted" and status["halt_reason"] == "failed_cleanup"
    with pytest.raises(ToolInputError) as halted:
        await manager.run_status(first["run_id"], "advance")
    assert halted.value.code == "execution_halted"
    assert len([call for call in transport[1] if call[0] == "submit"]) == 1


@pytest.mark.asyncio
async def test_compare_retains_unmeasured_trials_and_rejects_changed_task_verifier(execution, transport):
    manager, prepared, source, validation_id = execution
    bind_digest(manager, prepared, transport)
    baseline = await manager.run_candidate(prepared["candidate_id"], "baseline", repetitions=2)
    designed = await manager.run_candidate(prepared["candidate_id"], "design", repetitions=1)
    compared = manager.compare_runs(baseline["run_id"], designed["run_id"])
    assert compared["baseline"]["planned_trials"] == 2 and compared["candidate"]["planned_trials"] == 1
    assert compared["candidate_minus_baseline_presence_bounds"] == [-1, 1]
    assert compared["behavioral_improvement_demonstrated"] is False
    (source.parent / "task/tests/test.sh").write_text("#!/bin/sh\nexit 1\n")
    previous = manager.store.get(validation_id, "designer_validation")
    experiment = manager.workbench.store.get(previous["configuration"]["experiment_id"], "experiment")
    configuration = manager.workbench.configure_experiment(str(source), experiment["behavior_id"], [],
                                                           deployment_id=experiment["deployment_id"])
    changed = manager.store.put("designer_validation", {**previous, "configuration": configuration,
                              "validation": manager.workbench.validate_experiment(configuration["experiment_id"])})
    different = manager.prepare_candidate(changed)
    other = await manager.run_candidate(different["candidate_id"], "changed-verifier")
    with pytest.raises(ToolInputError) as rejected:
        manager.compare_runs(baseline["run_id"], other["run_id"])
    assert rejected.value.code == "incomparable_development_runs"


@pytest.mark.asyncio
async def test_different_finalizer_scope_is_explicitly_partial_and_never_zeroed(execution, transport):
    manager, prepared, source, validation_id = execution
    bind_digest(manager, prepared, transport)
    baseline = await manager.run_candidate(prepared["candidate_id"], "all-graded")
    raw = yaml.safe_load(source.read_text())
    raw.update(isolate_agents=True, structure={"execution": {
        "depends_on": {"agent-2": ["agent-0", "agent-1"]}, "finalizers": ["agent-2"]}})
    raw["services"].update(enabled=True, access={"artifacts": {}})
    source.write_text(yaml.safe_dump(raw))
    previous = manager.store.get(validation_id, "designer_validation")
    experiment = manager.workbench.store.get(previous["configuration"]["experiment_id"], "experiment")
    configuration = manager.workbench.configure_experiment(str(source), experiment["behavior_id"], [],
                                                           deployment_id=experiment["deployment_id"])
    changed = manager.store.put("designer_validation", {**previous, "configuration": configuration,
                              "validation": manager.workbench.validate_experiment(configuration["experiment_id"])})
    different = manager.prepare_candidate(changed)
    assert different["grading_scope"]["graded_agent_slots"] == ["agent-3"]
    assert different["grading_scope"]["ungraded_agent_task_units_per_trial"] == 2
    other = await manager.run_candidate(different["candidate_id"], "only-finalizer-graded")
    compared = manager.compare_runs(baseline["run_id"], other["run_id"])
    assert compared["outcome_comparability"] == "partial; different_grading_scopes"
    assert "candidate_minus_baseline_presence_bounds" not in compared
    assert compared["candidate"]["measurements"]["counts"] == {"present": 0, "absent": 0, "unknown": 1}


@pytest.mark.asyncio
async def test_explicit_fetch_reconciles_actual_immutable_bundle_before_measurement(execution, transport,
                                                                                   tmp_path, monkeypatch):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    first = await manager.run_candidate(prepared["candidate_id"], "native-fixture")
    state = private_run(manager, first)
    job_id = state["trials"][0]["native_job_id"]
    fetched(manager.workbench, native_candidate(manager, prepared), tmp_path / "native", owned_job_id=job_id)
    original_status = manager.workbench.get_job_status

    async def already_fetched(identifier, action="inspect", **kwargs):
        if action == "fetch":
            job = manager.workbench.store.job(identifier)
            # Project the original fetched worker record just as the real client does.
            return {**job, "remote": json.loads((Path(job["results_path"]) / "job.json").read_text())}
        return await original_status(identifier, action, **kwargs)

    monkeypatch.setattr(manager.workbench, "get_job_status", already_fetched)
    status = await manager.run_status(first["run_id"], "fetch")
    assert status["status"] == "completed" and status["trial_statuses"] == {"evaluated": 1}
    assert status["measurements"]["counts"] == {"present": 1, "absent": 0, "unknown": 0}
    private = private_run(manager, first)["trials"][0]
    assert private["binding"]["task_set_sha256"] and private["export"]["export_id"]
    public = json.dumps([status, manager.compare_runs(first["run_id"], first["run_id"])])
    for secret in (job_id, str(tmp_path), private["evaluation_id"], private["binding"]["run_id"], "/pinned/base"):
        assert secret not in public
    result = Path(manager.workbench.store.job(job_id)["results_path"]) / "runs/run_contract/summary.json"
    result.write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        manager.compare_runs(first["run_id"], first["run_id"])


@pytest.mark.asyncio
async def test_unknown_cleanup_halts_before_next_trial(execution, transport):
    manager, prepared, _, _ = execution
    bind_digest(manager, prepared, transport)
    first = await manager.run_candidate(prepared["candidate_id"], "unknown-cleanup", repetitions=2)
    transport[0].update(status="failed", cleanup=None)
    status = await manager.run_status(first["run_id"])
    assert status["status"] == "halted" and status["halt_reason"] == "cleanup_not_confirmed"
    assert status["measurements"]["counts"]["unknown"] == 2
    with pytest.raises(ToolInputError) as halted:
        await manager.run_status(first["run_id"], "advance")
    assert halted.value.code == "execution_halted"
    assert len([call for call in transport[1] if call[0] == "submit"]) == 1


def test_public_grading_scope_is_bounded_but_private_identity_retains_all_slots(execution):
    manager, _, _, _ = execution
    slots = [f"agent-{index}" for index in range(1, 2001)]
    candidate = {"grading_scope": {"graded_agent_slots": slots, "agents": len(slots)}}
    public = manager._grading_scope(candidate)
    assert public["graded_agent_slots"] == slots[:20]
    assert public["graded_agent_slots_total"] == 2000 and public["graded_agent_slots_truncated"]
    assert candidate["grading_scope"]["graded_agent_slots"] == slots


def test_unbound_base_policy_is_rejected_before_execution_reservation(execution):
    manager, _, source, validation_id = execution
    previous = manager.store.get(validation_id, "designer_validation")
    experiment = manager.workbench.store.get(previous["configuration"]["experiment_id"], "experiment")
    configuration = manager.workbench.configure_experiment(str(source), experiment["behavior_id"], [])
    changed = manager.store.put("designer_validation", {**previous, "configuration": configuration,
                              "validation": manager.workbench.validate_experiment(configuration["experiment_id"])})
    with pytest.raises(ToolInputError) as rejected:
        manager.prepare_candidate(changed)
    assert rejected.value.code == "serving_policy_unbound"
    assert not manager._runs()
