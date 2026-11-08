"""Restricted policy lifecycle contracts; simulated lifecycle states are not agent/learning acceptance."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from mesocosm.learning.config import policy_id
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import Store, digest
from mesocosm.research_mcp.policy import PolicyManager, PolicySpace


def deployment(bench, name, *, base="local-base", adapter=None):
    native_job = "job-" + digest({"fixture": name})
    value = {"server": "https://serving.invalid/", "server_token_env": "PRIVATE_SERVING_TOKEN",
             "deployment_job_id": native_job, "policy": {
                 "id": policy_id(base, "a" * 40, "b" * 64, adapter, "bfloat16"),
                 "model": name, "base_model": base, "base_revision": "a" * 40,
                 "base_sha256": "b" * 64, "dtype": "bfloat16", "adapter_sha256": adapter,
                 "adapter_path": "/PRIVATE/adapter" if adapter else None}}
    identifier = bench.store.put("deployment", value)
    bench.store.save_job(native_job, {"kind": "deployment", "status": "ready", "deployment_id": identifier}, new=True)
    return identifier


@pytest.fixture
def setup(tmp_path, monkeypatch):
    bench = ResearchWorkbench(tmp_path / "native")
    base = tmp_path / "PRIVATE-base"
    base.mkdir()
    source = tmp_path / "PRIVATE-conversations.json"
    source.write_text(json.dumps([{"id": name, "task_id": name, "label": "normal", "messages": [
        {"role": "user", "content": "PRIVATE observation"}, {"role": "assistant", "content": "PRIVATE action"}]
    } for name in ("train", "development")]))
    dataset = bench.prepare_dataset([str(source)], {"split_assignments": {
        "train": "train", "development": "development"}})
    settings = {"deployments": {"base": deployment(bench, "base"),
                                 "adapter": deployment(bench, "adapter", adapter="c" * 64),
                                 "other": deployment(bench, "other", base="different-base")},
                "recipes": {"sft": {"dataset_id": dataset["dataset_id"], "config": {
                    "base_model": str(base), "base_sha256": "b" * 64, "dtype": "bfloat16"},
                    "parameter_choices": {"learning_rate": [.001, .01], "lora_rank": [8, 16]},
                    "venue": {"server": "https://serving.invalid/", "token_env": "PRIVATE_SERVING_TOKEN",
                              "base_model": "local-base", "base_revision": "a" * 40, "base_sha256": "b" * 64}}}}
    space = PolicySpace.model_validate(settings)
    bench.allowed_servers = space.allowed_servers(bench)
    space.checked(bench)
    store = Store(tmp_path / "designer")
    manager = PolicyManager(space, store, bench, "session-a", "d" * 64)
    launches = []
    monkeypatch.setattr("mesocosm.reproduction.jobs.start_worker", lambda store: launches.append(store.root))
    return manager, settings, launches


def test_catalog_opaque_paginated_and_bindings_native_exhaustive(setup):
    manager, _, _ = setup
    first = manager.catalog(limit=2)
    second = manager.catalog(offset=first["next_offset"], limit=2)
    text = json.dumps([first, second])
    assert "PRIVATE" not in text and "https://" not in text and "dataset-" not in text
    assert first["next_offset"] == 2 and second["next_offset"] is None
    assert second["recipes"][0]["parameter_choices"]["lora_rank"] == [8, 16]
    assert first["training_backend"]["methods"]["sft_lora"] == "supported"
    assert first["training_backend"]["methods"]["full_finetuning"] == "unsupported"
    group = {item["policy_id"]: item["compatibility_group"] for item in first["policies"]}
    assert group["base"] == group["adapter"]
    agents = ["PRIVATE-actual-3", "PRIVATE-actual-1", "PRIVATE-actual-2"]
    assert manager.bindings(None, None, agents) == {}
    shared = manager.bindings("adapter", None, agents)["agent_deployments"]
    assert shared == dict.fromkeys(agents, manager.space.deployments["adapter"])
    mixed = manager.bindings(None, {"agent-1": "base", "agent-2": "adapter", "agent-3": "base"}, agents)
    assert mixed["agent_deployments"][agents[1]] == manager.space.deployments["adapter"]
    for policy, population, code in [
        ("base", {"agent-1": "base"}, "conflicting_policy_selection"),
        (None, {"agent-1": "base"}, "incomplete_population"),
        (None, {"agent-1": "base", "agent-2": "other", "agent-3": "base"}, "incompatible_population"),
        (manager.space.deployments["base"], None, "unknown_policy"),
    ]:
        with pytest.raises(ToolInputError) as error:
            manager.bindings(policy, population, agents)
        assert error.value.code == code and "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("field", ["base_model", "tokenizer", "device", "dtype", "target_modules", "resume",
                                   "max_steps", "max_seconds", "max_train_tokens", "max_eval_tokens"])
def test_recipe_cannot_offer_arbitrary_model_paths_execution_or_budgets(setup, field):
    manager, settings, _ = setup
    settings["recipes"]["sft"]["parameter_choices"] = {field: [1]}
    with pytest.raises(ValueError):
        PolicySpace.model_validate(settings).checked(manager.workbench)


def test_recipe_validates_actual_dataset_and_training_serving_contract(setup):
    manager, settings, _ = setup
    settings["recipes"]["sft"]["config"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="same base hash and dtype"):
        PolicySpace.model_validate(settings).checked(manager.workbench)
    settings["recipes"]["sft"]["venue"] = None
    assert PolicySpace.model_validate(settings).checked(manager.workbench)
    recipe = manager.space.recipes["sft"]
    examples = manager.workbench.store.directory("datasets", recipe.dataset_id) / "examples.jsonl"
    examples.write_text(examples.read_text() + "\n")
    with pytest.raises(ValueError, match="examples changed"):
        manager.space.checked(manager.workbench)


def test_native_training_queue_exact_config_idempotent_and_reserved_once(setup):
    manager, _, launches = setup
    result = manager.train_policy("sft", {"learning_rate": .01}, "request-1")
    assert result["status"] == "queued" and len(launches) == 1
    record = manager.store.job(result["policy_job_id"])
    native = manager.workbench.store.job(record["training_job_id"])
    assert native["kind"] == "training" and native["config"]["learning_rate"] == .01
    assert native["training_sources"] and native["dataset_id"] == manager.space.recipes["sft"].dataset_id
    assert manager.train_policy("sft", {"learning_rate": .01}, "request-1") == result
    assert len(launches) == 1
    for args, code in [(("sft", {"learning_rate": .001}, "request-1"), "request_id_reused"),
                       (("sft", {}, "request-2"), "training_budget_exhausted"),
                       (("sft", {"learning_rate": .1}, "request-3"), "invalid_policy_parameters"),
                       (("sft", {"max_steps": 1}, "request-3"), "invalid_policy_parameters")]:
        with pytest.raises(ToolInputError) as error:
            manager.train_policy(*args)
        assert error.value.code == code
    assert "PRIVATE" not in json.dumps(result) and record["training_job_id"] not in json.dumps(result)


def test_compute_reservations_are_atomic_and_never_refunded_by_failure(setup, monkeypatch):
    manager, settings, _ = setup
    settings.update(max_training_jobs=8, max_total_steps=100)
    manager.space = PolicySpace.model_validate(settings).checked(manager.workbench)
    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE backend failure")
    monkeypatch.setattr(manager.workbench, "submit_training", fail)
    def submit(index):
        try:
            return manager.train_policy("sft", {}, f"concurrent-{index}")
        except ToolInputError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(submit, range(4)))
    accepted = [value for value in results if isinstance(value, dict)]
    assert len(accepted) == 1 and accepted[0]["status"] == "submission_uncertain"
    assert results.count("training_budget_exhausted") == 3
    assert "PRIVATE" not in json.dumps(results)


@pytest.mark.parametrize("cap", ["max_total_train_tokens", "max_total_eval_tokens", "max_total_seconds"])
def test_each_aggregate_compute_ceiling_is_independently_enforced(setup, cap):
    manager, settings, launches = setup
    settings.update(max_training_jobs=2, max_total_steps=200, max_total_train_tokens=2_000_000,
                    max_total_eval_tokens=2_000_000, max_total_seconds=7200)
    settings[cap] = 3600 if cap == "max_total_seconds" else 1_000_000
    manager.space = PolicySpace.model_validate(settings).checked(manager.workbench)
    manager.train_policy("sft", {}, "one")
    with pytest.raises(ToolInputError) as error:
        manager.train_policy("sft", {}, "two")
    assert error.value.code == "training_budget_exhausted" and len(launches) == 1
    budget = manager.catalog()["training_budget"]
    name = {"max_total_train_tokens": "train_tokens", "max_total_eval_tokens": "eval_tokens",
            "max_total_seconds": "seconds"}[cap]
    assert budget["remaining"][name] == 0


@pytest.mark.asyncio
async def test_owned_status_actual_native_cancel_and_no_private_status_fields(setup):
    manager, _, _ = setup
    queued = manager.train_policy("sft", {}, "request")
    public_id = queued["policy_job_id"]
    native_id = manager.store.job(public_id)["training_job_id"]
    inspected = await manager.policy_status(public_id)
    assert inspected["status"] == "queued"
    cancelled = await manager.policy_status(public_id, action="cancel")
    assert cancelled["status"] == "cancelled"
    assert (manager.workbench.store.directory("jobs", native_id) / "cancel.requested").exists()
    assert "PRIVATE" not in json.dumps(cancelled)
    other = PolicyManager(manager.space, manager.store, manager.workbench, "session-b", "d" * 64)
    for caller, identifier in [(manager, native_id), (other, public_id)]:
        with pytest.raises(ToolInputError) as error:
            await caller.policy_status(identifier)
        assert error.value.code == "unknown_policy_job"
    with pytest.raises(ToolInputError) as error:
        await manager.activate_policy(public_id)
    assert error.value.code == "training_not_completed"


@pytest.mark.asyncio
async def test_cpu_training_recipe_without_venue_has_explicit_activation_limit(setup):
    manager, settings, _ = setup
    settings["recipes"]["sft"]["venue"] = None
    settings["recipes"]["sft"]["config"]["dtype"] = "float32"
    manager.space = PolicySpace.model_validate(settings).checked(manager.workbench)
    result = manager.train_policy("sft", {}, "cpu")
    assert manager.workbench.store.job(manager.store.job(result["policy_job_id"])["training_job_id"]
                                      )["config"]["dtype"] == "float32"
    with pytest.raises(ToolInputError) as error:
        await manager.activate_policy(result["policy_job_id"])
    assert error.value.code == "serving_not_configured"


@pytest.mark.asyncio
async def test_activation_routes_completed_native_artifact_and_uncertain_load_only_reconciles(setup, monkeypatch):
    manager, _, _ = setup
    queued = manager.train_policy("sft", {}, "request")
    public_id = queued["policy_job_id"]
    owned = manager.store.job(public_id)
    monkeypatch.setattr("mesocosm.research_mcp.policy.training_artifact", lambda store, identifier: {
        "config": {key: value for key, value in owned["config"].items() if key != "resume"}})
    activated, reconciled = [], []
    native_deploy_job = "job-" + "e" * 64
    async def activate(**kwargs):
        activated.append(kwargs)
        return {"job_id": native_deploy_job, "status": "activation_uncertain"}
    monkeypatch.setattr(manager.workbench, "deploy_policy", activate)
    uncertain = await manager.activate_policy(public_id)
    assert uncertain["status"] == "activation_uncertain" and len(activated) == 1
    assert activated[0]["training_job_id"] == owned["training_job_id"]
    assert activated[0]["dtype"] == "bfloat16" and "PRIVATE" not in activated[0]["model"]
    with pytest.raises(ToolInputError) as error:
        await manager.activate_policy(public_id)
    assert error.value.code == "activation_uncertain" and len(activated) == 1
    ready_deployment = deployment(manager.workbench, "newly-trained", adapter="f" * 64)
    async def reconcile(identifier, action):
        reconciled.append((identifier, action))
        return {"job_id": native_deploy_job, "status": "ready", "deployment_id": ready_deployment}
    monkeypatch.setattr(manager.workbench, "get_job_status", reconcile)
    ready = await manager.activate_policy(public_id, reconcile=True)
    assert ready["status"] == "ready" and ready["policy_id"].startswith("policy-")
    assert reconciled == [(native_deploy_job, "reconcile")] and len(activated) == 1
    assert manager.bindings(ready["policy_id"], None, ["one", "two", "three"])["agent_deployments"] == {
        "one": ready_deployment, "two": ready_deployment, "three": ready_deployment}
    assert "PRIVATE" not in json.dumps(ready)
    other = PolicyManager(manager.space, manager.store, manager.workbench, "other-session", "d" * 64)
    with pytest.raises(ToolInputError) as error:
        other.bindings(ready["policy_id"], None, ["one", "two", "three"])
    assert error.value.code == "unknown_policy"


@pytest.mark.asyncio
async def test_interrupted_activation_recovers_native_admission_without_reloading(setup, monkeypatch):
    manager, _, _ = setup
    public_id = manager.train_policy("sft", {}, "request")["policy_job_id"]
    value = manager.store.job(public_id)
    monkeypatch.setattr("mesocosm.research_mcp.policy.training_artifact", lambda store, identifier: {
        "config": {key: item for key, item in value["config"].items() if key != "resume"}})
    native_id = "job-" + "f" * 64
    value.update(activation_key="unfinished-native-request", status="activation_uncertain")
    manager.store.save_job(public_id, value)
    manager.workbench.store.save_job(native_id, {"kind": "deployment", "status": "activation_uncertain",
        "policy": {"model": manager._alias(public_id)},
        "training_artifact": {"training_job_id": value["training_job_id"]}}, new=True)
    ready = deployment(manager.workbench, "reconciled-trained", adapter="e" * 64)
    reconciles = []
    async def reconcile(identifier, action):
        reconciles.append((identifier, action))
        return {"job_id": native_id, "status": "ready", "deployment_id": ready}
    monkeypatch.setattr(manager.workbench, "get_job_status", reconcile)
    result = await manager.activate_policy(public_id, reconcile=True)
    assert result["status"] == "ready" and reconciles == [(native_id, "reconcile")]
    with pytest.raises(ToolInputError) as error:
        await manager.policy_status(public_id, "cancel")
    assert error.value.code == "activation_already_started"


@pytest.mark.asyncio
async def test_queue_launch_failure_recovers_saved_native_job_without_resubmission(setup, monkeypatch):
    manager, _, _ = setup
    launches = []
    def fail_after_admission(store):
        launches.append(1)
        raise OSError("PRIVATE worker launch failure")
    monkeypatch.setattr("mesocosm.reproduction.jobs.start_worker", fail_after_admission)
    uncertain = manager.train_policy("sft", {}, "request")
    assert uncertain["status"] == "submission_uncertain"
    recovered = await manager.policy_status(uncertain["policy_job_id"])
    assert recovered["status"] == "queued" and launches == [1]
    assert manager.train_policy("sft", {}, "request") == recovered
    assert launches == [1]


def test_public_parameter_input_cannot_coerce_arbitrary_shapes(setup):
    manager, _, _ = setup
    for parameters in ([], "PRIVATE", {"learning_rate": True}, {"lora_rank": "8"}):
        with pytest.raises(ToolInputError) as error:
            manager.train_policy("sft", parameters, "invalid")
        assert error.value.code == "invalid_policy_parameters"


@pytest.mark.asyncio
async def test_same_request_concurrency_never_queues_a_second_optimizer_job(setup):
    manager, _, launches = setup
    results = await asyncio.gather(*(asyncio.to_thread(manager.train_policy, "sft", {}, "same") for _ in range(4)))
    assert len({value["policy_job_id"] for value in results}) == 1 and len(launches) == 1
