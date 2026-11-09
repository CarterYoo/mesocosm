"""Restricted policy lifecycle contracts; simulated lifecycle states are not agent/learning acceptance."""

from __future__ import annotations

import json

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
def setup(tmp_path):
    bench = ResearchWorkbench(tmp_path / "native")
    settings = {"deployments": {"base": deployment(bench, "base"),
                                 "adapter": deployment(bench, "adapter", adapter="c" * 64),
                                 "other": deployment(bench, "other", base="different-base")}}
    space = PolicySpace.model_validate(settings)
    bench.allowed_servers = space.allowed_servers(bench)
    space.checked(bench)
    store = Store(tmp_path / "designer")
    manager = PolicyManager(space, store, bench, "session-a", "d" * 64)
    return manager


def test_catalog_opaque_paginated_and_bindings_native_exhaustive(setup):
    manager = setup
    first = manager.catalog(limit=2)
    second = manager.catalog(offset=first["next_offset"], limit=2)
    text = json.dumps([first, second])
    assert "PRIVATE" not in text and "https://" not in text and "dataset-" not in text
    assert first["next_offset"] == 2 and second["next_offset"] is None
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
