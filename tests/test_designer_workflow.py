"""Optional designer S and development execution boundaries, not live worker acceptance."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from test_designer_mutable import mutable_setup  # noqa: F401

from mesocosm.reproduction.store import canonical, digest
from mesocosm.research_mcp.server import tool_schemas
from mesocosm.research_mcp.service import EXECUTION_TOOLS, STRUCTURE_TOOLS, DesignerWorkbench


@pytest.fixture
def workflow(mutable_setup, tmp_path):  # noqa: F811
    settings, _, native = mutable_setup
    path = Path(settings.base_world)
    raw = yaml.safe_load(path.read_text())
    raw["services"]["enabled"] = True
    raw["artifacts"] = {"enabled": True, "paths": ["answer.txt"]}
    for agent in raw["agents"]:
        agent["budget"] = {"requests": 10, "completion_tokens": 1000}
    path.write_text(yaml.safe_dump(raw))
    (path.parent / "task/task.toml").write_text("[agent]\ntimeout_sec = 10\n")
    structure_path = tmp_path / "structure-space.json"
    structure_path.write_text(json.dumps({"roles": {"worker": "hidden-worker-0", "reviewer": "hidden-worker-2"},
        "teams": ["alpha"], "max_agents": 6, "allowed_services": ["board", "chat"]}))
    execution_path = tmp_path / "execution-space.json"
    execution_path.write_text(json.dumps({"worker": "http://127.0.0.1:8099", "token_env": "PRIVATE_WORKER_KEY"}))
    settings = settings.model_copy(update={"session_id": "workflow", "structure_space": str(structure_path),
                                           "execution_space": str(execution_path)})
    design = {"population": [{"slot": f"agent-{index}", "role": "reviewer" if index == 4 else "worker",
                              "team": "alpha"} for index in range(1, 5)],
              "visibility": {"services": {"board": {"read": ["agent-4"]}, "chat": {"write": ["agent-4"]}}},
              "work": {f"agent-{index}": ["agent-4"] for index in range(1, 4)},
              "reviews": {"agent-4": ["agent-1", "agent-2", "agent-3"]},
              "execution": {"depends_on": {"agent-4": ["agent-1", "agent-2", "agent-3"]},
                            "finalizers": ["agent-4"]}}
    return settings, DesignerWorkbench(settings), {"changes": [], "structure": design,
            "agent_policies": {f"agent-{index}": "red" if index == 4 else "blue" for index in range(1, 5)}}, native


def test_all_optional_tools_have_valid_closed_schemas(workflow):
    _, designer, spec, _ = workflow
    schemas = tool_schemas(True, True, True)
    assert len(schemas) == len(designer.tools) == 12
    assert set(schemas) == set(designer.tools)
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
    for name in ("check_spec", "submit_spec", "prepare_candidate"):
        Draft202012Validator(schemas[name]).validate({"spec": spec})
    assert set(STRUCTURE_TOOLS + EXECUTION_TOOLS) <= set(schemas)
    assert len(canonical(schemas).encode()) < 32_000


@pytest.mark.asyncio
async def test_composed_population_policy_binding_and_candidate_keep_exact_source_pins(workflow):
    settings, designer, spec, native = workflow
    original = Path(settings.base_world).read_bytes()
    checked = await designer.call("check_spec", {"spec": spec})
    assert checked["valid"], checked
    prepared = await designer.call("prepare_candidate", {"spec": spec})
    assert prepared["validated"] and prepared["grading_scope"]["graded_agent_slots"] == ["agent-4"]
    candidate = designer.store.get(prepared["candidate_id"], "designer_candidate")
    validation = designer.store.get(candidate["validation_id"], "designer_validation")
    experiment = designer.workbench.store.get(candidate["experiment_id"], "experiment")
    assert validation["agent_ids"] == [f"agent-{index}" for index in range(1, 5)]
    assert validation["composition"]["source_sha256"] == designer.base_world_sha256
    assert experiment["source_manifest_sha256"] != designer.base_world_sha256
    assert experiment["agent_deployments"] == {f"agent-{index}": native["red" if index == 4 else "blue"]
                                               for index in range(1, 5)}
    assert experiment["configuration"]["manifest"]["structure"]["execution"] == spec["structure"]["execution"]
    assert Path(settings.base_world).read_bytes() == original
    accepted = await designer.call("submit_spec", {"spec": spec})
    assert accepted["accepted"]
    assert (await designer.call("prepare_candidate", {"spec": spec}))["code"] == "immutable_submission"
    assert (await designer.call("run_candidate", {"candidate_id": prepared["candidate_id"],
                                                "request_id": "after-final"}))["code"] == "immutable_submission"
    public = canonical([checked, prepared, designer.capabilities(), designer.structure_catalog(), accepted])
    assert "hidden-worker" not in public and "PRIVATE" not in public and str(settings.structure_space) not in public
    assert all(identifier not in public for identifier in native.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["structure_space", "execution_space"])
async def test_operator_domain_drift_stops_new_actions(workflow, field):
    settings, designer, spec, _ = workflow
    path = Path(getattr(settings, field))
    path.write_text(path.read_text() + "\n")
    result = await designer.call("prepare_candidate", {"spec": spec})
    assert result["code"] == "study_context_changed"


@pytest.mark.asyncio
async def test_invalid_graph_is_actionable_without_private_values(workflow):
    _, designer, spec, _ = workflow
    invalid = copy.deepcopy(spec)
    invalid["structure"]["execution"]["depends_on"]["agent-1"] = ["agent-4"]
    result = await designer.call("check_spec", {"spec": invalid})
    assert result["valid"] is False
    assert "cycle" in canonical(result) and "hidden-worker" not in canonical(result)
    invalid = copy.deepcopy(spec)
    invalid["agent_policies"].pop("agent-4")
    result = await designer.call("check_spec", {"spec": invalid})
    assert not result["valid"] and result["issues"][0]["code"] == "incomplete_population"


@pytest.mark.asyncio
async def test_execution_rejects_native_ids_and_survives_restart_with_same_handles(workflow):
    settings, designer, spec, _ = workflow
    prepared = await designer.call("prepare_candidate", {"spec": spec})
    assert prepared.get("validated"), prepared
    candidate = designer.store.get(prepared["candidate_id"], "designer_candidate")
    rejected = await designer.call("run_candidate", {"candidate_id": candidate["experiment_id"], "request_id": "raw"})
    assert rejected["code"] == "unknown_development_handle"
    restarted = DesignerWorkbench(settings)
    assert restarted.context_sha256 == designer.context_sha256
    assert restarted.prepare_candidate(spec) == prepared
    assert restarted.check_spec({"changes": [], "structure": None})["spec_sha256"] == digest({"changes": []})
