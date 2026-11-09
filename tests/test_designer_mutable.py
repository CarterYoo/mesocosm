"""Mutable-P boundary contracts over native frozen candidates, not behavioral acceptance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from test_reproduction_service import world_file

from mesocosm.learning.config import policy_id
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import digest
from mesocosm.research_mcp.corpus import Corpus
from mesocosm.research_mcp.server import tool_schemas
from mesocosm.research_mcp.service import POLICY_TOOLS, TOOLS, DesignerWorkbench, Settings


@pytest.fixture
def mutable_setup(tmp_path):
    source = world_file(tmp_path / "world")
    raw = yaml.safe_load(source.read_text())
    raw["model"]["id"] = "HIDDEN-base-model"
    for index, agent in enumerate(raw["agents"]):
        agent["id"] = f"hidden-worker-{index}"
    source.write_text(yaml.safe_dump(raw))
    workbench = ResearchWorkbench(tmp_path / "workbench")
    definition = workbench.define_behavior({"name": "HIDDEN-evaluator", "stages": {
        "execution": {"event": {"kind": "action_execution"}}}})
    native = {}
    for handle, endpoint in (("blue", "http://127.0.0.1:8000"), ("red", "http://127.0.0.1:8000"),
                             ("other", "http://127.0.0.1:8001")):
        adapter = "c" * 64 if handle == "red" else None
        identifier = "job-" + digest(handle)
        native[handle] = workbench.store.put("deployment", {
            "policy": {"id": policy_id("/HIDDEN-base", "a" * 40, "b" * 64, adapter),
                       "base_model": "/HIDDEN-base", "base_revision": "a" * 40, "base_sha256": "b" * 64,
                       "adapter_sha256": adapter, "adapter_path": "/HIDDEN-adapter" if adapter else None,
                       "model": "HIDDEN-" + handle, "dtype": "bfloat16"},
            "server": endpoint, "server_token_env": "HIDDEN_TOKEN", "deployment_job_id": identifier,
        })
        workbench.store.save_job(identifier, {"kind": "deployment", "status": "ready",
                                              "deployment_id": native[handle]}, new=True)
    policy_space = tmp_path / "policy-space.json"
    policy_space.write_text(json.dumps({"deployments": native}))
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(json.dumps({"factors": {"services.board": [False, True]}}))
    corpus = Corpus(tmp_path / "corpus")
    settings = Settings(store=str(tmp_path / "designer"), workbench_root=str(workbench.store.root),
                        base_world=str(source), behavior_id=definition["behavior_id"],
                        allowlist=str(allowlist), corpus_root=str(corpus.public_root),
                        session_id="mutable-session", policy_space=str(policy_space))
    return settings, DesignerWorkbench(settings), native


def final_candidate(designer):
    with designer.store.connection() as db:
        submission = json.loads(db.execute("SELECT body FROM objects WHERE kind='designer_submission'").fetchone()[0])
    validation = designer.store.get(submission["validation_id"], "designer_validation")
    return submission, validation, designer.workbench.store.get(validation["configuration"]["experiment_id"],
                                                               "experiment")


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["shared", "mixed"])
async def test_policy_selection_freezes_exact_native_bindings_without_public_identities(mutable_setup, selection):
    settings, designer, native = mutable_setup
    selected = {"policy_id": "red"} if selection == "shared" else {"agent_policies": {
        "agent-1": "red", "agent-2": "blue", "agent-3": "blue"}}
    spec = {"changes": [], **selected}
    checked = await designer.call("check_spec", {"spec": spec})
    assert checked["valid"]
    accepted = await designer.call("submit_spec", {"spec": spec})
    assert accepted == {"accepted": True, "spec_sha256": digest(spec), "immutable": True}
    final, validation, experiment = final_candidate(designer)
    expected = {f"hidden-worker-{index}": native["red" if index == 0 or selection == "shared" else "blue"]
                for index in range(3)}
    assert experiment["agent_deployments"] == expected
    assert validation["policy_bindings"] == {"agent_deployments": expected}
    assert final["context_sha256"] == validation["context_sha256"] == designer.context_sha256
    models = experiment["configuration"]["manifest"]
    assert [agent["model"] for agent in models["agents"]] == [
        "HIDDEN-red", *(["HIDDEN-red"] * 2 if selection == "shared" else ["HIDDEN-blue"] * 2)]
    restarted = DesignerWorkbench(settings)
    assert await restarted.call("submit_spec", {"spec": spec}) == accepted
    public = json.dumps([designer.capabilities(), designer.policy_catalog(), checked, accepted])
    assert "hidden" not in public.lower() and str(Path(settings.policy_space)) not in public
    assert all(identifier not in public for identifier in native.values())


@pytest.mark.asyncio
async def test_unknown_incomplete_incompatible_or_raw_policy_selection_is_rejected(mutable_setup):
    _, designer, native = mutable_setup
    cases = [({"policy_id": native["red"]}, "unknown_policy"),
             ({"agent_policies": {"agent-1": "red"}}, "incomplete_population"),
             ({"agent_policies": {"agent-1": "red", "agent-2": "blue", "agent-3": "other"}},
              "incompatible_population")]
    for selected, expected in cases:
        result = await designer.call("check_spec", {"spec": {"changes": [], **selected}})
        assert result["code"] == expected
        assert "HIDDEN" not in json.dumps(result)
        assert all(identifier not in json.dumps(result) for identifier in native.values())
    both = await designer.call("check_spec", {"spec": {"policy_id": "red", "agent_policies": {"agent-1": "red"}}})
    assert not both["valid"] and both["issues"][0]["code"] == "invalid_shape"
    for factor in ("model.id", "agent.model", "agent.persona", "agent.skills", "agent.mcp"):
        result = await designer.call("check_spec", {"spec": {"policy_id": "red", "changes": [
            {"factor": factor, "value": "HIDDEN-direct-instruction"}]}})
        assert not result["valid"] and result["issues"][0]["code"] == "locked_factor"


@pytest.mark.asyncio
async def test_native_policy_errors_identify_safe_policy_field(mutable_setup, monkeypatch):
    _, designer, _ = mutable_setup

    def reject(*args, **kwargs):
        raise ValueError("HIDDEN-checkpoint /private/path SECRET_TOKEN=secret")

    monkeypatch.setattr(designer.workbench, "configure_experiment", reject)
    result = await designer.call("check_spec", {"spec": {"policy_id": "red"}})
    assert not result["valid"]
    assert result["issues"][0]["field"] == "policy_id"
    assert result["issues"][0]["code"] == "invalid_native_policy_combination"
    assert "HIDDEN" not in json.dumps(result) and "SECRET" not in json.dumps(result)


@pytest.mark.asyncio
async def test_policy_space_bytes_bind_session_during_calls_and_restarts(mutable_setup):
    settings, designer, _ = mutable_setup
    spec = {"changes": [], "policy_id": "blue"}
    accepted = await designer.call("submit_spec", {"spec": spec})
    source = Path(settings.policy_space)
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    for candidate in (designer, DesignerWorkbench(settings)):
        for name, args in (("check_spec", {"spec": spec}), ("submit_spec", {"spec": spec})):
            result = await candidate.call(name, args)
            assert result["code"] == "study_context_changed"
            assert "HIDDEN" not in json.dumps(result) and str(source) not in json.dumps(result)
    source.write_bytes(original)
    assert await DesignerWorkbench(settings).call("submit_spec", {"spec": spec}) == accepted


@pytest.mark.asyncio
async def test_frozen_native_binding_must_equal_selected_policy(mutable_setup, monkeypatch):
    _, designer, native = mutable_setup
    configure = designer.workbench.configure_experiment

    def substitute(*args, **kwargs):
        kwargs["agent_deployments"] = {agent: native["red"] for agent in designer.agent_ids}
        return configure(*args, **kwargs)

    monkeypatch.setattr(designer.workbench, "configure_experiment", substitute)
    result = await designer.call("submit_spec", {"spec": {"policy_id": "blue"}})
    assert result["code"] == "study_context_changed"
    with designer.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='designer_submission'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_policy_resolution_cannot_change_during_validation(mutable_setup, monkeypatch):
    _, designer, native = mutable_setup
    bindings = designer.policy_manager.bindings
    validate = designer.workbench.validate_experiment

    def substitute_after_validation(experiment_id):
        result = validate(experiment_id)
        monkeypatch.setattr(designer.policy_manager, "bindings", lambda *args: {
            "agent_deployments": {agent: native["red"] for agent in designer.agent_ids}})
        return result

    monkeypatch.setattr(designer.workbench, "validate_experiment", substitute_after_validation)
    result = await designer.call("submit_spec", {"spec": {"policy_id": "blue"}})
    assert result["code"] == "study_context_changed"
    with designer.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='designer_submission'").fetchone()[0] == 0
    monkeypatch.setattr(designer.policy_manager, "bindings", bindings)


@pytest.mark.asyncio
async def test_mutable_tool_errors_are_bounded_private_and_every_call_audited(mutable_setup):
    settings, designer, native = mutable_setup
    calls = [("capabilities", {}), ("policy_catalog", {}),
             ("read_record", {"record_id": "/HIDDEN-private-store"})]
    for name, arguments in calls:
        response = await designer.call(name, arguments)
        text = json.dumps(response)
        assert len(text.encode()) <= 24_000
        assert "hidden" not in text.lower()
        assert settings.workbench_root not in text and settings.policy_space not in text
        assert all(identifier not in text for identifier in native.values())
    records = designer.store.audit_records()
    assert len([item for item in records if item["kind"] == "designer.call.started"]) == len(calls)
    assert len([item for item in records if item["kind"] == "designer.call.completed"]) == len(calls)


@pytest.mark.asyncio
async def test_mutable_stdio_has_seven_closed_tools_and_native_policy_validation(mutable_setup, tmp_path):
    pytest.importorskip("mcp")
    from jsonschema import Draft202012Validator
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    settings, _, _ = mutable_setup
    config = tmp_path / "settings.json"
    config.write_text(settings.model_dump_json())
    params = StdioServerParameters(command=sys.executable,
                                  args=["-m", "mesocosm.cli.main", "designer", "serve", "--config", str(config)],
                                  env={**os.environ})
    async with stdio_client(params) as (reader, writer), ClientSession(reader, writer) as session:
        await session.initialize()
        listing = await session.list_tools()
        assert {tool.name for tool in listing.tools} == {"mesocosm_" + name for name in TOOLS + POLICY_TOOLS}
        assert len(listing.model_dump_json().encode()) < 24_000
        for tool in listing.tools:
            Draft202012Validator.check_schema(tool.inputSchema)
            if tool.name.endswith("spec"):
                Draft202012Validator(tool.inputSchema).validate({"spec": {"policy_id": "red", "changes": []}})
        result = await session.call_tool("mesocosm_check_spec", {"spec": {"policy_id": "red"}})
        assert json.loads(result.content[0].text)["valid"]
        assert "HIDDEN" not in result.model_dump_json()
        annotations = {tool.name: tool.annotations.readOnlyHint for tool in listing.tools}
        assert annotations["mesocosm_policy_catalog"]
        assert not any(annotations["mesocosm_" + name] for name in POLICY_TOOLS if name != "policy_catalog")


def test_legacy_schema_stays_closed_without_policy_space(mutable_setup):
    settings, _, _ = mutable_setup
    fixed = DesignerWorkbench(settings.model_copy(update={"policy_space": None, "session_id": "fixed-session"}))
    assert fixed.tools == TOOLS
    assert set(tool_schemas()) == set(TOOLS)
    assert "policy_id" not in tool_schemas()["check_spec"]["properties"]["spec"]["properties"]
    assert fixed.check_spec({"changes": []})["spec_sha256"] == digest({"changes": []})
    assert not fixed.check_spec({"changes": [], "policy_id": "red"})["valid"]
    assert fixed.capabilities()["locked"][:2] == ["model", "population"]
