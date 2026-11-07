"""Designer boundary software contracts, never a substitute for isolated real-agent acceptance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from test_reproduction_service import world_file
from typer.testing import CliRunner

from mesocosm.cli.main import app
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import digest
from mesocosm.research_mcp.corpus import Corpus
from mesocosm.research_mcp.service import MAX_RESPONSE_BYTES, DesignerWorkbench, Settings
from mesocosm.research_mcp.spec import SUPPORTED_FACTORS, Allowlist


@pytest.fixture
def setup(tmp_path):
    world = world_file(tmp_path / "source")
    raw = yaml.safe_load(world.read_text())
    raw["model"]["id"] = "HIDDEN-model-checkpoint"
    raw["agents"][0]["persona"] = {"username": "private", "name": "HIDDEN-persona"}
    world.write_text(yaml.safe_dump(raw))
    bench = ResearchWorkbench(tmp_path / "workbench")
    behavior = bench.define_behavior({"name": "HIDDEN-rubric", "stages": {"onset": {"event": {"kind": "speech"}}}})
    allow = tmp_path / "allowlist.json"
    allow.write_text(json.dumps({"factors": {"services.board": [False, True],
                                             "services.chat.max_chars": [100],
                                             "task_services": ["isolated", "shared"]}}))
    teacher = tmp_path / "teacher.json"
    teacher.write_text(json.dumps({"manifest": raw, "events": [
        {"kind": "speech", "agent_id": "HIDDEN-frontier-model", "text": "We should review the result."}]}))
    corpus = Corpus(tmp_path / "corpus")
    imported = corpus.import_source(teacher, policy={"source_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest(),
                                                   "reviewer": "operator", "events": {"/events/0": ["/text"]}})
    settings = Settings(store=str(tmp_path / "designer"), workbench_root=str(bench.store.root),
                        base_world=str(world), behavior_id=behavior["behavior_id"], allowlist=str(allow),
                        corpus_root=str(corpus.public_root), corpus_ids=[imported["corpus_id"]], session_id="session-a")
    return settings, DesignerWorkbench(settings)


@pytest.mark.asyncio
async def test_locked_factors_rejected_with_actionable_closed_domains(setup):
    _, workbench = setup
    for factor in ("model.id", "agent.model", "agent.persona", "agent.skills", "agent.mcp", "tasks",
                   "services.wiki.docs", "internet.enabled", "judge.monitor.rubric", "observe.activations",
                   "agent.budget.requests", "resources.model_requests", "services.cache.seed"):
        result = await workbench.call("check_spec", {"spec": {"changes": [{"factor": factor, "value": "secret"}]}})
        assert result["valid"] is False
        issue = result["issues"][0]
        assert issue["field"] == "changes.0.factor" and issue["code"] == "locked_factor"
        assert issue["allowed_factors"] == ["services.board", "services.chat.max_chars", "task_services"]
    invalid = workbench.check_spec({"changes": [{"factor": "task_services", "value": "secret"}]})
    assert invalid["issues"][0]["allowed_values"] == ["isolated", "shared"]
    assert not workbench.check_spec({"changes": [], "persona": "secret"})["valid"]


def test_all_allowlisted_supported_fields_are_finite_and_budgets_locked():
    assert "history.max_chars" in SUPPORTED_FACTORS
    assert "artifacts.paths" in SUPPORTED_FACTORS
    for field in ("agent.budget.requests", "resources.model_requests", "services.cache.seed", "agent.persona"):
        with pytest.raises(ValueError):
            Allowlist(factors={field: [1]}).checked()
    with pytest.raises(ValueError):
        Allowlist(factors={"services.board": ["yes"]}).checked()
    with pytest.raises(ValueError):
        Allowlist(factors={"services.board": []}).checked()


@pytest.mark.asyncio
async def test_real_configuration_validation_and_private_error_projection(setup):
    _, designer = setup
    result = await designer.call("check_spec", {"spec": {"changes": []}})
    assert result["valid"] is True
    assert result["validation_scope"] == "configure_experiment_and_validate_experiment"
    with designer.store.connection() as db:
        saved = db.execute("SELECT body FROM objects WHERE kind='designer_validation'").fetchone()
    private = json.loads(saved[0])
    assert designer.workbench.store.get(private["configuration"]["experiment_id"], "experiment")
    assert private["validation"]["valid"]
    # The base has disabled chat. This individual value passes the allowlist,
    # but the actual native dependency validation rejects it.
    invalid = await designer.call("check_spec", {"spec": {"changes": [{"factor": "services.chat.max_chars", "value": 100}]}})
    assert invalid["valid"] is False
    assert invalid["issues"][0]["code"] == "invalid_native_combination"
    assert invalid["issues"][0]["field"] == "changes.0.value"
    assert invalid["issues"][0]["factor"] == "services.chat.max_chars"
    assert invalid["issues"][0]["allowed_values"] == [100]
    assert "enabled chat" in invalid["issues"][0]["dependency_hints"][0]
    assert invalid["issues"][0]["related_allowed_factors"] == []
    assert "remove this change" in invalid["issues"][0]["message"]
    assert "HIDDEN" not in json.dumps(invalid)


@pytest.mark.asyncio
async def test_static_chat_dependency_hint_allows_rechecking_approved_combination(setup):
    settings, designer = setup
    choices = {"factors": {"services.chat.max_chars": [200, 1000, 2000],
                           "services.chat.enabled": [False, True], "services.chat.channels": [[], ["general"]]}}
    Path(settings.allowlist).write_text(json.dumps(choices))
    source = Path(settings.base_world)
    raw = yaml.safe_load(source.read_text())
    raw["services"]["enabled"] = True
    source.write_text(yaml.safe_dump(raw))
    designer = DesignerWorkbench(settings.model_copy(update={"session_id": "chat-dependency-test"}))
    capabilities_before = designer.capabilities()
    change = {"factor": "services.chat.max_chars", "value": 2000}
    rejected = await designer.call("check_spec", {"spec": {"changes": [change]}})
    assert not rejected["valid"]
    issue = rejected["issues"][0]
    assert issue["factor"] == "services.chat.max_chars" and issue["allowed_values"] == [200, 1000, 2000]
    assert issue["related_allowed_factors"] == ["services.chat.enabled"]
    assert issue["hint_scope"] == "general_contract; does_not_report_current_world_settings"
    accepted = await designer.call("check_spec", {"spec": {"changes": [
        change, {"factor": "services.chat.enabled", "value": True}]}})
    assert accepted["valid"]
    assert designer.capabilities() == capabilities_before
    assert "HIDDEN" not in json.dumps(rejected | {"capabilities": capabilities_before})
    assert settings.base_world not in json.dumps(rejected)


@pytest.mark.parametrize("factor", ["services.chat.channels", "services.chat.max_chars", "services.chat.max_messages"])
def test_chat_hints_publish_only_related_approved_factor_names(factor):
    allowlist = Allowlist(factors={factor: [[]] if factor.endswith("channels") else [100],
                                  "services.chat.enabled": [False, True]}).checked()
    hint = allowlist.dependency_hints(factor)
    assert hint["related_allowed_factors"] == ["services.chat.enabled"]
    assert "enabled chat" in hint["dependency_hints"][0]
    assert "current" not in hint


@pytest.mark.asyncio
async def test_native_failure_hints_never_reflect_exception_or_locked_values(setup, monkeypatch):
    settings, designer = setup

    def rejected(*args, **kwargs):
        raise ValueError("HIDDEN-model HIDDEN-population HIDDEN-rubric /outside/public/corpus HIDDEN-budget=9")

    monkeypatch.setattr(designer.workbench, "configure_experiment", rejected)
    result = await designer.call("check_spec", {"spec": {"changes": [
        {"factor": "services.chat.max_chars", "value": 100}]}})
    text = json.dumps(result)
    assert "HIDDEN" not in text and "/outside/public/corpus" not in text and settings.base_world not in text
    assert result["issues"][0]["allowed_values"] == [100]
    assert "enabled chat" in result["issues"][0]["dependency_hints"][0]


@pytest.mark.asyncio
async def test_submission_idempotent_immutable_across_restarts_and_concurrency(setup):
    settings, designer = setup
    spec = {"changes": []}
    results = await asyncio.gather(*(designer.call("submit_spec", {"spec": spec}) for _ in range(4)))
    assert all(result == results[0] for result in results)
    assert results[0] == {"accepted": True, "spec_sha256": digest(spec), "immutable": True}
    restarted = DesignerWorkbench(settings)
    assert await restarted.call("submit_spec", {"spec": spec}) == results[0]
    changed = await restarted.call("submit_spec", {"spec": {"changes": [{"factor": "services.board", "value": False}]}})
    assert changed["code"] == "immutable_submission"
    with designer.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='designer_submission'").fetchone()[0] == 1
        final = json.loads(db.execute("SELECT body FROM objects WHERE kind='designer_submission'").fetchone()[0])
    validation = designer.store.get(final["validation_id"], "designer_validation")
    assert validation["validation"]["valid"]
    experiment_id = validation["configuration"]["experiment_id"]
    assert validation["validation"]["experiment_id"] == experiment_id
    experiment = designer.workbench.store.get(experiment_id, "experiment")
    assert experiment["source_manifest_sha256"] == designer.base_world_sha256
    assert final["context_sha256"] == validation["context_sha256"] == designer.context_sha256
    assert set(results[0]) == {"accepted", "spec_sha256", "immutable"}


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_file", ["base_world", "allowlist"])
async def test_setup_drift_rejected_during_session_and_after_restart_preserving_final(setup, changed_file):
    settings, designer = setup
    spec = {"changes": []}
    final_result = await designer.call("submit_spec", {"spec": spec})
    assert final_result["accepted"]
    with designer.store.connection() as db:
        before = db.execute("SELECT id,body FROM objects WHERE kind='designer_submission'").fetchall()
    target = Path(getattr(settings, changed_file))
    original = target.read_bytes()
    # Actual bytes are bound, including changes at the same filesystem path.
    target.write_bytes(original + b"\n")
    for candidate in (designer, DesignerWorkbench(settings)):
        for tool in ("check_spec", "submit_spec"):
            rejected = await candidate.call(tool, {"spec": spec})
            assert rejected["ok"] is False and rejected["code"] == "study_context_changed"
            assert "HIDDEN" not in json.dumps(rejected) and str(target) not in json.dumps(rejected)
    with designer.store.connection() as db:
        assert db.execute("SELECT id,body FROM objects WHERE kind='designer_submission'").fetchall() == before
    target.write_bytes(original)
    restored = DesignerWorkbench(settings)
    assert await restored.call("submit_spec", {"spec": spec}) == final_result


@pytest.mark.asyncio
async def test_same_session_restart_cannot_change_fixed_model_before_final_submission(setup):
    settings, designer = setup
    source = Path(settings.base_world)
    raw = yaml.safe_load(source.read_text())
    raw["model"]["id"] = "HIDDEN-different-student-policy"
    source.write_text(yaml.safe_dump(raw))
    for candidate in (designer, DesignerWorkbench(settings)):
        result = await candidate.call("check_spec", {"spec": {"changes": []}})
        assert result["code"] == "study_context_changed"
    with designer.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='designer_submission'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_context_change_during_native_validation_cannot_freeze_final(setup, monkeypatch):
    settings, designer = setup
    original_validate = designer.workbench.validate_experiment

    def mutate_during_validation(experiment_id):
        result = original_validate(experiment_id)
        source = Path(settings.base_world)
        source.write_bytes(source.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(designer.workbench, "validate_experiment", mutate_during_validation)
    result = await designer.call("submit_spec", {"spec": {"changes": []}})
    assert result["code"] == "study_context_changed"
    with designer.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='designer_submission'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_frozen_candidate_must_match_original_base_even_if_source_is_restored(setup, monkeypatch):
    settings, designer = setup
    original_configure = designer.workbench.configure_experiment
    source = Path(settings.base_world)
    original_bytes = source.read_bytes()

    def changed_candidate(*args, **kwargs):
        raw = yaml.safe_load(original_bytes)
        raw["model"]["id"] = "HIDDEN-temporarily-replaced-policy"
        source.write_text(yaml.safe_dump(raw))
        try:
            return original_configure(*args, **kwargs)
        finally:
            source.write_bytes(original_bytes)

    monkeypatch.setattr(designer.workbench, "configure_experiment", changed_candidate)
    result = await designer.call("submit_spec", {"spec": {"changes": []}})
    assert result["code"] == "study_context_changed"
    assert "HIDDEN" not in json.dumps(result)
    with designer.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM objects WHERE kind='designer_submission'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_all_tool_responses_are_public_bounded_and_calls_audited(setup):
    settings, designer = setup
    identifier = settings.corpus_ids[0]
    calls = [("capabilities", {}), ("corpus_search", {"corpus_id": identifier}),
             ("corpus_stats", {"corpus_id": identifier}), ("read_events", {"corpus_id": identifier}),
             ("check_spec", {"spec": {"changes": []}}), ("submit_spec", {"spec": {"changes": []}}),
             ("read_record", {"record_id": "/operator/secret"})]
    for name, args in calls:
        response = await designer.call(name, args)
        text = json.dumps(response)
        assert len(text.encode()) <= MAX_RESPONSE_BYTES
        assert "HIDDEN" not in text and settings.base_world not in text
        assert settings.workbench_root not in text and settings.behavior_id not in text
        assert "source_path" not in text and "source_pointer" not in text
    audit = designer.store.audit_records()
    starts = [row for row in audit if row["kind"] == "designer.call.started"]
    finishes = [row for row in audit if row["kind"] == "designer.call.completed"]
    assert len(starts) == len(finishes) == len(calls)
    assert starts[-1]["data"]["arguments"]["record_id"] == "/operator/secret"
    assert all(row["data"]["result_sha256"] and row["at"] for row in finishes)


@pytest.mark.asyncio
async def test_actual_stdio_mcp_discovery_and_calls(setup, tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    settings, _ = setup
    config = tmp_path / "settings.json"
    config.write_text(settings.model_dump_json())
    params = StdioServerParameters(command=sys.executable,
                                  args=["-m", "mesocosm.cli.main", "designer", "serve", "--config", str(config)],
                                  env={**os.environ})
    async with stdio_client(params) as (reader, writer), ClientSession(reader, writer) as session:
        await session.initialize()
        listed = await session.list_tools()
        assert {item.name for item in listed.tools} == {"mesocosm_" + name for name in (
            "capabilities", "corpus_search", "corpus_stats", "read_events", "check_spec", "submit_spec")}
        assert len(json.dumps(listed.model_dump()).encode()) < MAX_RESPONSE_BYTES
        from jsonschema import Draft202012Validator
        for tool in listed.tools:
            Draft202012Validator.check_schema(tool.inputSchema)
            if tool.name.endswith("spec"):
                Draft202012Validator(tool.inputSchema).validate({"spec": {"changes": [{
                    "factor": "services.board", "value": False}]}})
        response = await session.call_tool("mesocosm_check_spec", {"spec": {"changes": []}})
        assert json.loads(response.content[0].text)["valid"] is True
        assert "HIDDEN" not in response.model_dump_json()


def test_cli_mask_requires_policy_and_help_is_available(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["designer", "--help"]).exit_code == 0
    result = runner.invoke(app, ["designer", "mask-corpus", str(tmp_path / "absent"), "--store", str(tmp_path / "corpus"), "--json"])
    assert result.exit_code == 1
    assert "source-hash-bound" in json.loads(result.stdout)["message"]
