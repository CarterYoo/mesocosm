"""Agent-facing contracts: focused discovery, useful errors, no input-value echo."""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from mesocosm.reproduction import behavior
from mesocosm.reproduction.cli import research_app
from mesocosm.reproduction.contracts import describe_tool, validate_arguments
from mesocosm.reproduction.dataset import DatasetConfig
from mesocosm.reproduction.errors import ToolInputError, error_payload
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.training import SFTConfig


@pytest.mark.parametrize("name", ["define_behavior", "configure_experiment", "prepare_dataset",
                                 "submit_training", "get_job_status", "describe_tool", "list_factors",
                                 "read_events", "read_record", "list_records"])
def test_selected_contract_example_validates_without_any_runtime(name):
    contract = describe_tool(ResearchWorkbench, name)
    assert contract["inputSchema"]["additionalProperties"] is False
    assert contract["example_scope"].startswith("Argument shape only")
    assert len(json.dumps(contract).encode()) < 8_000
    parsed = validate_arguments(getattr(ResearchWorkbench, name), contract["example"])
    if name == "define_behavior":
        assert behavior.define_behavior(parsed["spec"])["name"] == "recorded-shell-execution"
    elif name == "prepare_dataset":
        assert DatasetConfig.model_validate(parsed["config"]).seed == 42
    elif name == "submit_training":
        assert SFTConfig.model_validate(parsed["config"]).dtype == "bfloat16"
        assert parsed["start"] is False
    # Detailed contracts stay out of the normal discovery catalog.
    index = ResearchWorkbench.tools()
    assert all("inputSchema" not in entry for entry in index["tools"])


def test_nested_configs_are_actual_models_not_opaque_dicts():
    dataset = describe_tool(ResearchWorkbench, "prepare_dataset")["inputSchema"]
    training = describe_tool(ResearchWorkbench, "submit_training")["inputSchema"]
    assert dataset["$defs"]["DatasetConfig"] == DatasetConfig.model_json_schema()
    assert training["$defs"]["SFTConfig"] == SFTConfig.model_json_schema()
    assert training["$defs"]["SFTConfig"]["required"] == ["base_model", "base_sha256"]
    changes = describe_tool(ResearchWorkbench, "configure_experiment")["inputSchema"]["$defs"]["FactorChange"]
    assert changes["required"] == ["factor", "value"]
    assert changes["properties"]["scope"]["enum"] == ["global", "role", "agent"]


def test_job_action_enum_and_state_guidance_are_discoverable():
    contract = describe_tool(ResearchWorkbench, "get_job_status")
    assert set(contract["inputSchema"]["properties"]["action"]["enum"]) == {
        "inspect", "fetch", "cancel", "start", "resubmit", "reconcile"}
    guidance = " ".join(contract["guidance"])
    assert all(term in guidance for term in ("terminal", "same remote identity", "preflight_failed", "Factorial"))


def test_description_refuses_non_tools_without_echoing_name():
    with pytest.raises(ToolInputError) as caught:
        describe_tool(ResearchWorkbench, "PRIVATE-UNKNOWN-METHOD")
    payload = error_payload("describe_tool", caught.value)
    assert payload["code"] == "unknown_tool"
    assert "PRIVATE-UNKNOWN-METHOD" not in json.dumps(payload)


def test_recursive_behavior_schema_preserves_canonical_rule_semantics():
    event = {"event": {"kind": "action_execution", "where": {"data.returncode": {"gte": 0}}}}
    spec = {"name": "observable", "stages": {"execution": {"all": [
        event, {"any": [event, {"not": {"event": {"kind": "tool_result"}}}]},
        {"sequence": [event, {"event": {"kind": "outcome", "min_count": 1}}]},
    ]}}, "required_stages": ["execution"]}
    result = validate_arguments(ResearchWorkbench.define_behavior, {"spec": spec})
    assert result == {"spec": spec}
    assert behavior.define_behavior(result["spec"]) == behavior.define_behavior(spec)


@pytest.mark.parametrize("spec", [
    {},
    {"name": "x", "stages": {"bogus": {}}},
    {"name": "x", "stages": {"execution": {"event": {"kind": "fiction"}}}},
    {"name": "x", "stages": {"execution": {"event": {"kind": "speech", "min_count": True}}}},
    {"name": "x", "stages": {"execution": {"sequence": [
        {"event": {"kind": "speech", "min_count": 2}}]}}},
    {"name": "x", "stages": {"execution": {"event": {"kind": "speech", "where": {"_private": 1}}}}},
    {"name": "x", "stages": {"execution": {"event": {"kind": "speech"}}}, "required_stages": ["outcome"]},
])
def test_behavior_rejected_before_persistence_with_safe_field_guidance(spec):
    with pytest.raises(ToolInputError) as caught:
        validate_arguments(ResearchWorkbench.define_behavior, {"spec": spec})
    payload = error_payload("define_behavior", caught.value)
    assert payload["code"] == "invalid_arguments"
    assert payload["retryable"] is False
    assert 1 <= len(payload["issues"]) <= 8
    assert all(issue["path"][0] == "spec" for issue in payload["issues"])
    assert "diagnostic_ref" not in payload


def test_nested_private_keys_and_values_are_not_reflected_in_errors():
    secret = "UPSTREAM-TOKEN-SECRET-123"
    with pytest.raises(ToolInputError) as caught:
        validate_arguments(ResearchWorkbench.define_behavior, {"spec": {
            "name": secret, "stages": {secret: {"event": {"kind": secret}}}, secret: secret}})
    payload = error_payload("define_behavior", caught.value)
    assert secret not in json.dumps(payload)
    assert any("<key>" in issue["path"] for issue in payload["issues"])


def test_validation_errors_are_bounded_and_report_omissions():
    with pytest.raises(ToolInputError) as caught:
        validate_arguments(ResearchWorkbench.define_behavior, {f"private-{i}": i for i in range(40)})
    payload = error_payload("define_behavior", caught.value)
    assert len(payload["issues"]) == 8
    assert payload["issues_truncated"] is True
    assert payload["issue_count"] == 41
    assert len(json.dumps(payload).encode()) < 4_000


def test_argument_validation_preserves_omitted_defaults_and_rejects_bool_as_count():
    assert validate_arguments(ResearchWorkbench.get_job_status, {"job_id": "job-" + "0" * 64}) == {
        "job_id": "job-" + "0" * 64}
    with pytest.raises(ToolInputError):
        validate_arguments(ResearchWorkbench.plan_factorial, {
            "baseline_experiment_id": "x", "base_deployment_id": "x", "trained_deployment_id": "x",
            "environment_changes": [], "idempotency_key": "x", "repetitions": True})


@pytest.mark.parametrize("change", [
    {"factor": "agent.model", "value": "x", "scope": "global", "target": "agent-1"},
    {"factor": "agent.model", "value": "x", "scope": "agent"},
    {"factor": "agent.model", "value": "x", "when": "mid_run"},
])
def test_factor_change_schema_catches_scope_and_timing_before_manifest_io(change):
    with pytest.raises(ToolInputError):
        validate_arguments(ResearchWorkbench.configure_experiment, {
            "base_world": "/does/not/exist", "behavior_id": "behavior-" + "0" * 64, "changes": [change]})


def test_actual_training_cross_field_error_is_safe_and_actionable():
    with pytest.raises(ToolInputError) as caught:
        validate_arguments(ResearchWorkbench.submit_training, {
            "dataset_id": "dataset-" + "0" * 64, "idempotency_key": "x", "config": {
                "base_model": "/private/model", "base_sha256": "0" * 64, "dtype": "float16"}})
    payload = error_payload("submit_training", caught.value)
    assert "loss-scaling" in payload["issues"][0]["constraint"]
    assert "/private/model" not in json.dumps(payload)


def test_unknown_runtime_exception_has_no_invented_diagnostic_or_input_echo():
    payload = error_payload("run_experiment", ValueError("upstream secret and private record"))
    assert payload["error_type"] == "ValueError"
    assert payload["code"] == "request_rejected"
    assert payload["retryable"] is False
    assert "secret" not in json.dumps(payload)
    assert "diagnostic_ref" not in payload
    assert "Inspect the private store/job diagnostics" not in payload["message"]


def test_cli_prejob_errors_identify_fields_without_creating_fake_diagnostics(tmp_path):
    runner = CliRunner()
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"spec": {}}))
    root = tmp_path / "store"
    result = runner.invoke(research_app, ["call", "define_behavior", "--request", str(request),
                                        "--root", str(root), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["code"] == "invalid_arguments"
    assert {tuple(issue["path"]) for issue in payload["issues"]} == {("spec", "name"), ("spec", "stages")}
    assert result.stderr == ""
    assert "diagnostic_ref" not in payload
    with sqlite3.connect(root / "research.sqlite3") as db:
        assert all(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
                   for table in ("objects", "requests", "audit", "jobs"))


def test_cli_malformed_json_reports_location_without_source_text(tmp_path):
    request = tmp_path / "bad.json"
    request.write_text('{"private-token": SECRET}')
    result = CliRunner().invoke(research_app, ["call", "define_behavior", "--request", str(request),
        "--root", str(tmp_path / "store"), "--json"])
    payload = json.loads(result.stdout)
    assert payload["code"] == "invalid_json"
    assert payload["line"] == 1
    assert payload["column"] > 0
    assert "SECRET" not in json.dumps(payload)


def test_view_schema_bounds_and_record_kinds_match_implementation():
    from mesocosm.reproduction.views import KINDS
    for name in ("list_factors", "read_record", "read_events", "list_records"):
        properties = describe_tool(ResearchWorkbench, name)["inputSchema"]["properties"]
        assert properties["offset"]["minimum"] == 0
        assert properties["limit"]["minimum"] == 1
        assert properties["limit"]["maximum"] == 100
    properties = describe_tool(ResearchWorkbench, "list_records")["inputSchema"]["properties"]
    assert set(properties["kind"]["enum"]) == KINDS
    pointer = describe_tool(ResearchWorkbench, "read_record")["inputSchema"]["properties"]["pointer"]
    assert pointer["maxLength"] == 4096


@pytest.mark.parametrize("arguments", [
    {"limit": 101}, {"offset": -1}, {"limit": True}, {"kind": "arbitrary"},
])
def test_invalid_views_fail_before_store_access(arguments):
    with pytest.raises(ToolInputError) as caught:
        validate_arguments(ResearchWorkbench.list_records, arguments)
    payload = error_payload("list_records", caught.value)
    assert payload["code"] == "invalid_arguments"
    if arguments == {"limit": 101}:
        assert payload["issues"][0]["bounds"] == {"le": 100}


@pytest.mark.parametrize("pointer", ["missing-slash", "/bad~escape", "/" + "x" * 4096])
def test_invalid_pointers_are_field_errors(pointer):
    with pytest.raises(ToolInputError) as caught:
        validate_arguments(ResearchWorkbench.read_record, {"record_id": "x", "pointer": pointer})
    assert caught.value.issues[0]["path"] == ["pointer"]


def test_since_version_is_an_inspect_only_returned_hash():
    arguments = {"job_id": "job-" + "0" * 64, "since_version": "a" * 64}
    assert validate_arguments(ResearchWorkbench.get_job_status, arguments) == arguments
    for invalid in ({**arguments, "action": "cancel"}, {**arguments, "since_version": "bad"}):
        with pytest.raises(ToolInputError):
            validate_arguments(ResearchWorkbench.get_job_status, invalid)


def test_unknown_tool_name_cannot_flood_or_inject_error_output():
    payload = error_payload("secret\n" + "x" * 100_000, ValueError("private"))
    assert payload["tool"] == "<invalid>"
    assert len(json.dumps(payload)) < 512
