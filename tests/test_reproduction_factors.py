"""Pure configuration checks; these fixtures are not native acceptance evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mesocosm.reproduction.factors import configure_experiment, list_factors, validate_experiment


def world() -> dict:
    return {"name": "factor-test", "profile": "research", "model": {"id": "model-a"},
            "models": [{"id": "model-b"}], "gateway": {"url": "http://gateway:8080"},
            "agents": [{"id": f"agent-{i}"} for i in range(3)],
            "tasks": [{"dataset": "declared/source"}]}


def change(factor: str, value: object, **kwargs: object) -> dict:
    return {"factor": factor, "value": value, **kwargs}


def test_catalog_types_defaults_current_and_enforcement_boundaries():
    entries = {item["name"]: item for item in list_factors(world())}
    budget = entries["agent.budget.requests"]
    assert budget["default"] is None
    assert budget["current"] == {f"agent-{i}": None for i in range(3)}
    assert budget["schema"]["anyOf"][0] == {"minimum": 1, "type": "integer"}
    assert budget["change_timing"] == ["before_run", "mid_run"]
    assert entries["services.access"]["change_timing"] == ["before_run"]
    assert entries["model.adapter"]["status"] == "unsupported"
    assert entries["agent.model"]["effective_current"] == {f"agent-{i}": "model-a" for i in range(3)}
    assert all(item.get("native_acceptance") != "supported" for item in entries.values())
    assert "agent.id" not in entries and "gateway.api_key" not in entries
    root = Path(__file__).parents[1]
    for item in entries.values():
        for evidence in item["evidence"]:
            assert (root / evidence).is_file(), (item["name"], evidence)
        if item.get("enforcement"):
            assert (root / item["enforcement"]).is_file()
    json.dumps(list_factors())


def test_ordered_global_role_agent_resolution_is_explicit_and_does_not_mutate():
    base = world()
    original = copy.deepcopy(base)
    result = configure_experiment(base, [
        change("agent.budget.requests", 12),
        change("agent.budget.requests", 7, scope="role", target="reviewers"),
        change("agent.budget.requests", 3, scope="agent", target="agent-2"),
    ], {"reviewers": ["agent-1", "agent-2"]})
    assert base == original
    assert [agent["budget"]["requests"] for agent in result["manifest"]["agents"]] == [12, 7, 3]
    receipts = result["effective_changes"]
    assert [len(receipt["applied"]) for receipt in receipts] == [3, 2, 1]
    assert receipts[2]["applied"] == [{"agent_id": "agent-2", "before": 7, "after": 3}]
    assert result == configure_experiment(base, [
        change("agent.budget.requests", 12),
        change("agent.budget.requests", 7, scope="role", target="reviewers"),
        change("agent.budget.requests", 3, scope="agent", target="agent-2"),
    ], {"reviewers": ["agent-1", "agent-2"]})


def test_hashes_describe_effective_configuration_and_requested_order():
    base = world()
    a = configure_experiment(base, [change("agent.budget.requests", 4)])
    b = configure_experiment(base, [change("agent.budget.requests", 5), change("agent.budget.requests", 4)])
    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert a["configuration_sha256"] != b["configuration_sha256"]
    assert a["base_manifest_sha256"] != a["manifest_sha256"]


@pytest.mark.parametrize("roles", [
    {"team": ["ghost"]}, {"team": ["agent-0", "agent-0"]}, {"": ["agent-0"]},
    {"empty": []}, {"team": "agent-0"}, {"team": [True]}, [],
])
def test_bad_roles_fail(roles):
    with pytest.raises(ValueError, match="role"):
        configure_experiment(world(), [], roles)


@pytest.mark.parametrize("requested_change", [
    change("agent.id", "other", scope="agent", target="agent-0"),
    change("name", "new-name"), change("model.reasoning", True),
    change("memory", "carry", scope="role", target="reviewers"),
    change("agent.budget.requests", 1, target="agent-0"),
    change("agent.budget.requests", 1, scope="role", target="unknown"),
    change("agent.budget.requests", 1, scope="agent", target="unknown"),
    change("agent.budget.requests", 1, when="mid_run"),
    change("agent.budget.requests", 1, unexpected=True),
    {"factor": "memory"},
])
def test_unsupported_or_ambiguous_requests_fail(requested_change):
    with pytest.raises(ValueError):
        configure_experiment(world(), [requested_change], {"reviewers": ["agent-0"]})


@pytest.mark.parametrize("value", [True, "9", 0, -1, 3.5])
def test_integer_budget_is_strict_and_bounded(value):
    with pytest.raises(ValueError):
        configure_experiment(world(), [change("agent.budget.requests", value)])


def test_multiple_changes_validated_together_not_half_applied():
    result = configure_experiment(world(), [change("services.chat.max_chars", 200),
                                           change("services.chat.enabled", True)])
    assert result["manifest"]["services"]["chat"]["max_chars"] == 200
    with pytest.raises(ValueError, match="enabled"):
        configure_experiment(world(), [change("services.chat.max_chars", 200)])
    with pytest.raises(ValueError, match="feedback"):
        configure_experiment(world(), [change("learning.mode", "in_context")])


def test_per_role_route_changes_preserve_real_agent_identity():
    result = configure_experiment(world(), [change("agent.model", "model-b", scope="role", target="review")],
                                  {"review": ["agent-1"]})
    assert [a["id"] for a in result["manifest"]["agents"]] == ["agent-0", "agent-1", "agent-2"]
    assert result["manifest"]["agents"][1]["model"] == "model-b"
    with pytest.raises(ValueError, match="unknown model"):
        configure_experiment(world(), [change("agent.model", "ghost")])


def test_sampling_defaults_types_and_conflicts():
    request = [change("model.sampling", {"temperature": 0.7, "top_p": 0.95, "seed": 42})]
    result = configure_experiment(world(), request)
    assert result["manifest"]["model"]["sampling"]["top_k"] == -1
    assert result["effective_changes"][0]["applied"][0]["after"]["min_p"] == 0.0
    base = world()
    base["model"]["kind"] = "openai"
    with pytest.raises(ValueError, match="vllm"):
        configure_experiment(base, request)
    with pytest.raises(ValueError):
        configure_experiment(world(), [change("model.sampling", {"temperature": float("nan"), "top_p": 1, "seed": 0})])


def test_validation_preserves_relative_source_paths_and_removes_temporary_input(tmp_path):
    base = world()
    base["tasks"] = [{"path": "missing-task"}]
    report = validate_experiment(base, tmp_path)
    assert not report["valid"]
    assert str(tmp_path / "missing-task") in " ".join(report["errors"])
    assert report["capability_boundaries"]["native_worker_acceptance"] == "unverified"
    assert not list(tmp_path.glob(".mesocosm-validate-*"))


def test_validation_rejects_unknown_model_and_inactive_treatment(tmp_path):
    base = world()
    base["agents"][0]["model"] = "absent"
    base["services"] = {"chat": {"max_chars": 4}}
    result = validate_experiment(base, tmp_path)
    assert not result["valid"]
    assert any("unknown model" in item for item in result["errors"])
    assert any("enabled" in item for item in result["errors"])


def test_validation_reports_schema_errors_and_missing_base_directory(tmp_path):
    assert not validate_experiment({"unknown": 3}, tmp_path)["valid"]
    assert not validate_experiment(world(), tmp_path / "missing")["valid"]


def test_requirement_audit_retains_all_ids_exact_text_and_honest_boundaries():
    audit = json.loads((Path(__file__).parents[1] / "docs/reproduction-requirements.json").read_text())
    rows = audit["requirements"]
    assert audit["source"]["checklist_count"] == 185
    assert [row["id"] for row in rows] == [f"R{i:03d}" for i in range(1, 186)]
    assert len({row["text"] for row in rows}) == 185
    assert all(row["text"] and row["boundary"] for row in rows)
    assert all(row["status"] in {"supported", "partial", "unsupported", "unverified"} for row in rows)
    assert all(row["native_acceptance"] != "supported" for row in rows)
    assert rows[139]["status"] == "unsupported"  # Exact live rewind is not a file snapshot.
    assert rows[158]["status"] == "unsupported"  # Activation recording is not steering.
    assert rows[45]["status"] == "unsupported"  # Static agent list is not runtime cloning.
