"""Actual graph compilation and original-source preservation; no live-agent acceptance claim."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml
from pydantic import ValidationError

from mesocosm.core.manifest import WorldManifest
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.store import digest
from mesocosm.research_mcp.structure import StructureInput, StructureSpace


@pytest.fixture
def setup(tmp_path):
    raw = {
        "name": "structure-test",
        "model": {"id": "PRIVATE-model"},
        "gateway": {"url": "http://gateway.invalid", "api_key": "PRIVATE-key"},
        "agents": [
            {
                "id": f"private-{index}",
                "budget": {"requests": 7},
                "tools": {"todowrite": False},
                "skills": ["approved-skill"],
                "persona": {
                    "username": "operator" if index == 1 else f"person-{index}",
                    "brief": "PRIVATE original role instructions",
                },
            }
            for index in range(1, 4)
        ],
        "tasks": [{"path": "original-task"}],
        "services": {"enabled": True, "chat": {"channels": ["general", "team"]}},
        "artifacts": {"enabled": True, "paths": ["answer.txt"]},
        "rewards": {"enabled": True},
    }
    # A valid legacy artifact world becomes a valid graph world only after E disables rewards.
    world = WorldManifest.model_validate(raw)
    path = tmp_path / "world.yaml"
    path.write_text(yaml.safe_dump(raw))
    space = StructureSpace(
        roles={"worker": "private-1", "reviewer": "private-3"},
        teams=["team-a", "team-b"],
        min_agents=3,
        max_agents=6,
        allowed_services=["board", "chat"],
        channels=["general", "team"],
        repositories=["project"],
    )
    space.checked(world)
    design = {
        "population": [
            {"slot": f"agent-{index}", "role": "reviewer" if index == 3 else "worker", "team": "team-a"}
            for index in range(1, 4)
        ],
        "visibility": {
            "services": {"board": {"read": ["agent-3"]}, "chat": {"write": ["agent-3"]}},
            "channels": {"team": {"read": ["team:team-a"], "write": ["agent-3"]}},
        },
        "work": {"agent-1": ["agent-3"], "agent-2": ["agent-3"]},
        "reviews": {"agent-3": ["agent-1", "agent-2"]},
        "execution": {"depends_on": {"agent-3": ["agent-1", "agent-2"]}, "finalizers": ["agent-3"]},
    }
    changes = [{"factor": "rewards.enabled", "value": False}]
    return space, path, design, changes, raw


def test_capabilities_hide_private_templates_and_bound_public_domains(setup):
    space, _, _, _, _ = setup
    encoded = json.dumps(space.capabilities())
    assert "PRIVATE" not in encoded and "private-" not in encoded
    assert len(encoded.encode()) < 8000
    assert "team-specific rewards are unsupported" in encoded
    assert space.capabilities()["population"]["roles"] == ["reviewer", "worker"]
    assert space.capabilities()["automatic_prerequisites"]["isolate_agents"] is True


def test_population_clones_approved_templates_and_compiles_all_three_graphs(setup):
    space, path, raw_design, changes, raw = setup
    original = path.read_bytes()
    design = StructureInput.model_validate(raw_design)
    result = space.compose(path, design, changes)
    manifest = result["manifest"]
    assert result["agent_ids"] == ["agent-1", "agent-2", "agent-3"]
    assert result["roles"] == {
        "worker": ["agent-1", "agent-2"],
        "reviewer": ["agent-3"],
        "team-a": ["agent-1", "agent-2", "agent-3"],
    }
    assert manifest["structure"] == {key: raw_design[key] for key in ("work", "reviews", "execution")}
    assert manifest["services"]["access"]["artifacts"] == raw_design["work"]
    assert manifest["services"]["access"]["channels"]["team"]["read"] == result["agent_ids"]
    assert manifest["services"]["peer_review"]["protocol"] == "graph"
    assert manifest["services"]["chat"]["enabled"] is True and manifest["isolate_agents"] is True
    assert manifest["agents"][0]["persona"]["username"] == "operator"
    assert manifest["agents"][1]["persona"]["username"] == "operator-agent-2"
    for agent in manifest["agents"]:
        assert agent["budget"]["requests"] == 7 and agent["tools"] == {"todowrite": False}
        assert agent["persona"]["brief"] == "PRIVATE original role instructions"
        assert agent["skills"] == [str(path.parent / "approved-skill")]
    assert manifest["model"] == WorldManifest.model_validate(raw).model_dump(mode="json")["model"]
    assert manifest["tasks"][0]["path"] == str(path.parent / "original-task")
    assert manifest["gateway"]["api_key"] == "PRIVATE-key"
    assert path.read_bytes() == original
    assert raw_design["population"] == design.model_dump()["population"]
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["composition_sha256"] == digest(
        {key: value for key, value in result.items() if key != "composition_sha256"}
    )
    assert space.compose(path, design, changes) == result
    assert result["applied_changes"][0] == {"index": 0, "factor": "rewards.enabled", "before": True, "after": False}


def test_no_population_retains_actual_ids_but_public_graphs_are_ordinal(setup):
    space, path, design, changes, _ = setup
    design.pop("population")
    design["visibility"]["channels"]["team"]["read"] = ["agent-1", "agent-2", "agent-3"]
    result = space.compose(path, StructureInput.model_validate(design), changes)
    assert result["agent_ids"] == ["private-1", "private-2", "private-3"]
    assert result["manifest"]["structure"]["reviews"] == {"private-3": ["private-1", "private-2"]}
    assert result["manifest"]["services"]["access"]["channels"]["team"]["read"] == result["agent_ids"]


@pytest.mark.parametrize("field", ["persona", "tools", "model", "budget", "mcp", "skills", "instructions"])
def test_population_has_no_arbitrary_instruction_or_execution_fields(setup, field):
    _, _, design, _, _ = setup
    design["population"][0][field] = "PRIVATE injected instruction"
    with pytest.raises(ValidationError):
        StructureInput.model_validate(design)


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda design: design["population"][0].update(role="unapproved"), "unknown_structure_role"),
        (lambda design: design["population"][0].update(team="unapproved"), "unknown_structure_team"),
        (lambda design: design["population"][0].update(slot="agent-2"), "invalid_population_slots"),
        (lambda design: design["work"].update({"agent-4": ["agent-3"]}), "unknown_structure_slot"),
        (
            lambda design: design["visibility"]["services"].update({"wiki": {"read": ["agent-1"]}}),
            "visibility_outside_domain",
        ),
        (
            lambda design: design["visibility"]["channels"].update({"hidden": {"read": ["agent-1"]}}),
            "visibility_outside_domain",
        ),
        (
            lambda design: design["visibility"].update(git={"hidden": {"write": ["agent-1"]}}),
            "visibility_outside_domain",
        ),
        (
            lambda design: design["visibility"]["channels"]["team"].update(read=["team:team-a", "agent-1"]),
            "overlapping_visibility_selectors",
        ),
        (lambda design: design["visibility"]["channels"]["team"].update(read=["team:team-b"]), "empty_team_selector"),
        (
            lambda design: design["visibility"].update(artifacts={"agent-1": ["agent-2"]}),
            "conflicting_artifact_visibility",
        ),
    ],
)
def test_invalid_public_domains_and_conflicting_acl_are_actionable_without_private_values(setup, mutation, code):
    space, path, design, changes, _ = setup
    mutation(design)
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), changes)
    assert error.value.code == code
    assert "private-" not in str(error.value) and "PRIVATE" not in json.dumps(error.value.issues)


@pytest.mark.parametrize(
    "factor,value",
    [
        ("agent.persona", {"brief": "target behavior"}),
        ("tasks", []),
        ("services.access", {}),
        ("isolate_agents", False),
        ("services.peer_review.protocol", "ratings"),
        ("services.chat.enabled", False),
    ],
)
def test_changes_cannot_override_locked_inputs_or_compiled_enforcement(setup, factor, value):
    space, path, design, changes, _ = setup
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), changes + [{"factor": factor, "value": value}])
    assert error.value.code in {"locked_factor", "structure_managed_factor", "structure_review_chat_required"}


@pytest.mark.parametrize(
    "mutation,phrase",
    [
        (lambda design: design["execution"]["depends_on"].update({"agent-1": ["agent-3"]}), "cycle"),
        (lambda design: design["execution"].update(finalizers=["agent-1"]), "finalizer"),
        (lambda design: design["reviews"].update({"agent-1": ["agent-2"]}), "reviewer"),
        (lambda design: design["visibility"]["services"].pop("chat"), "chat write"),
        (lambda design: design["visibility"]["services"].pop("board"), "board read"),
    ],
)
def test_native_graph_dependencies_are_checked_and_private_ids_are_projected(setup, mutation, phrase):
    space, path, design, changes, _ = setup
    design.pop("population")
    design["visibility"]["channels"]["team"]["read"] = ["agent-3"]
    mutation(design)
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), changes)
    assert error.value.code == "invalid_structure_combination"
    text = json.dumps(error.value.issues)
    assert phrase in text and "private-" not in text and "PRIVATE" not in text and str(path) not in text


def test_unrelated_native_settings_are_not_silently_overridden_to_make_graphs_valid(setup):
    space, path, design, _, _ = setup
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), [])
    assert "reward/feedback/history/reputation" in json.dumps(error.value.issues)


def test_three_agents_and_thirty_two_agents_are_hard_public_limits(setup):
    space, path, design, changes, _ = setup
    for count in (2, 33):
        candidate = copy.deepcopy(design)
        candidate["population"] = [{"slot": f"agent-{index}", "role": "worker"} for index in range(1, count + 1)]
        with pytest.raises(ValidationError):
            StructureInput.model_validate(candidate)
    design["population"].extend({"slot": f"agent-{index}", "role": "worker"} for index in range(4, 8))
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), changes)
    assert error.value.code == "population_outside_domain"


@pytest.mark.parametrize("count", [3, 6, 32])
def test_population_and_native_graphs_support_general_n_agents_with_unique_template_clones(setup, count):
    space, path, design, changes, _ = setup
    space.max_agents = 32
    finalizer = f"agent-{count}"
    workers = [f"agent-{index}" for index in range(1, count)]
    design["population"] = [{"slot": slot, "role": "worker", "team": "team-a"} for slot in workers]
    design["population"].append({"slot": finalizer, "role": "reviewer", "team": "team-a"})
    design["work"] = {slot: [finalizer] for slot in workers}
    design["reviews"] = {finalizer: workers}
    design["execution"] = {"depends_on": {finalizer: workers}, "finalizers": [finalizer]}
    design["visibility"]["services"] = {"board": {"read": [finalizer]}, "chat": {"write": [finalizer]}}
    design["visibility"]["channels"]["team"]["write"] = [finalizer]
    result = space.compose(path, StructureInput.model_validate(design), changes)
    world = WorldManifest.model_validate(result["manifest"])
    assert len(world.agents) == count
    assert len({agent.username for agent in world.agents}) == count
    assert all(len(agent.username) <= 31 for agent in world.agents)
    assert world.structure.waves(result["agent_ids"]) == [workers, [finalizer]]
    assert world.services.access.artifacts == design["work"]


def test_native_diagnostic_slot_projection_does_not_cascade_when_base_ids_overlap_public_slots(setup):
    space, path, design, changes, raw = setup
    for index, agent in enumerate(raw["agents"]):
        agent["id"] = f"agent-{index}"
    path.write_text(yaml.safe_dump(raw))
    space.roles = {"worker": "agent-0", "reviewer": "agent-2"}
    design.pop("population")
    design["visibility"]["channels"] = {}
    design["reviews"] = {"agent-2": ["agent-1"]}
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), changes)
    assert "reviewer agent-2 must execute" in json.dumps(error.value.issues)


@pytest.mark.parametrize("field", ["resource", "selector", "sources"])
def test_public_visibility_collections_and_strings_are_bounded(setup, field):
    _, _, design, _, _ = setup
    if field == "resource":
        design["visibility"]["channels"] = {"x" * 65: {"read": ["agent-1"]}}
    elif field == "selector":
        design["visibility"]["channels"]["team"]["read"] = ["x" * 54]
    else:
        design["visibility"]["direct_messages"] = {f"source-{index}": [] for index in range(33)}
    with pytest.raises(ValidationError):
        StructureInput.model_validate(design)


def test_invalid_native_artifact_settings_offer_safe_concrete_repair(setup):
    space, path, design, changes, _ = setup
    with pytest.raises(ToolInputError) as error:
        space.compose(
            path, StructureInput.model_validate(design), changes + [{"factor": "artifacts.paths", "value": []}]
        )
    assert "Enable approved declared paths together" in str(error.value)
    assert "PRIVATE" not in str(error.value)


def test_side_task_recipient_rewriting_is_not_a_graph_compatibility_workaround(setup):
    space, path, design, changes, raw = setup
    raw["profile"] = "fixture"
    raw["tasks"][0]["side_task"] = ["private-1"]
    path.write_text(yaml.safe_dump(raw))
    original = path.read_bytes()
    with pytest.raises(ToolInputError) as error:
        space.compose(path, StructureInput.model_validate(design), changes)
    assert error.value.code == "unsupported_structure_task"
    assert "side-task verification" in str(error.value)
    assert path.read_bytes() == original
