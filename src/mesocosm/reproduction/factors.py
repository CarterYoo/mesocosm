"""Schema-backed factors over existing enforcement points, never prompt substitutes.

Configuration is a pure, ordered transformation. Runtime admission and native worker
acceptance remain separate; a listed factor is not evidence of a successful experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
from typing import Any, get_args

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from mesocosm.core.condition_validation import inactive_condition_errors
from mesocosm.core.episode_validation import episode_condition_errors
from mesocosm.core.manifest import AgentSpec, WorldManifest, validate_manifest

# Deliberate allowlist: schema presence alone is not an enforcement capability.
# Each entry is (field names, enforcement source, evidence, limitation/dependency).
_GROUPS = [
    ("agent.budget.completion_tokens agent.budget.prompt_tokens agent.budget.requests",
     "src/mesocosm/gateway/app.py", "tests/test_conditions.py",
     "Per-agent admission at the model gateway; running requests are not undone."),
    ("agent.model agent.tools agent.persona agent.mcp agent.skills",
     "src/mesocosm/env/compose.py", "tests/test_compose_cli.py",
     "OpenCode configuration before startup; roles are configuration groups, not permission grants."),
    ("model.context_limit model.output_limit models",
     "src/mesocosm/env/compose.py", "tests/test_manifest.py",
     "Declared model limits configure the harness; upstream availability and capabilities need live validation."),
    ("model.hf_repo model.revision",
     "src/mesocosm/core/manifest.py", "tests/test_manifest.py",
     "Provenance declarations only: these fields do not load weights or prove the served revision."),
    ("model.sampling model.sampling.temperature model.sampling.top_p model.sampling.seed "
     "model.sampling.top_k model.sampling.min_p",
     "src/mesocosm/gateway/sampling.py", "tests/test_model_sampling.py",
     "vLLM routes only; complete sampling specification required; incompatible with rollout-bound training. "
     "Seed is fixed per request, not an environment or deterministic-replay seed."),
    ("memory history.episodes history.scope history.max_chars",
     "src/mesocosm/core/history.py", "tests/test_history.py",
     "Memory is global at task boundaries. Bounded recorded history requires comparison/reset; "
     "no arbitrary harness-state serialization or semantic retrieval."),
    ("mode task_services task_start tasks",
     "src/mesocosm/core/run.py", "tests/test_experiment_conditions.py",
     "Original task/environment/verifier contract is preserved; installed backend support must validate. "
     "Changing task references is not modifying hidden answers."),
    ("resources.model_requests resources.completion_tokens resources.model_concurrency resources.service_requests",
     "src/mesocosm/gateway/app.py", "tests/test_team_resources.py",
     "Model quotas are team-wide gateway admission; service quotas require enabled services. "
     "These are not CPU/GPU allocation or storage budgets."),
    ("services.enabled services.board services.chat.enabled services.chat.channels services.chat.max_chars "
     "services.chat.max_messages services.git services.wiki.docs services.mcp services.cache.enabled "
     "services.cache.writable services.cache.list_recent channels isolate_agents",
     "src/mesocosm/env/compose.py", "tests/test_research_conditions.py",
     "Declared actual services and mounts at startup. Chat limits require enabled chat. "
     "Shared text cache is not a build cache; no scripted messages or invented history."),
    ("services.access",
     "src/mesocosm/core/access.py", "tests/test_access.py",
     "Requires enabled services and isolated agent networks; shared_dir is incompatible. "
     "Service ACLs do not prevent forwarding by an authorized reader."),
    ("internet.enabled internet.allow internet.deny internet.mode internet.replay_file "
     "internet.budget.requests internet.budget.bytes",
     "src/mesocosm/env/web.py", "tests/test_web.py",
     "Only configured web-gateway access. Replay reuses recorded HTTP flows, not model or whole-world state."),
    ("rewards.enabled rewards.source rewards.terms rewards.success_threshold rewards.allocation rewards.team_weight "
     "learning.mode learning.feedback.fields learning.feedback.delay_episodes learning.feedback.max_chars",
     "src/mesocosm/core/rewards.py", "tests/test_rewards.py",
     "Measured comparison-episode rewards; learning.mode is context feedback, not weight training. "
     "Component metrics and peer review dependencies are validated."),
    ("services.peer_review.enabled services.peer_review.protocol services.peer_review.topology "
     "services.peer_review.matching_seed reputation.enabled reputation.window reputation.metrics "
     "task_assignment.policy task_assignment.participants task_assignment.score task_assignment.seed",
     "src/mesocosm/core/episode_controls.py", "tests/test_research_conditions.py",
     "Research peer review uses ratings and actual measured outcomes. Selective assignment is task allocation, "
     "not a general message scheduler. Cross-field dependencies apply."),
    ("artifacts.enabled artifacts.paths artifacts.eligibility artifacts.success_threshold artifacts.show_author "
     "artifacts.show_score artifacts.max_bytes",
     "src/mesocosm/core/artifacts.py", "tests/test_artifact_exchange.py",
     "Only declared actual UTF-8 outputs after isolated verification; needs board and measured rewards."),
    ("judge.monitor.enabled judge.monitor.scope judge.monitor.visibility judge.monitor.timing "
     "judge.monitor.decision_timeout_seconds judge.monitor.decision_delay_seconds judge.monitor.poll_seconds "
     "judge.monitor.on_flag judge.monitor.include_shared judge.monitor.rubric judge.monitor.rubric_version "
     "judge.monitor.max_input_chars judge.monitor.max_calls judge.monitor.output_tokens judge.monitor.model "
     "judge.monitor.threshold",
     "src/mesocosm/judge/monitor.py", "tests/test_monitor.py",
     "Catalog model required. Pre-action supports allow/deny/terminate at OpenCode's before-tool boundary, "
     "not syscall supervision or rollback; posthoc settings cannot claim live enforcement."),
    ("observe.activations observe.token_logprobs observe.top_logprobs observe.capture.layers "
     "observe.capture.raw observe.capture.positions observe.capture.probes",
     "src/mesocosm/observe/activations.py", "tests/test_activations.py",
     "Opt-in token-ID/logprob recording and separate activation recomputation; no live steering. "
     "Closed routes cannot expose activations. Exact serving/replay alignment remains separately measured."),
]

_UNSUPPORTED = {
    "agent.lifecycle": "Dynamic creation, cloning, delegation and arbitrary process restoration are not implemented.",
    "message.transform": "Message paraphrase, deletion, sender substitution and arbitrary delivery interception "
                         "are not implemented.",
    "scheduler.protocol": "General priority scheduling, voting and virtual-time protocols are not implemented.",
    "model.reasoning": "A portable reasoning control is not enforced by the current gateway/harness contract.",
    "model.quantization": "Weight quantization is an inference deployment choice, not a mutable world condition.",
    "model.adapter": "Adapter activation belongs to the validated training/serving workflow, not manifest patching.",
    "observe.steering": "Live activation steering and patching are not implemented.",
    "memory.edit": "Arbitrary live memory replacement, masking and retrieval policies are not implemented.",
    "resources.gpu": "World manifests cannot allocate GPU credits or enforce a provider billing limit.",
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _field(path: str) -> FieldInfo:
    model: type[BaseModel] = AgentSpec if path.startswith("agent.") else WorldManifest
    parts = path.split(".")[1:] if path.startswith("agent.") else path.split(".")
    for index, part in enumerate(parts):
        field = model.model_fields[part]
        if index != len(parts) - 1:
            candidates = (field.annotation, *get_args(field.annotation))
            model = next(candidate for candidate in candidates
                         if isinstance(candidate, type) and issubclass(candidate, BaseModel))
    return field


def _adapter(path: str) -> TypeAdapter[Any]:
    field = _field(path)
    return TypeAdapter(field.rebuild_annotation())


def _get(value: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return copy.deepcopy(value)


def _set(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    for key in parts[:-1]:
        if value.get(key) is None:
            value[key] = {}
        value = value[key]
    value[parts[-1]] = copy.deepcopy(replacement)


def _catalog() -> dict[str, dict[str, Any]]:
    catalog = {}
    for names, source, evidence, constraint in _GROUPS:
        for name in names.split():
            field = _field(name)
            schema = _adapter(name).json_schema()
            value = field.get_default(call_default_factory=True)
            if isinstance(value, BaseModel):
                value = value.model_dump(mode="json")
            if field.is_required():
                value = None
            budget = name.startswith("agent.budget.")
            catalog[name] = {
                "name": name, "description": field.description or constraint,
                "schema": schema, "default": value, "required": field.is_required(),
                "unit": ("tokens" if "tokens" in name or "limit" in name else
                         "seconds" if "seconds" in name else "characters" if "chars" in name else
                         "bytes" if "bytes" in name else "count" if "requests" in name else "dimensionless"),
                "scopes": ["global", "role", "agent"] if name.startswith("agent.") else ["global"],
                "change_timing": ["before_run", "mid_run"] if budget else ["before_run"],
                "live_api": "mesocosm conditions set (condition=budget)" if budget else None,
                "category": ("provenance" if name in {"model.hf_repo", "model.revision"} else
                             "observation" if name.startswith("observe.") else
                             "evaluation" if name.startswith("judge.") else
                             "initial_content" if name in {"services.wiki.docs", "tasks"} else "execution_policy"),
                "status": "supported", "evidence_scope": "software_contract",
                "native_acceptance": "unverified", "enforcement": source,
                "evidence": [evidence], "constraints": [constraint],
            }
    return catalog


def list_factors(world: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return allowed configuration fields, runtime boundaries and effective current values."""
    normalized = WorldManifest.model_validate(world).model_dump(mode="json") if world is not None else None
    catalog = _catalog()
    for name, entry in catalog.items():
        entry["current"] = (None if normalized is None else
                            {agent["id"]: _get(agent, name[6:]) for agent in normalized["agents"]}
                            if name.startswith("agent.") else _get(normalized, name))
        if name == "agent.model" and normalized is not None:
            entry["effective_current"] = {agent["id"]: agent["model"] or normalized["model"]["id"]
                                          for agent in normalized["agents"]}
    return [*catalog.values(), *({"name": name, "status": "unsupported", "constraints": [reason],
                                 "scopes": [], "change_timing": [], "current": None, "evidence": []}
                                for name, reason in _UNSUPPORTED.items())]


def _structural_errors(world: WorldManifest) -> list[str]:
    errors = [*inactive_condition_errors(world), *episode_condition_errors(world)]
    catalog = world.catalog()
    if len(catalog) != 1 + len(world.models):
        errors.append("model IDs must be unique")
    for agent in world.agents:
        if agent.harness != "opencode":
            errors.append(f"rejected_unsupported: agent {agent.id} harness {agent.harness}")
        if agent.model is not None and agent.model not in catalog:
            errors.append(f"agent {agent.id}: unknown model {agent.model}")
    if world.rewards.enabled and world.mode != "comparison":
        errors.append("rewards require comparison mode")
    if world.learning.mode != "none" and not world.rewards.enabled:
        errors.append("in_context feedback requires rewards.enabled")
    return errors


def configure_experiment(base_world: dict[str, Any], changes: list[dict[str, Any]],
                         roles: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Apply ordered, scoped, typed changes without mutating the base or executing a run.

    Each change has factor, value, scope (global/role/agent), optional target and
    when=before_run. Later changes win and retain their own before/after receipt.
    Agent and world identities are immutable through this interface. None is a
    value only for nullable factors; it is never an implicit delete operator.
    """
    base = WorldManifest.model_validate(base_world).model_dump(mode="json")
    manifest = copy.deepcopy(base)
    agent_ids = {agent["id"] for agent in base["agents"]}
    role_map = copy.deepcopy({} if roles is None else roles)
    if not isinstance(role_map, dict):
        raise ValueError("roles must map role names to agent ID lists")
    for role, members in role_map.items():
        if (not isinstance(role, str) or not role.strip() or not isinstance(members, list) or not members
                or not all(isinstance(member, str) for member in members)
                or len(set(members)) != len(members) or set(members) - agent_ids):
            raise ValueError(f"invalid role membership: {role!r}")
    if not isinstance(changes, list):
        raise ValueError("changes must be an ordered list")
    catalog = _catalog()
    receipts = []
    for index, change in enumerate(changes):
        if (not isinstance(change, dict) or set(change) - {"factor", "value", "scope", "target", "when"}
                or "factor" not in change or "value" not in change):
            raise ValueError(f"change {index}: expected factor, value, scope, target, when")
        name = change["factor"]
        if not isinstance(name, str) or name not in catalog:
            raise ValueError(f"rejected_unsupported: factor {name!r}; identities and arbitrary patches are immutable")
        scope = change.get("scope", "global")
        if scope not in catalog[name]["scopes"]:
            raise ValueError(f"factor {name}: unsupported scope {scope!r}")
        if change.get("when", "before_run") != "before_run":
            raise ValueError("rejected_unsupported: configure_experiment cannot apply live changes; "
                             "only the existing budget condition API can change active worlds")
        target = change.get("target")
        if scope == "global":
            if target is not None:
                raise ValueError("global changes cannot specify a target")
            selected = agent_ids
        elif scope == "role":
            if not isinstance(target, str) or target not in role_map:
                raise ValueError(f"unknown role {target!r}")
            selected = set(role_map[target])
        else:
            if not isinstance(target, str) or target not in agent_ids:
                raise ValueError(f"unknown agent {target!r}")
            selected = {target}
        adapter = _adapter(name)
        value = adapter.dump_python(adapter.validate_python(change["value"], strict=True), mode="json")
        affected = []
        if name.startswith("agent."):
            for agent in manifest["agents"]:
                if agent["id"] in selected:
                    before = _get(agent, name[6:])
                    _set(agent, name[6:], value)
                    affected.append({"agent_id": agent["id"], "before": before, "after": copy.deepcopy(value)})
        else:
            before = _get(manifest, name)
            _set(manifest, name, value)
            affected.append({"agent_id": None, "before": before, "after": copy.deepcopy(value)})
        receipts.append({"index": index, "factor": name, "scope": scope, "target": target,
                         "when": "before_run", "applied": affected})
    parsed = WorldManifest.model_validate(manifest)
    errors = _structural_errors(parsed)
    if errors:
        raise ValueError("\n".join(errors))
    normalized = parsed.model_dump(mode="json")
    payload = {"schema_version": 1, "manifest": normalized, "roles": role_map, "effective_changes": receipts,
               "base_manifest_sha256": _hash(base), "manifest_sha256": _hash(normalized),
               "validation_scope": "schema_and_structural_dependencies; call validate_experiment before execution"}
    return {**payload, "configuration_sha256": _hash(payload)}


def validate_experiment(world: dict[str, Any], base_dir: pathlib.Path) -> dict[str, Any]:
    """Run the existing source/backend-aware validator; never allocate execution resources."""
    boundaries = {
        "configuration": "supported", "native_worker_acceptance": "unverified",
        "model_serving_identity": "unverified", "deterministic_world_replay": "unsupported",
        "mid_run_changes": ["agent.budget"], "behavior_reproduction": "unverified",
        "activation_alignment": "unverified", "full_live_snapshot_restore": "unsupported",
    }
    path: pathlib.Path | None = None
    try:
        parsed = WorldManifest.model_validate(world)
        errors = _structural_errors(parsed)
        directory = pathlib.Path(base_dir).resolve(strict=True)
        # The validator resolves all relative sources from its input's parent. A
        # temporary adjacent manifest preserves that contract without rewriting tasks.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=".mesocosm-validate-",
                                         dir=directory, encoding="utf-8", delete=False) as handle:
            path = pathlib.Path(handle.name)
            json.dump(parsed.model_dump(mode="json"), handle, allow_nan=False)
        report = validate_manifest(path)
        errors.extend(report.errors)
        warnings = list(report.warnings)
        if len(parsed.agents) < 3:
            warnings.append("Research acceptance requires at least three real agents; this configuration has fewer.")
        warnings.append("Static validation does not verify Docker isolation, model availability, "
                        "original evaluator behavior or real-agent acceptance.")
        return {"valid": not errors, "errors": list(dict.fromkeys(errors)), "warnings": warnings,
                "manifest_sha256": _hash(parsed.model_dump(mode="json")),
                "capability_boundaries": boundaries, "resolved": report.resolved}
    except (ValueError, OSError, TypeError) as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "capability_boundaries": boundaries}
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
