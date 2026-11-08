"""Compile bounded public populations and graphs into actual native enforcement fields.

The operator owns role templates and domains. The designer supplies relationships and
opaque role choices, never instructions, paths, model identities or evaluator settings.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import Any, NoReturn

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, model_validator

from mesocosm.core.access import AccessSpec, ServiceName, validate_access
from mesocosm.core.config import absolutize_manifest
from mesocosm.core.manifest import PeerReviewSpec, WorldManifest
from mesocosm.core.structure import ExecutionGraph, StructureSpec, structure_errors
from mesocosm.reproduction import factors
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.store import canonical, digest
from mesocosm.research_mcp.corpus import _read_file

_HANDLE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
_SLOT = re.compile(r"^agent-([1-9]|[12][0-9]|3[0-2])$")
_PROTECTED = ("structure", "services.access", "services.peer_review", "isolate_agents")
_CHANGE_PREFIXES = (
    "services.chat.",
    "services.cache.",
    "artifacts.",
    "history.",
    "reputation.",
    "rewards.",
    "learning.",
    "task_assignment.",
)
_CHANGE_FIELDS = {"services.board", "services.git", "task_services", "channels"}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PopulationMember(_Strict):
    slot: str = Field(pattern=r"^agent-([1-9]|[12][0-9]|3[0-2])$")
    role: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,47}$")
    team: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,47}$")


class StructureInput(_Strict):
    population: list[PopulationMember] | None = Field(default=None, min_length=3, max_length=32)
    visibility: AccessSpec
    work: dict[str, list[str]] = Field(default_factory=dict, max_length=32)
    reviews: dict[str, list[str]] = Field(default_factory=dict, max_length=32)
    execution: ExecutionGraph

    @model_validator(mode="after")
    def bounded(self) -> StructureInput:
        if len(canonical(self.model_dump(mode="json")).encode()) > 24_000:
            raise ValueError("structure input exceeds 24000 bytes")
        graph = StructureSpec(work=self.work, reviews=self.reviews, execution=self.execution)
        values = [*graph.work, *graph.reviews, *graph.execution.depends_on, *graph.execution.finalizers]
        values += [
            target
            for edges in (graph.work, graph.reviews, graph.execution.depends_on)
            for targets in edges.values()
            for target in targets
        ]
        if any(not _SLOT.fullmatch(value) for value in values):
            raise ValueError("structure graphs require public agent-1 through agent-32 slots")
        if len(graph.execution.finalizers) > 32:
            raise ValueError("execution finalizers exceed 32 agents")
        access = self.visibility.model_dump(mode="json")
        for category in ("services", "git", "channels", "direct_messages", "artifacts"):
            entries = access[category] or {}
            limit = 9 if category == "services" else 16 if category in {"git", "channels"} else 32
            if len(entries) > limit or any(len(key) > 64 for key in entries):
                raise ValueError("visibility domains exceed their public collection/string bounds")
            lists = (
                [grant[operation] for grant in entries.values() for operation in ("read", "write")]
                if category in {"services", "git", "channels"}
                else list(entries.values())
            )
            if any(len(members) > 48 or any(len(value) > 53 for value in members) for members in lists):
                raise ValueError("visibility selectors exceed their public collection/string bounds")
        if (
            any(
                len(targets) > 32
                for edges in (graph.work, graph.reviews, graph.execution.depends_on)
                for targets in edges.values()
            )
            or len(graph.execution.depends_on) > 32
        ):
            raise ValueError("structure graphs exceed 32 agents")
        return self


def _fail(code: str, message: str, field: str, **details: Any) -> NoReturn:
    raise ToolInputError(code, message, issues=[{"field": field, "code": code, "message": message, **details}])


class StructureSpace(_Strict):
    roles: dict[str, str] = Field(min_length=1, max_length=16)
    teams: list[str] = Field(default_factory=list, max_length=16)
    min_agents: int = Field(default=3, ge=3, le=32)
    max_agents: int = Field(default=8, ge=3, le=32)
    allowed_services: list[ServiceName] = Field(default_factory=list, max_length=9)
    channels: list[str] = Field(default_factory=list, max_length=16)
    repositories: list[str] = Field(default_factory=list, max_length=16)
    _base_agent_ids: list[str] = PrivateAttr(default_factory=list)

    def checked(self, base_world: WorldManifest) -> StructureSpace:
        agents = {agent.id: agent for agent in base_world.agents}
        if self.min_agents > self.max_agents or not self.min_agents <= len(agents) <= self.max_agents:
            raise ValueError("structure agent bounds must include the base population and require at least three")
        if (
            any(not _HANDLE.fullmatch(role) for role in self.roles)
            or any(not _HANDLE.fullmatch(team) for team in self.teams)
            or set(self.roles) & {self._team_role(team) for team in self.teams}
        ):
            raise ValueError("structure handles require bounded public identifiers and distinct team group names")
        if len({self._team_role(team) for team in self.teams}) != len(self.teams):
            raise ValueError("structure teams must have distinct compiled group names")
        if set(self.roles.values()) - set(agents):
            raise ValueError("structure roles must select existing approved base-world agents")
        if any(agents[identifier].url is not None for identifier in self.roles.values()):
            raise ValueError("cloneable role templates require default native harness URLs")
        for choices in (self.teams, self.allowed_services, self.channels, self.repositories):
            if len(set(choices)) != len(choices):
                raise ValueError("structure domains require distinct exact values")
        if set(self.channels) - set(base_world.services.chat.channels):
            raise ValueError("structure channels must be approved existing base-world channels")
        if set(self.repositories) - set(base_world.services.git):
            raise ValueError("structure repositories must be approved existing base-world repositories")
        if any(not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", value) for value in [*self.channels, *self.repositories]):
            raise ValueError("structure resource domains require bounded public identifiers")
        if len(canonical(self.model_dump()).encode()) > 8000:
            raise ValueError("structure space exceeds its response bound")
        self._base_agent_ids = list(agents)
        return self

    @staticmethod
    def _team_role(team: str) -> str:
        return team if team.startswith("team-") else "team-" + team

    def capabilities(self) -> dict[str, Any]:
        return {
            "population": {
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
                "roles": sorted(self.roles),
                "teams": self.teams,
                "default_agent_count": len(self._base_agent_ids),
                "slot_rule": "List contiguous agent-1..agent-N in order; omit population to retain base agents.",
            },
            "visibility": {
                "services": self.allowed_services,
                "channels": self.channels,
                "repositories": self.repositories,
                "member_selectors": "Public agent slots or team:<approved team>; overlapping selectors are rejected.",
                "unlisted_access": "denied",
                "artifacts": "Must exactly equal the work graph.",
            },
            "graphs": {
                "work": "Source slot -> readers; readers execute after source completion.",
                "reviews": "Reviewer slot -> subjects; reviewer executes after subjects and must submit every verdict.",
                "execution": "depends_on plus finalizers; only finalizers receive original verifier measurements.",
            },
            "automatic_prerequisites": {
                "services.enabled": True,
                "isolate_agents": True,
                "services.access.artifacts": "work graph",
                "services.peer_review": "graph when reviews exist; disabled otherwise",
                "services.chat.enabled": "enabled when reviews exist; otherwise unchanged",
            },
            "requirements": [
                "Docker isolated comparison, reset memory, all-agent assignment, as_ready task starts",
                "No reward/feedback/history/reputation episode barriers; no shared_dir",
                "Work edges require approved declared artifacts and board read grants",
                "Teams expand permissions and persist as role groups; team-specific rewards are unsupported",
            ],
            "runtime_acceptance": "unverified",
        }

    def _population(
        self, world: WorldManifest, design: StructureInput
    ) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
        if design.population is None:
            agents = [agent.model_dump(mode="json") for agent in world.agents]
            slots = {f"agent-{index}": agent.id for index, agent in enumerate(world.agents, 1)}
            roles = {role: [identifier] for role, identifier in self.roles.items()}
            return agents, slots, roles
        population = design.population
        if not self.min_agents <= len(population) <= self.max_agents:
            _fail(
                "population_outside_domain",
                "Choose a population within the advertised agent bounds.",
                "structure.population",
                min_agents=self.min_agents,
                max_agents=self.max_agents,
            )
        if [member.slot for member in population] != [f"agent-{index}" for index in range(1, len(population) + 1)]:
            _fail(
                "invalid_population_slots", "List contiguous agent-1..agent-N once, in order.", "structure.population"
            )
        if any(member.role not in self.roles for member in population):
            _fail(
                "unknown_structure_role",
                "Choose only advertised roles.",
                "structure.population",
                allowed_roles=sorted(self.roles),
            )
        if any(member.team is not None and member.team not in self.teams for member in population):
            _fail(
                "unknown_structure_team",
                "Choose only advertised teams.",
                "structure.population",
                allowed_teams=self.teams,
            )
        templates = {agent.id: agent for agent in world.agents}
        agents, roles = [], {}
        used_names = {member.slot for member in population}
        for member in population:
            agent = templates[self.roles[member.role]].model_dump(mode="json")
            agent["id"], agent["url"] = member.slot, None
            if agent["persona"] is not None:
                persona = agent["persona"]
                username = persona["username"]
                if username in used_names:
                    attempt = 0
                    while True:
                        suffix = "-" + member.slot + (f"-{attempt}" if attempt else "")
                        candidate = username[: 31 - len(suffix)] + suffix
                        if candidate not in used_names:
                            username = candidate
                            break
                        attempt += 1
                persona["username"] = username
                used_names.add(username)
                if persona.get("hostname"):
                    persona["hostname"] = member.slot
            agents.append(agent)
            roles.setdefault(member.role, []).append(member.slot)
            if member.team is not None:
                roles.setdefault(self._team_role(member.team), []).append(member.slot)
        return agents, {member.slot: member.slot for member in population}, roles

    def _visibility(self, design: StructureInput, slots: dict[str, str], roles: dict[str, list[str]]) -> AccessSpec:
        access = design.visibility.model_dump(mode="json")
        if (
            set(access["services"]) - set(self.allowed_services)
            or set(access["git"]) - set(self.repositories)
            or set(access["channels"]) - set(self.channels)
        ):
            _fail(
                "visibility_outside_domain",
                "Use only advertised services, repositories and channels.",
                "structure.visibility",
                allowed_services=self.allowed_services,
                allowed_channels=self.channels,
                allowed_repositories=self.repositories,
            )

        def members(values: list[str], field: str) -> list[str]:
            expanded: list[str] = []
            for selector in values:
                if selector in slots:
                    selected = [slots[selector]]
                elif selector.startswith("team:") and selector[5:] in self.teams:
                    selected = roles.get(self._team_role(selector[5:]), [])
                    if not selected:
                        _fail("empty_team_selector", "The selected team has no population members.", field)
                else:
                    _fail(
                        "unknown_visibility_member", "Use an existing public slot or advertised populated team.", field
                    )
                if set(expanded) & set(selected):
                    _fail(
                        "overlapping_visibility_selectors", "Use each member once; choose a team or its members.", field
                    )
                expanded.extend(selected)
            return expanded

        for category in ("services", "git", "channels"):
            for identifier, grant in access[category].items():
                for operation in ("read", "write"):
                    grant[operation] = members(
                        grant[operation], f"structure.visibility.{category}.{identifier}.{operation}"
                    )
        for category in ("direct_messages", "artifacts"):
            if access[category] is None:
                continue
            mapped = {}
            for source, readers in access[category].items():
                if source not in slots:
                    _fail(
                        "unknown_visibility_member",
                        "Visibility sources require an existing public slot.",
                        "structure.visibility." + category,
                    )
                mapped[slots[source]] = members(readers, "structure.visibility." + category)
            access[category] = mapped
        work = {slots[source]: [slots[reader] for reader in readers] for source, readers in design.work.items()}
        if access["artifacts"] is not None and access["artifacts"] != work:
            _fail(
                "conflicting_artifact_visibility",
                "Artifact visibility must exactly equal the work graph.",
                "structure.visibility.artifacts",
            )
        access["artifacts"] = work
        return AccessSpec.model_validate(access)

    def compose(self, base_world_path: Path, design: StructureInput, changes: list[dict[str, Any]]) -> dict[str, Any]:
        source = _read_file(Path(base_world_path))
        world = WorldManifest.model_validate(yaml.safe_load(source))
        self.checked(world)
        design = StructureInput.model_validate(design)
        agents, slots, roles = self._population(world, design)
        if any(task.side_task is not None for task in world.tasks):
            _fail(
                "unsupported_structure_task",
                "Graph execution cannot preserve side-task verification; use an approved compatible task.",
                "structure",
            )
        referenced = set(design.work) | set(design.reviews) | set(design.execution.depends_on)
        referenced.update(design.execution.finalizers)
        referenced.update(
            target
            for edges in (design.work, design.reviews, design.execution.depends_on)
            for targets in edges.values()
            for target in targets
        )
        if referenced - set(slots):
            _fail("unknown_structure_slot", "Graphs must reference existing public population slots.", "structure")
        manifest = world.model_dump(mode="json")
        receipts = []
        for index, change in enumerate(changes):
            if not isinstance(change, dict) or set(change) != {"factor", "value"}:
                _fail("invalid_structure_change", "Supply only typed global factor/value changes.", f"changes.{index}")
            name = change["factor"]
            if not isinstance(name, str) or name.startswith(_PROTECTED):
                _fail(
                    "structure_managed_factor",
                    "Omit this change; the structure compiler owns graphs, ACLs and isolation.",
                    f"changes.{index}.factor",
                )
            if name not in _CHANGE_FIELDS and not name.startswith(_CHANGE_PREFIXES):
                _fail("locked_factor", "Choose only advertised finite S/E factors.", f"changes.{index}.factor")
            if design.reviews and name == "services.chat.enabled" and change["value"] is not True:
                _fail(
                    "structure_review_chat_required",
                    "Review graphs require enabled chat; remove this disabling change.",
                    f"changes.{index}.value",
                )
            try:
                adapter = factors._adapter(name)
                value = adapter.dump_python(adapter.validate_python(change["value"], strict=True), mode="json")
            except (KeyError, ValueError, TypeError):
                _fail("invalid_structure_change", "Choose an exact advertised native factor value.", f"changes.{index}")
            before = factors._get(manifest, name)
            factors._set(manifest, name, value)
            receipts.append({"index": index, "factor": name, "before": before, "after": copy.deepcopy(value)})
        manifest["agents"] = agents
        manifest["structure"] = {
            "work": {slots[key]: [slots[item] for item in values] for key, values in design.work.items()},
            "reviews": {slots[key]: [slots[item] for item in values] for key, values in design.reviews.items()},
            "execution": {
                "depends_on": {
                    slots[key]: [slots[item] for item in values] for key, values in design.execution.depends_on.items()
                },
                "finalizers": [slots[item] for item in design.execution.finalizers],
            },
        }
        manifest["services"]["access"] = self._visibility(design, slots, roles).model_dump(mode="json")
        automatic = {
            "services.enabled": True,
            "isolate_agents": True,
            "services.peer_review": PeerReviewSpec(
                enabled=bool(design.reviews), protocol="graph" if design.reviews else "legacy"
            ).model_dump(mode="json"),
        }
        if design.reviews:
            automatic["services.chat.enabled"] = True
        for name, value in automatic.items():
            before = factors._get(manifest, name)
            factors._set(manifest, name, value)
            receipts.append({"source": "structure_prerequisite", "factor": name, "before": before, "after": value})
        # Parse each native field first to obtain the native dependency diagnostics without
        # reflecting a Pydantic input dump (which contains private models and source paths).
        fields = {}
        for name, field in WorldManifest.model_fields.items():
            try:
                fields[name] = TypeAdapter(field.rebuild_annotation()).validate_python(manifest[name])
            except (ValueError, TypeError):
                requirements = {
                    "artifacts": "Enable approved declared paths together; use completed eligibility and no scores.",
                    "rewards": "Graph execution requires disabled rewards and compatible default reward controls.",
                    "services": "Check enabled chat, approved channels and graph-compatible service controls.",
                    "agents": "Role templates require distinct native identities; ask the operator to inspect them.",
                }
                message = f"The proposed {name} settings are incompatible. " + requirements.get(
                    name, "Check the exact approved values and their native dependencies."
                )
                _fail("invalid_structure_combination", message, "structure")
        candidate = WorldManifest.model_construct(**fields)
        errors = structure_errors(candidate)
        assert candidate.services.access is not None
        errors.extend(
            validate_access(
                candidate.services.access,
                list(slots.values()),
                candidate.services.git,
                candidate.services.chat.channels,
                shared_directory="shared_dir" in candidate.channel_types(),
            )
        )
        if errors:
            public = []
            pattern = re.compile(
                r"(?<![a-z0-9-])(?:" + "|".join(re.escape(value) for value in slots.values()) + r")(?![a-z0-9-])"
            )
            names = {identifier: slot for slot, identifier in slots.items()}
            for error in errors[:8]:
                error = pattern.sub(lambda match: names[match.group()], error)
                public.append({"field": "structure", "code": "invalid_structure_combination", "message": error})
            raise ToolInputError(
                "invalid_structure_combination",
                "Repair the listed graph and native-setting dependencies.",
                issues=public,
            )
        try:
            normalized = WorldManifest.model_validate(manifest)
            if factors._structural_errors(normalized):
                raise ValueError("inactive or incompatible condition")
        except (ValueError, TypeError):
            _fail(
                "invalid_structure_combination",
                "The graphs conflict with approved S/E settings. Check enabled components and dependencies.",
                "structure",
            )
        final = normalized.model_dump(mode="json")
        absolutize_manifest(final, Path(base_world_path).resolve().parent)
        payload = {
            "manifest": final,
            "agent_ids": [agent["id"] for agent in final["agents"]],
            "roles": roles,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "structure_space_sha256": digest(self.model_dump()),
            "design": design.model_dump(mode="json"),
            "applied_changes": receipts,
        }
        return {**payload, "composition_sha256": digest(payload)}
