"""Separate work, review and execution graphs for native comparison tasks.

Work edges authorize artifact readers, review edges authorize verdicts, and execution edges
admit real harness stages. None of these edges are inferred from prompt text.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from mesocosm.core.manifest import WorldManifest


class GraphModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExecutionGraph(GraphModel):
    depends_on: dict[str, list[str]] = Field(default_factory=dict)
    finalizers: list[str] = Field(min_length=1)

    @field_validator("finalizers")
    @classmethod
    def unique_finalizers(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("execution.finalizers must be unique")
        return values


class StructureSpec(GraphModel):
    work: dict[str, list[str]] = Field(default_factory=dict,
        description="Artifact source agent to authorized reader agents; no implicit grants")
    reviews: dict[str, list[str]] = Field(default_factory=dict,
        description="Reviewer to subjects; every assigned verdict is required before stage completion")
    execution: ExecutionGraph

    def waves(self, agents: list[str]) -> list[list[str]]:
        remaining, completed = set(agents), set[str]()
        waves: list[list[str]] = []
        while remaining:
            ready = [agent for agent in agents if agent in remaining
                     and set(self.execution.depends_on.get(agent, [])) <= completed]
            if not ready:
                raise ValueError("execution.depends_on contains a cycle")
            waves.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)
        return waves


def structure_errors(world: WorldManifest) -> list[str]:
    graph = world.structure
    peer = world.services.peer_review
    if graph is None:
        return (["rejected_unsupported: peer_review.protocol: graph requires structure"]
                if peer.protocol == "graph" else [])
    errors: list[str] = []
    agents = [agent.id for agent in world.agents]
    known = set(agents)
    maps = {"work": graph.work, "reviews": graph.reviews, "execution.depends_on": graph.execution.depends_on}
    for name, edges in maps.items():
        referenced = set(edges) | {target for targets in edges.values() for target in targets}
        if referenced - known:
            errors.append(f"structure.{name} references unknown agents: {sorted(referenced - known)}")
        for source, targets in edges.items():
            if source in targets or len(set(targets)) != len(targets):
                errors.append(f"structure.{name}.{source} must name unique other agents")
    finals = set(graph.execution.finalizers)
    if finals - known:
        errors.append(f"structure.execution.finalizers references unknown agents: {sorted(finals - known)}")
    try:
        graph.waves(agents)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        def ancestors(agent: str, seen: set[str] | None = None) -> set[str]:
            found = set() if seen is None else seen
            for parent in graph.execution.depends_on.get(agent, []):
                if parent not in found:
                    found.add(parent)
                    ancestors(parent, found)
            return found

        reached = set().union(*(ancestors(agent) for agent in finals)) if finals else set()
        if known - finals - reached:
            errors.append("every non-final agent must precede a designated finalizer")
        if any(finals & set(parents) for parents in graph.execution.depends_on.values()):
            errors.append("designated finalizers must be execution sinks")
        for reviewer, subjects in graph.reviews.items():
            if not set(subjects) <= ancestors(reviewer):
                errors.append(f"reviewer {reviewer} must execute after every reviewed subject")
        for reader in known:
            sources = {source for source, readers in graph.work.items() if reader in readers}
            if not sources <= ancestors(reader):
                errors.append(f"artifact reader {reader} must execute after every granted work source")
        for finalizer in finals:
            if not known - finals <= ancestors(finalizer):
                errors.append(f"finalizer {finalizer} must execute after every subordinate agent")
    if (world.execution.backend != "docker" or world.mode != "comparison" or world.memory != "reset"
            or world.task_services != "isolated" or world.task_assignment.policy != "all"
            or world.task_start != "as_ready" or any(task.stagger_seconds for task in world.tasks)):
        errors.append("structure requires Docker isolated comparison, reset memory, all agents and as_ready starts")
    if world.rewards.enabled or world.learning.mode != "none" or world.history.episodes or world.reputation.enabled:
        errors.append("structure cannot combine with reward/feedback/history/reputation episode barriers")
    if graph.reviews:
        if not peer.enabled or peer.protocol != "graph" or not world.services.chat.enabled:
            errors.append("structure.reviews requires enabled chat and peer_review.protocol: graph")
    elif peer.enabled:
        errors.append("structure without review edges requires peer_review disabled")
    access = world.services.access
    if not world.services.enabled or not world.isolate_agents or access is None:
        errors.append("structure requires enabled services, isolated agents and explicit services.access")
    else:
        if access.artifacts != graph.work:
            errors.append("structure.work must exactly match services.access.artifacts")
        for reviewer in graph.reviews:
            if not access.allows(reviewer, "chat", "write"):
                errors.append(f"reviewer {reviewer} requires chat write access")
        if world.artifacts.enabled:
            for readers in graph.work.values():
                if any(not access.allows(reader, "board", "read") for reader in readers):
                    errors.append("artifact readers require board read access")
    if graph.work and not world.artifacts.enabled:
        errors.append("structure.work requires declared artifact exchange")
    if world.artifacts.enabled and (world.artifacts.eligibility != "completed" or world.artifacts.show_score):
        errors.append("structure artifacts require completed eligibility and no score exposure")
    if "shared_dir" in world.channel_types():
        errors.append("structure cannot enforce visibility through shared_dir")
    return [f"rejected_unsupported: {error}" for error in errors]
