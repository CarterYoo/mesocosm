"""Finite operator-approved S/E choices, with no designer-authored instruction fields."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mesocosm.reproduction import factors
from mesocosm.reproduction.store import canonical
from mesocosm.research_mcp.structure import StructureInput

# Whole objects and unbounded string fields are deliberately absent. Strings in
# approved values (for example artifact names) are exact operator-owned choices.
SUPPORTED_FACTORS = frozenset({
    "services.board", "services.chat.enabled", "services.chat.channels", "services.chat.max_chars",
    "services.chat.max_messages", "channels", "artifacts.enabled", "artifacts.paths", "artifacts.eligibility",
    "artifacts.success_threshold", "artifacts.show_author", "artifacts.show_score", "artifacts.max_bytes",
    "services.peer_review.enabled", "services.peer_review.protocol", "services.peer_review.topology",
    "services.peer_review.matching_seed", "reputation.enabled", "reputation.window", "reputation.metrics",
    "rewards.allocation", "rewards.team_weight", "history.episodes", "history.scope", "history.max_chars",
    "learning.mode", "learning.feedback.fields", "learning.feedback.delay_episodes", "learning.feedback.max_chars",
    "task_assignment.policy", "task_assignment.participants", "task_assignment.score", "task_assignment.seed",
    "services.git", "services.cache.enabled", "services.cache.writable", "services.cache.list_recent",
    "isolate_agents", "task_services",
})

# These are general contracts from condition_validation, episode_validation and
# manifest validation. Never derive public hints from exception text or base values.
_DEPENDENCIES = {
    "services.chat.channels": ("Nondefault chat channels require enabled chat.", ("services.chat.enabled",)),
    "services.chat.max_chars": ("Nondefault chat character limits require enabled chat.",
                                ("services.chat.enabled",)),
    "services.chat.max_messages": ("Nondefault chat message limits require enabled chat.",
                                   ("services.chat.enabled",)),
    "services.peer_review.enabled": ("Peer review requires enabled chat; ratings requires synchronized rewards.",
                                     ("services.chat.enabled", "services.peer_review.protocol")),
    "services.peer_review.protocol": ("Ratings requires synchronized rewards; peer review requires enabled chat.",
                                      ("services.peer_review.enabled", "services.chat.enabled")),
    "services.peer_review.topology": ("random_ring requires enabled ratings peer review.",
                                      ("services.peer_review.enabled", "services.peer_review.protocol")),
    "services.peer_review.matching_seed": ("A nonzero matching seed requires random_ring peer review.",
                                           ("services.peer_review.topology",)),
    "rewards.team_weight": ("A nondefault team weight applies only to mixed reward allocation.",
                             ("rewards.allocation",)),
    "history.scope": ("Nondefault history options require a positive history.episodes choice.",
                       ("history.episodes",)),
    "history.max_chars": ("Nondefault history options require a positive history.episodes choice.",
                           ("history.episodes",)),
    "learning.feedback.fields": ("Nondefault feedback options require learning.mode=in_context; peer review "
                                  "feedback fields also require enabled ratings peer review.",
                                  ("learning.mode", "services.peer_review.enabled", "services.peer_review.protocol")),
    "learning.feedback.delay_episodes": ("Nondefault feedback options require learning.mode=in_context.",
                                          ("learning.mode",)),
    "learning.feedback.max_chars": ("Nondefault feedback options require learning.mode=in_context.",
                                     ("learning.mode",)),
    "artifacts.enabled": ("Artifact exchange requires a board, synchronized rewards and isolated task services.",
                           ("services.board", "task_services")),
    "task_services": ("Artifact exchange and selective task assignment require isolated task services.",
                       ("artifacts.enabled", "task_assignment.policy")),
    "task_assignment.policy": ("Selective assignment requires isolated tasks, individual rewards, "
                                "no peer review and an approved participant count below the fixed agent count.",
                                ("task_services", "rewards.allocation", "services.peer_review.enabled",
                                 "task_assignment.participants")),
}


def _bounded_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for issue in issues[:8]:
        if len(canonical(bounded + [issue]).encode()) > 20_000:
            break
        bounded.append(issue)
    return bounded


class StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Change(StrictObject):
    factor: str = Field(min_length=1, max_length=96)
    value: Any


class DesignerSpec(StrictObject):
    changes: list[Change] = Field(default_factory=list, max_length=40)


class MutableDesignerSpec(DesignerSpec):
    policy_id: str | None = Field(default=None, min_length=1, max_length=96)
    agent_policies: dict[str, str] | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def exclusive_policy(self) -> MutableDesignerSpec:
        if self.policy_id is not None and self.agent_policies is not None:
            raise ValueError("Choose a shared policy_id or complete agent_policies, not both.")
        return self


class StructuredDesignerSpec(DesignerSpec):
    structure: StructureInput | None = None


class StructuredMutableDesignerSpec(MutableDesignerSpec):
    structure: StructureInput | None = None


def designer_spec_type(mutable_policy: bool = False, mutable_structure: bool = False) -> type[DesignerSpec]:
    if mutable_structure:
        return StructuredMutableDesignerSpec if mutable_policy else StructuredDesignerSpec
    return MutableDesignerSpec if mutable_policy else DesignerSpec


class Allowlist(StrictObject):
    factors: dict[str, list[Any]]

    def dependency_hints(self, name: str) -> dict[str, Any]:
        if name not in _DEPENDENCIES:
            return {}
        requirement, related = _DEPENDENCIES[name]
        choices = [factor for factor in related if factor in self.factors]
        return {"dependency_hints": [requirement], "related_allowed_factors": choices,
                "hint_scope": "general_contract; does_not_report_current_world_settings"}

    def native_issues(self, spec: DesignerSpec) -> list[dict[str, Any]]:
        issues = [{"field": f"changes.{index}.value", "factor": change.factor,
                   "code": "invalid_native_combination", "allowed_values": self.factors[change.factor],
                   "message": "This combination failed native validation. Check the general requirements and "
                              "approved related choices, or remove this change and recheck.",
                   **self.dependency_hints(change.factor)} for index, change in enumerate(spec.changes)]
        if not issues:
            issues = [{"field": "changes", "code": "invalid_native_combination",
                       "message": "The unchanged world failed native validation; the operator must inspect "
                                  "the private diagnostic before a design can be validated."}]
        return _bounded_issues(issues)

    def checked(self) -> Allowlist:
        if not self.factors or set(self.factors) - SUPPORTED_FACTORS:
            raise ValueError("allowlist must contain supported S/E factors only")
        for name, choices in self.factors.items():
            if not 1 <= len(choices) <= 32 or len({canonical(value) for value in choices}) != len(choices):
                raise ValueError("each allowlist field requires 1 to 32 distinct exact values")
            if len(canonical(choices).encode()) > 12_000:
                raise ValueError("allowlist domain exceeds response byte bound")
            for value in choices:
                # Use the actual factor type, not a parallel approximation of it.
                factors._adapter(name).validate_python(value, strict=True)
                if len(canonical(value).encode()) > 1200:
                    raise ValueError("allowlist choice exceeds byte bound")
        return self

    def issues(self, spec: DesignerSpec) -> list[dict[str, Any]]:
        issues = []
        seen = set()
        for index, change in enumerate(spec.changes):
            path = f"changes.{index}"
            if change.factor not in self.factors:
                issues.append({"field": path + ".factor", "code": "locked_factor",
                               "message": "This field is locked. Choose a factor from mesocosm_capabilities.",
                               "allowed_factors": sorted(self.factors)})
            elif canonical(change.value) not in {canonical(value) for value in self.factors[change.factor]}:
                issues.append({"field": path + ".value", "factor": change.factor, "code": "outside_domain",
                               "message": "Choose one exact allowed value.",
                               "allowed_values": self.factors[change.factor]})
            elif change.factor in seen:
                issues.append({"field": path + ".factor", "factor": change.factor, "code": "duplicate_factor",
                               "message": "Specify each factor once."})
            seen.add(change.factor)
        return _bounded_issues(issues)
