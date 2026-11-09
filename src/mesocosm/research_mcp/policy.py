"""Opaque choices for approved, ready policy deployments."""

from __future__ import annotations

import json
import re
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from mesocosm.execution.client import worker_url
from mesocosm.learning.config import PolicyIdentity
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import Store, digest

_HANDLE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PolicySpace(_Strict):
    deployments: dict[str, str] = Field(default_factory=dict, max_length=64)

    def allowed_servers(self, workbench: ResearchWorkbench) -> dict[str, str | None]:
        servers: dict[str, str | None] = {}
        venues = [(value["server"], value["server_token_env"])
                  for identifier in self.deployments.values()
                  for value in [workbench.store.get(identifier, "deployment")]]
        for endpoint, variable in venues:
            endpoint = worker_url(endpoint)
            if variable is not None and not re.fullmatch(r"[A-Z_][A-Z0-9_]*", variable):
                raise ValueError("invalid policy venue credential variable")
            if endpoint in servers and servers[endpoint] != variable:
                raise ValueError("one policy venue cannot have conflicting credentials")
            servers[endpoint] = variable
        return servers

    def checked(self, workbench: ResearchWorkbench) -> PolicySpace:
        for handle in self.deployments:
            if not _HANDLE.fullmatch(handle):
                raise ValueError("policy handles require 1 to 64 public identifier characters")
        self.allowed_servers(workbench)
        for identifier in self.deployments.values():
            _deployment(workbench, identifier)
        return self


def _fail(code: str, message: str, field: str) -> NoReturn:
    raise ToolInputError(code, message, issues=[{"field": field, "code": code, "message": message}])


def _deployment(workbench: ResearchWorkbench, identifier: str) -> dict[str, Any]:
    value = workbench.store.get(identifier, "deployment")
    PolicyIdentity.model_validate(value["policy"])
    job = workbench.store.job(value["deployment_job_id"])
    if job.get("kind") != "deployment" or job.get("status") != "ready" or job.get("deployment_id") != identifier:
        raise ValueError("policy deployment is not ready")
    return value


def _compatibility(value: dict[str, Any]) -> tuple[Any, ...]:
    return (worker_url(value["server"]), *(value["policy"][key] for key in
            ("base_model", "base_revision", "base_sha256", "dtype")))


def _group(value: dict[str, Any]) -> str:
    return "group-" + digest(_compatibility(value))


class PolicyManager:
    def __init__(self, space: PolicySpace, store: Store, workbench: ResearchWorkbench,
                 session_id: str, context_sha256: str):
        self.space, self.store, self.workbench = space, store, workbench
        self.session_id, self.context_sha256 = session_id, context_sha256

    def _resolve(self, identifier: str, field: str) -> tuple[str, dict[str, Any]]:
        if identifier in self.space.deployments:
            native = self.space.deployments[identifier]
        elif isinstance(identifier, str) and re.fullmatch(r"policy-[a-f0-9]{64}", identifier):
            try:
                owned = self.store.get(identifier, "policy")
            except (KeyError, ValueError):
                _fail("unknown_policy", "Choose a ready policy advertised in this session.", field)
            if owned.get("session_id") != self.session_id or owned.get("context_sha256") != self.context_sha256:
                _fail("unknown_policy", "Choose a ready policy advertised in this session.", field)
            native = owned["deployment_id"]
        else:
            _fail("unknown_policy", "Choose a ready policy advertised in this session.", field)
        try:
            return native, _deployment(self.workbench, native)
        except (KeyError, ValueError):
            _fail("policy_not_ready", "The selected policy is no longer ready; ask the operator to inspect it.", field)
        raise AssertionError("unreachable")

    def catalog(self, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 10:
            _fail("invalid_page", "Use offset >= 0 and limit between 1 and 10.", "offset")
        identifiers = set(self.space.deployments)
        with self.store.connection() as db:
            for identifier, raw in db.execute("SELECT id,body FROM objects WHERE kind='policy'"):
                value = json.loads(raw)
                if value.get("session_id") == self.session_id and value.get("context_sha256") == self.context_sha256:
                    identifiers.add(identifier)
        entries = sorted(identifiers)
        policies = []
        for identifier in entries[offset:offset + limit]:
            _, deployed = self._resolve(identifier, "policy_id")
            policies.append({"policy_id": identifier, "compatibility_group": _group(deployed)})
        end = offset + len(policies)
        return {"policies": policies, "total": len(entries),
                "next_offset": end if end < len(entries) else None}

    def bindings(self, policy_id: str | None, agent_policies: dict[str, str] | None,
                 agent_ids: list[str]) -> dict[str, Any]:
        if policy_id is None and agent_policies is None:
            return {}
        if policy_id is not None and agent_policies is not None:
            _fail("conflicting_policy_selection", "Choose shared policy_id or agent_policies, once.", "policy_id")
        if policy_id is not None:
            choices = {agent: policy_id for agent in agent_ids}
        else:
            assert agent_policies is not None
            slots = {f"agent-{index}": agent for index, agent in enumerate(agent_ids, 1)}
            if set(agent_policies) != set(slots):
                _fail("incomplete_population", "Assign one ready policy to every advertised agent slot.",
                      "agent_policies")
            choices = {slots[slot]: identifier for slot, identifier in agent_policies.items()}
        resolved = {agent: self._resolve(identifier, "policy_id" if policy_id else "agent_policies")
                    for agent, identifier in choices.items()}
        if len({_compatibility(value) for _, value in resolved.values()}) != 1:
            _fail("incompatible_population", "Mixed policies must have the same advertised compatibility group.",
                  "agent_policies")
        return {"agent_deployments": {agent: identifier for agent, (identifier, _) in resolved.items()}}
