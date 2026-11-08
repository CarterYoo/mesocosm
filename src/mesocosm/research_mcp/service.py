"""Restricted study tools; native inputs and full evidence stay operator-side."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

from mesocosm.core.manifest import WorldManifest
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import Store, canonical, digest
from mesocosm.research_mcp.corpus import Corpus, _read_file
from mesocosm.research_mcp.spec import Allowlist, DesignerSpec, MutableDesignerSpec, StrictObject

TOOLS = ("capabilities", "corpus_search", "corpus_stats", "read_events", "check_spec", "submit_spec")
POLICY_TOOLS = ("policy_catalog", "train_policy", "policy_status", "activate_policy")
MAX_RESPONSE_BYTES = 24_000
PUBLIC_SETUP = (
    "The student agents use a small language model in the approximately four-billion-parameter size class. "
    "Post-training P is fixed by the operator. Your design changes only the approved structure and environment "
    "choices; student identity and evaluation settings are not design variables."
)
MUTABLE_SETUP = (
    "You may change policy P through approved policy handles, complete agent-slot populations and bounded "
    "SFT/LoRA recipes, alongside the approved structure and environment choices. Model identities, original "
    "training examples and evaluation settings remain private. A completed weight update does not establish "
    "serving readiness or behavioral improvement."
)


def _spec_value(spec: DesignerSpec) -> dict[str, Any]:
    value = spec.model_dump()
    for field in ("policy_id", "agent_policies"):
        if value.get(field) is None:
            value.pop(field, None)
    return value


class Settings(StrictObject):
    store: str
    workbench_root: str
    base_world: str
    behavior_id: str
    allowlist: str
    corpus_root: str
    corpus_ids: list[str] = Field(default_factory=list, max_length=32)
    session_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    policy_space: str | None = None


class DesignerWorkbench:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.store)
        self.workbench = ResearchWorkbench(settings.workbench_root)
        self.corpus = Corpus(settings.corpus_root, public_only=True)
        allowlist_bytes = _read_file(Path(settings.allowlist))
        self.allowlist = Allowlist.model_validate_json(allowlist_bytes).checked()
        base_world_bytes = _read_file(Path(settings.base_world))
        self.base_world_sha256 = hashlib.sha256(base_world_bytes).hexdigest()
        self.allowlist_sha256 = hashlib.sha256(allowlist_bytes).hexdigest()
        self.policy_space = None
        self.policy_manager = None
        self.agent_ids: list[str] = []
        self.policy_space_sha256 = None
        if settings.policy_space is not None:
            from mesocosm.research_mcp.policy import PolicySpace

            policy_bytes = _read_file(Path(settings.policy_space))
            self.policy_space = PolicySpace.model_validate_json(policy_bytes)
            self.workbench.allowed_servers = self.policy_space.allowed_servers(self.workbench)
            self.policy_space.checked(self.workbench)
            self.policy_space_sha256 = hashlib.sha256(policy_bytes).hexdigest()
            world = WorldManifest.model_validate(yaml.safe_load(base_world_bytes))
            self.agent_ids = [agent.id for agent in world.agents]
        self.tools = TOOLS + POLICY_TOOLS if self.policy_space is not None else TOOLS
        # Bind final submission to the operator's session setup without publishing it.
        context: dict[str, Any] = {"settings": settings.model_dump(exclude_none=True),
                                   "base_world_sha256": self.base_world_sha256,
                                   "allowlist_sha256": self.allowlist_sha256,
                                   "allowlist_values": self.allowlist.model_dump()}
        if self.policy_space is not None:
            context["policy_space_sha256"] = self.policy_space_sha256
        self.context_sha256 = digest(context)
        if self.policy_space is not None:
            from mesocosm.research_mcp.policy import PolicyManager

            self.policy_manager = PolicyManager(self.policy_space, self.store, self.workbench,
                                                settings.session_id, self.context_sha256)
        context_id = self.store.put("designer_context", context)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            key = "designer-context:" + settings.session_id
            row = db.execute("SELECT digest FROM requests WHERE key=?", (key,)).fetchone()
            self.context_changed = bool(row and row[0] != self.context_sha256)
            if row is None:
                db.execute("INSERT INTO requests VALUES (?,?,?,?,?,NULL)",
                           (key, self.context_sha256, "designer_context", "completed",
                            canonical({"context_id": context_id})))

    def _assert_context(self) -> None:
        changed = self.context_changed
        try:
            changed |= hashlib.sha256(_read_file(Path(self.settings.base_world))).hexdigest() != self.base_world_sha256
            changed |= hashlib.sha256(_read_file(Path(self.settings.allowlist))).hexdigest() != self.allowlist_sha256
            if self.settings.policy_space is not None:
                changed |= (hashlib.sha256(_read_file(Path(self.settings.policy_space))).hexdigest()
                            != self.policy_space_sha256)
                assert self.policy_space is not None
                self.policy_space.checked(self.workbench)
        except (OSError, ValueError, KeyError):
            changed = True
        if changed:
            raise ToolInputError("study_context_changed", "The study setup changed. Restore the original "
                                 "operator setup or start a new designer session.")

    def _policies(self):
        if self.policy_manager is None:
            raise ToolInputError("unavailable_tool", "Use the advertised designer tools.")
        return self.policy_manager

    def _assert_open(self) -> None:
        with self.store.connection() as db:
            final = db.execute("SELECT 1 FROM requests WHERE key=?",
                               ("designer-final:" + self.settings.session_id,)).fetchone()
        if final:
            raise ToolInputError("immutable_submission", "This session already has an immutable final spec.")

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        self.store.audit("designer.call.started", {"tool": name, "arguments": arguments,
                                                  "session_id": self.settings.session_id})
        try:
            if name not in self.tools:
                raise ToolInputError("unavailable_tool", "Use the advertised designer tools.")
            mutating = name in {"train_policy", "policy_status", "activate_policy", "submit_spec"}
            if self.policy_manager is not None and mutating:
                with self.store.lease("designer-mutation:" + self.settings.session_id):
                    self._assert_context()
                    result = getattr(self, name)(**arguments)
                    if inspect.isawaitable(result):
                        result = await result
                    self._assert_context()
            else:
                result = getattr(self, name)(**arguments)
                if inspect.isawaitable(result):
                    result = await result
            if len(canonical(result).encode()) > MAX_RESPONSE_BYTES:
                raise ToolInputError("response_too_large", "Reduce limit or use a narrower filter.")
        except Exception as exc:
            # Never reflect native validator messages, paths, values or arbitrary keys.
            self.store.put("designer_diagnostic", {"tool": name, "type": type(exc).__name__,
                                                   "message": str(exc), "at": started})
            result = {"ok": False, "code": "invalid_arguments",
                      "message": "Check the advertised input fields, bounds and closed domains."}
            if isinstance(exc, ToolInputError):
                result.update(code=exc.code, message=str(exc), issues=exc.issues)
        self.store.audit("designer.call.completed", {"tool": name, "result_sha256": digest(result),
                                                    "elapsed_seconds": time.time() - started})
        return result

    def capabilities(self, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 10:
            raise ToolInputError("invalid_page", "Use offset >= 0 and limit between 1 and 10.")
        names = sorted(self.allowlist.factors)
        entries = [{"factor": name, "allowed_values": self.allowlist.factors[name],
                    **self.allowlist.dependency_hints(name)} for name in names]
        page: list[dict[str, Any]] = []
        for entry in entries[offset:offset + limit]:
            if len(canonical(page + [entry]).encode()) > 20_000:
                break
            page.append(entry)
        end = offset + len(page)
        result = {"student_setup": PUBLIC_SETUP, "factors": page, "total": len(names),
                "next_offset": end if end < len(names) else None, "corpus_ids": self.settings.corpus_ids,
                "spec_format": {"changes": [{"factor": "<listed factor>", "value": "<exact listed value>"}]},
                "locked": ["model", "population", "persona", "skills", "mcp", "tasks", "wiki_docs", "internet",
                           "judge", "observe", "budgets", "free_text"], "runtime_acceptance": "unverified"}
        if self.policy_manager is not None:
            result.update(student_setup=MUTABLE_SETUP, policy_catalog_tool="mesocosm_policy_catalog",
                          agent_slots=[f"agent-{index + 1}" for index in range(len(self.agent_ids))],
                          policy_selection="Omit P to retain the base policy, choose policy_id for all slots, or "
                                           "supply agent_policies naming every listed slot exactly once.",
                          spec_format={"changes": [{"factor": "<listed factor>", "value": "<exact listed value>"}],
                                       "policy_id": "<approved or activated public policy handle>"})
            result["locked"] = ["raw_model_config", "persona", "skills", "mcp", "tasks", "wiki_docs", "internet",
                                "judge", "observe", "budgets", "free_text"]
        return result

    def policy_catalog(self, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        self._assert_context()
        return self._policies().catalog(offset, limit)

    def train_policy(self, recipe_id: str, request_id: str,
                     parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        self._assert_context()
        self._assert_open()
        result = self._policies().train_policy(recipe_id, {} if parameters is None else parameters, request_id)
        self._assert_context()
        return result

    async def policy_status(self, policy_job_id: str,
                            action: Literal["inspect", "cancel"] = "inspect") -> dict[str, Any]:
        self._assert_context()
        result = await self._policies().policy_status(policy_job_id, action)
        self._assert_context()
        return result

    async def activate_policy(self, policy_job_id: str, reconcile: bool = False) -> dict[str, Any]:
        self._assert_context()
        self._assert_open()
        result = await self._policies().activate_policy(policy_job_id, reconcile)
        self._assert_context()
        return result

    def _corpus(self, corpus_id: str) -> None:
        if corpus_id not in self.settings.corpus_ids:
            raise ToolInputError("unknown_corpus", "Choose a corpus_id listed by mesocosm_capabilities.")

    def corpus_search(self, corpus_id: str, query: str = "", agent_id: str | None = None,
                      kind: str | None = None, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        self._corpus(corpus_id)
        return self.corpus.search(corpus_id, query, agent_id=agent_id, kind=kind, offset=offset, limit=limit)

    def corpus_stats(self, corpus_id: str) -> dict[str, Any]:
        self._corpus(corpus_id)
        return self.corpus.stats(corpus_id)

    def read_events(self, corpus_id: str, event_ids: list[str] | None = None,
                    agent_id: str | None = None, kind: str | None = None, offset: int = 0, limit: int = 20,
                    text_offset: int = 0, text_limit: int = 2000) -> dict[str, Any]:
        self._corpus(corpus_id)
        return self.corpus.read_events(corpus_id, event_ids=event_ids, agent_id=agent_id, kind=kind,
                                       offset=offset, limit=limit, text_offset=text_offset, text_limit=text_limit)

    def check_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._check_spec(spec)[0]

    def _check_spec(self, spec: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        self._assert_context()
        try:
            parsed = (MutableDesignerSpec if self.policy_manager is not None else DesignerSpec).model_validate(spec)
        except ValidationError:
            return {"valid": False, "issues": [{"field": "spec", "code": "invalid_shape",
                    "message": ("Supply approved changes and optionally policy_id or complete agent_policies."
                                if self.policy_manager is not None else
                                "Supply only changes, each containing factor and value; at most 40 changes.")}]}, None
        issues = self.allowlist.issues(parsed)
        fingerprint = digest(_spec_value(parsed))
        if issues:
            return {"valid": False, "spec_sha256": fingerprint, "issues": issues}, None
        try:
            bindings = (self._policies().bindings(parsed.policy_id, parsed.agent_policies, self.agent_ids)
                        if isinstance(parsed, MutableDesignerSpec) else {})
            configured = self.workbench.configure_experiment(
                self.settings.base_world, self.settings.behavior_id, parsed.model_dump()["changes"], **bindings)
            self._assert_context()
            experiment = self.workbench.store.get(configured["experiment_id"], "experiment")
            if experiment["source_manifest_sha256"] != self.base_world_sha256:
                raise ToolInputError("study_context_changed", "The fixed study setup changed during validation. "
                                     "Restore the original operator setup or start a new designer session.")
            if experiment.get("agent_deployments") != bindings.get("agent_deployments"):
                raise ToolInputError("study_context_changed", "The selected policy binding changed during validation. "
                                     "Start a new designer session after the operator inspects the setup.")
            validation = self.workbench.validate_experiment(configured["experiment_id"])
            self._assert_context()
            if isinstance(parsed, MutableDesignerSpec) and bindings != self._policies().bindings(
                    parsed.policy_id, parsed.agent_policies, self.agent_ids):
                raise ToolInputError("study_context_changed", "The selected policy binding changed during validation. "
                                     "Start a new designer session after the operator inspects the setup.")
            validation_id = self.store.put("designer_validation", {"spec_sha256": fingerprint,
                                           "context_sha256": self.context_sha256,
                                           "configuration": configured, "validation": validation,
                                           **({"policy_bindings": bindings}
                                              if self.policy_manager is not None else {})})
            if not validation["valid"]:
                raise ValueError(json.dumps(validation))
        except Exception as exc:
            self.store.put("designer_diagnostic", {"spec_sha256": fingerprint, "message": str(exc)})
            if isinstance(exc, ToolInputError):
                raise
            # Paths, rubric and locked base values in native exceptions remain private.
            public_issues = self.allowlist.native_issues(parsed)
            if isinstance(parsed, MutableDesignerSpec) and (parsed.policy_id is not None
                                                          or parsed.agent_policies is not None):
                policy_issue = {"field": "policy_id" if parsed.policy_id is not None else "agent_policies",
                                "code": "invalid_native_policy_combination",
                                "message": "Choose compatible catalog policies. Population members require one "
                                           "serving endpoint, base model, revision, base hash and dtype."}
                public_issues = [policy_issue] + (public_issues if parsed.changes else [])
            return {"valid": False, "spec_sha256": fingerprint, "issues": public_issues[:8]}, None
        return {"valid": True, "spec_sha256": fingerprint, "issues": [],
                "validation_scope": "configure_experiment_and_validate_experiment",
                "runtime_acceptance": "unverified"}, validation_id

    def submit_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        # The session is supplied by the operator, never selectable by a designer.
        # A transaction enforces a single final even with concurrent MCP calls.
        self._assert_context()
        parsed_model = (MutableDesignerSpec if self.policy_manager is not None else DesignerSpec).model_validate(spec)
        parsed = _spec_value(parsed_model)
        value = {"spec": parsed, "spec_sha256": digest(parsed), "context_sha256": self.context_sha256}
        key = "designer-final:" + self.settings.session_id
        with self.store.connection() as db:
            row = db.execute("SELECT digest,result FROM requests WHERE key=?", (key,)).fetchone()
            if row:
                if row[0] != digest(value):
                    raise ToolInputError("immutable_submission", "This session already has an immutable final spec.")
                return json.loads(row[1])
        checked, validation_id = self._check_spec(spec)
        if not checked["valid"]:
            return {"accepted": False, "check": checked}
        assert validation_id is not None
        self._assert_context()
        if isinstance(parsed_model, MutableDesignerSpec):
            validation = self.store.get(validation_id, "designer_validation")
            bindings = self._policies().bindings(parsed_model.policy_id, parsed_model.agent_policies, self.agent_ids)
            experiment = self.workbench.store.get(validation["configuration"]["experiment_id"], "experiment")
            if (bindings != validation["policy_bindings"]
                    or experiment.get("agent_deployments") != bindings.get("agent_deployments")):
                raise ToolInputError("study_context_changed", "The selected policy binding changed before submission. "
                                     "Start a new designer session after the operator inspects the setup.")
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,result FROM requests WHERE key=?", (key,)).fetchone()
            if row:
                if row[0] != digest(value):
                    raise ToolInputError("immutable_submission", "This session already has an immutable final spec.")
                return json.loads(row[1])
            result = {"accepted": True, "spec_sha256": value["spec_sha256"], "immutable": True}
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,NULL)",
                       (key, digest(value), "submit_spec", "completed", canonical(result)))
            final = value | {"validation_id": validation_id}
            identifier = "designer_submission-" + digest(final)
            db.execute("INSERT OR IGNORE INTO objects VALUES (?,?,?)",
                       (identifier, "designer_submission", canonical(final)))
        return result
