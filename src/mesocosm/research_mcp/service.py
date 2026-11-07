"""Six restricted tools; the workbench and full validation evidence stay operator-side."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import Store, canonical, digest
from mesocosm.research_mcp.corpus import Corpus, _read_file
from mesocosm.research_mcp.spec import Allowlist, DesignerSpec, StrictObject

TOOLS = ("capabilities", "corpus_search", "corpus_stats", "read_events", "check_spec", "submit_spec")
MAX_RESPONSE_BYTES = 24_000
PUBLIC_SETUP = (
    "The student agents use a small language model in the approximately four-billion-parameter size class. "
    "Post-training P is fixed by the operator. Your design changes only the approved structure and environment "
    "choices; student identity and evaluation settings are not design variables."
)


class Settings(StrictObject):
    store: str
    workbench_root: str
    base_world: str
    behavior_id: str
    allowlist: str
    corpus_root: str
    corpus_ids: list[str] = Field(default_factory=list, max_length=32)
    session_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class DesignerWorkbench:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.store)
        self.workbench = ResearchWorkbench(settings.workbench_root)
        self.corpus = Corpus(settings.corpus_root, public_only=True)
        allowlist_bytes = _read_file(Path(settings.allowlist))
        self.allowlist = Allowlist.model_validate_json(allowlist_bytes).checked()
        self.base_world_sha256 = hashlib.sha256(_read_file(Path(settings.base_world))).hexdigest()
        self.allowlist_sha256 = hashlib.sha256(allowlist_bytes).hexdigest()
        # Bind final submission to the operator's session setup without publishing it.
        context = {"settings": settings.model_dump(), "base_world_sha256": self.base_world_sha256,
                   "allowlist_sha256": self.allowlist_sha256, "allowlist_values": self.allowlist.model_dump()}
        self.context_sha256 = digest(context)
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
        except (OSError, ValueError):
            changed = True
        if changed:
            raise ToolInputError("study_context_changed", "The fixed study setup changed. Restore the original "
                                 "operator setup or start a new designer session.")

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        self.store.audit("designer.call.started", {"tool": name, "arguments": arguments,
                                                  "session_id": self.settings.session_id})
        try:
            if name not in TOOLS:
                raise ToolInputError("unavailable_tool", "Use one of the six advertised designer tools.")
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
        return {"student_setup": PUBLIC_SETUP, "factors": page, "total": len(names),
                "next_offset": end if end < len(names) else None, "corpus_ids": self.settings.corpus_ids,
                "spec_format": {"changes": [{"factor": "<listed factor>", "value": "<exact listed value>"}]},
                "locked": ["model", "population", "persona", "skills", "mcp", "tasks", "wiki_docs", "internet",
                           "judge", "observe", "budgets", "free_text"], "runtime_acceptance": "unverified"}

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
            parsed = DesignerSpec.model_validate(spec)
        except ValidationError:
            return {"valid": False, "issues": [{"field": "spec", "code": "invalid_shape",
                    "message": "Supply only changes, each containing factor and value; at most 40 changes."}]}, None
        issues = self.allowlist.issues(parsed)
        fingerprint = digest(parsed.model_dump())
        if issues:
            return {"valid": False, "spec_sha256": fingerprint, "issues": issues}, None
        try:
            configured = self.workbench.configure_experiment(
                self.settings.base_world, self.settings.behavior_id, parsed.model_dump()["changes"])
            self._assert_context()
            experiment = self.workbench.store.get(configured["experiment_id"], "experiment")
            if experiment["source_manifest_sha256"] != self.base_world_sha256:
                raise ToolInputError("study_context_changed", "The fixed study setup changed during validation. "
                                     "Restore the original operator setup or start a new designer session.")
            validation = self.workbench.validate_experiment(configured["experiment_id"])
            self._assert_context()
            validation_id = self.store.put("designer_validation", {"spec_sha256": fingerprint,
                                           "context_sha256": self.context_sha256,
                                           "configuration": configured, "validation": validation})
            if not validation["valid"]:
                raise ValueError(json.dumps(validation))
        except Exception as exc:
            self.store.put("designer_diagnostic", {"spec_sha256": fingerprint, "message": str(exc)})
            if isinstance(exc, ToolInputError):
                raise
            # Paths, rubric and locked base values in native exceptions remain private.
            return {"valid": False, "spec_sha256": fingerprint, "issues": self.allowlist.native_issues(parsed)}, None
        return {"valid": True, "spec_sha256": fingerprint, "issues": [],
                "validation_scope": "configure_experiment_and_validate_experiment",
                "runtime_acceptance": "unverified"}, validation_id

    def submit_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        # The session is supplied by the operator, never selectable by a designer.
        # A transaction enforces a single final even with concurrent MCP calls.
        self._assert_context()
        parsed = DesignerSpec.model_validate(spec).model_dump()
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
