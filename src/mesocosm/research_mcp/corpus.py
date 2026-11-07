"""Operator-reviewed transcript projections, with no public route to source bytes.

Review is deliberately explicit: a source-hash-bound policy names each releasable
event and each text field within it. This is not a semantic redaction classifier.
Operator originals and mappings must never be mounted in a designer sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

MAX_SOURCE_BYTES = 32 << 20
MAX_EVENTS = 100_000
MAX_TEXT_BYTES = 32_000
MAX_PAGE_BYTES = 23_000
PUBLIC_KINDS = frozenset({"speech", "action_request", "action_execution", "tool_result", "observation", "state_change"})
_ID = re.compile(r"corpus-[a-f0-9]{32}\Z")
_EVENT_ID = re.compile(r"event-[a-f0-9]{32}\Z")
_PRIVATE_FIELDS = frozenset({
    "config", "configuration", "settings", "metadata", "analysis_metadata", "run_config", "persona",
    "system", "developer", "prompt", "prompts", "system_prompt", "reward", "rewards", "reward_scheme",
    "policy", "conditions", "condition", "parameters", "source_location", "source", "source_path",
})


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _read_file(path: Path, limit: int = MAX_SOURCE_BYTES) -> bytes:
    """Reject symlinks, including symlinked parents, before reading untrusted input."""
    path = path.absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("source path must not traverse symlinks")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("source must be a bounded regular file")
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("source exceeds byte limit")
    return content


def _pointer(value: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("review field must be a JSON pointer")
    for part in pointer[1:].split("/"):
        if re.search(r"~(?![01])", part):
            raise ValueError("invalid JSON pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if key.lower() in _PRIVATE_FIELDS or "prompt" in key.lower() or "reward" in key.lower():
            raise ValueError("condition or prompt fields cannot be released")
        try:
            value = value[int(key)] if isinstance(value, list) and key.isdecimal() else value[key]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("review field does not exist") from exc
    return value


def _decode(content: bytes, source_format: str) -> Any:
    if source_format == "jsonl":
        return [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    return json.loads(content)


def _reject_held_out(value: Any) -> None:
    """Recognize declared splits, never infer a split from dialogue content."""
    from mesocosm.reproduction.service import _held_out

    if isinstance(value, list):
        for row in value:
            _reject_held_out(row)
    elif isinstance(value, dict):
        if any(_held_out(candidate) for candidate in (value, value.get("data"), value.get("info"))):
            raise ValueError("held-out test sources cannot be released to the designer corpus")
        for key in ("events", "messages"):
            if isinstance(value.get(key), list):
                _reject_held_out(value[key])


def _source(path: Path, source_format: str) -> bytes:
    if source_format != "bundle":
        return _read_file(path)
    from mesocosm.reproduction.evidence import inspect_transcript

    # Check every source before the existing normalizer follows native links.
    # Bundle import is operator-only and never executes any recorded content.
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError("source path must not traverse symlinks")
        if entry.is_file():
            total += entry.stat().st_size
    if total > MAX_SOURCE_BYTES:
        raise ValueError("source bundle exceeds byte limit")
    if any(parent.is_symlink() for parent in (path, *path.absolute().parents)):
        raise ValueError("source path must not traverse symlinks")
    normalized = inspect_transcript(path)
    _reject_held_out(normalized)
    # Inspect original declarations before normalization can discard event or
    # native-message metadata. The inventory includes summary and lineage files.
    for entry in normalized["source"]["files"]:
        original = path / entry["path"]
        content = _read_file(original)
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("source bundle changed while checking split declarations")
        # The existing bundle importer reads these two journals as JSONL and
        # every other inventoried source (including native turns) as JSON.
        source_kind = "jsonl" if entry["path"] in {"events.jsonl", "receipts.jsonl"} else "normalized"
        _reject_held_out(_decode(content, source_kind))
    # Bind review to the conditions as well as normalized event/native evidence.
    manifest = _read_file(path / "manifest.json")
    _reject_held_out(json.loads(manifest))
    normalized["private_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    return _json(normalized)


def _rows(content: bytes, source_format: str) -> list[tuple[str, dict[str, Any], int | None]]:
    if source_format not in {"normalized", "jsonl", "bundle"}:
        raise ValueError("unsupported corpus source format")
    value = _decode(content, source_format)
    _reject_held_out(value)
    rows: list[tuple[str, dict[str, Any], int | None]] = []
    prefix = "/events" if isinstance(value, dict) else ""
    entries = value.get("events") if isinstance(value, dict) else value
    if not isinstance(entries, list):
        raise ValueError("normalized source requires events[] or a list of events")
    rows = [(f"{prefix}/{index}", row, None) for index, row in enumerate(entries)]
    if len(rows) > MAX_EVENTS or any(not isinstance(row, dict) for _, row, _ in rows):
        raise ValueError("source events must be bounded objects")
    return rows


def _kind(row: dict[str, Any], source_format: str) -> str | None:
    data: dict[str, Any] = row["data"] if isinstance(row.get("data"), dict) else {}
    info: dict[str, Any] = row["info"] if isinstance(row.get("info"), dict) else {}
    if any(str(role).lower() in {"system", "developer"}
           for role in (row.get("role"), data.get("role"), info.get("role"))):
        return None
    kind = row.get("kind")
    if kind is None and row.get("role") in {"user", "assistant", "tool"}:
        kind = "tool_result" if row["role"] == "tool" else "speech"
    return kind if kind in PUBLIC_KINDS else None


def review_source(path: str | Path, source_format: str = "normalized") -> dict[str, Any]:
    """Operator-only inventory; inspect original bytes separately before approving.

    The returned candidates are locations, not a claim that their content is safe.
    No public MCP tool should call this function.
    """
    content = _source(Path(path), source_format)
    return {"source_sha256": hashlib.sha256(content).hexdigest(), "source_format": source_format,
            "candidates": [{"pointer": pointer, "kind": _kind(row, source_format),
                            "fields": sorted(row), "episode": episode}
                           for pointer, row, episode in _rows(content, source_format)]}


class Corpus:
    """Serve masked corpus files, or import them through the separate operator API."""

    def __init__(self, root: str | Path, *, public_only: bool = False):
        self.root = Path(root).absolute()
        self.public_only = public_only
        self.public_root = self.root if public_only else self.root / "public"

    def import_source(self, path: str | Path, *, policy: dict[str, Any],
                      source_format: str = "normalized") -> dict[str, Any]:
        if self.public_only:
            raise ValueError("corpus import requires the operator store")
        content = _source(Path(path), source_format)
        sha = hashlib.sha256(content).hexdigest()
        if (not isinstance(policy, dict) or policy.get("source_sha256") != sha
                or not isinstance(policy.get("reviewer"), str) or not policy["reviewer"].strip()
                or not isinstance(policy.get("events"), dict)):
            raise ValueError("import requires a source-hash-bound operator review policy")
        redactions = policy.get("redact_literals", [])
        if (not isinstance(redactions, list)
                or any(not isinstance(value, str) or not value for value in redactions)):
            raise ValueError("redact_literals must contain nonempty strings")
        rows = _rows(content, source_format)
        selected = policy["events"]
        if not selected or set(selected) - {pointer for pointer, _, _ in rows}:
            raise ValueError("review must select existing events explicitly")
        aliases: dict[str, str] = {}

        def alias(value: Any) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError("agent identity must be a string")
            return aliases.setdefault(value, f"agent-{len(aliases) + 1}")

        events: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        for pointer, row, episode in rows:
            if pointer not in selected:
                continue
            kind = _kind(row, source_format)
            if kind is None:
                raise ValueError("system, developer, condition and ambiguous events cannot be released")
            fields = selected[pointer]
            if (not isinstance(fields, list) or not fields
                    or any(not isinstance(field, str) for field in fields) or len(set(fields)) != len(fields)):
                raise ValueError("review event must select distinct text field pointers")
            texts = [_pointer(row, field) for field in fields]
            if any(not isinstance(text, str) for text in texts):
                raise ValueError("only explicitly reviewed string fields can be released")
            text = "\n".join(texts)
            for literal in sorted(redactions, key=len, reverse=True):
                text = text.replace(literal, "[MASKED]")
            # The operator's hash-bound field approval is the semantic review.
            # This literal check is only a second mechanical barrier.
            forbidden = policy.get("forbidden_literals", [])
            if (not isinstance(forbidden, list)
                    or any(not isinstance(value, str) or not value for value in forbidden)):
                raise ValueError("forbidden_literals must contain nonempty strings")
            if any(literal.casefold() in text.casefold() for literal in forbidden):
                raise ValueError("reviewed text still contains an operator-forbidden literal")
            if len(text.encode()) > MAX_TEXT_BYTES:
                raise ValueError("reviewed event text exceeds byte limit")
            event = {"event_id": "event-" + uuid.uuid4().hex, "sequence": len(events),
                     "agent_id": alias(row.get("agent_id", row.get("actor"))), "kind": kind, "text": text}
            if episode is not None:
                event["episode"] = episode
            round_index = row.get("round")
            if type(round_index) is int and round_index >= 0:
                event["round"] = round_index
            recipient = row.get("recipient")
            if recipient is not None:
                event["recipient_id"] = alias(recipient)
            events.append(event)
            mappings.append({"event_id": event["event_id"], "source_pointer": pointer, "fields": fields})
        corpus_id = "corpus-" + uuid.uuid4().hex
        public = {"version": 1, "corpus_id": corpus_id, "events": events}
        if len(_json(public)) > MAX_SOURCE_BYTES:
            raise ValueError("public corpus exceeds byte limit")
        operator = self.root / "operator" / corpus_id
        operator.mkdir(parents=True, mode=0o700)
        self.public_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        provenance = {"source_path": str(Path(path).absolute()), "source_sha256": sha,
                      "source_format": source_format, "policy": policy, "event_mappings": mappings,
                      "agent_aliases": aliases, "public_sha256": hashlib.sha256(_json(public)).hexdigest()}
        self._write(operator / "source.bin", content)
        self._write(operator / "provenance.json", _json(provenance))
        self._write(self.public_root / f"{corpus_id}.json", _json(public))
        return {"corpus_id": corpus_id, "event_count": len(events), "masked": True,
                "review_scope": "operator_reviewed_projection; semantic_decontamination_not_automated"}

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)

    def _load(self, corpus_id: str) -> list[dict[str, Any]]:
        if not isinstance(corpus_id, str) or not _ID.fullmatch(corpus_id):
            raise ValueError("invalid corpus ID")
        try:
            value = json.loads(_read_file(self.public_root / f"{corpus_id}.json"))
        except FileNotFoundError as exc:
            raise ValueError("unknown corpus ID") from exc
        if value.get("version") != 1 or value.get("corpus_id") != corpus_id:
            raise ValueError("invalid public corpus")
        allowed = {"event_id", "sequence", "agent_id", "kind", "text", "episode", "round", "recipient_id"}
        for event in value["events"]:
            if (not isinstance(event, dict) or set(event) - allowed
                    or not isinstance(event.get("event_id"), str) or not _EVENT_ID.fullmatch(event["event_id"])
                    or event.get("kind") not in PUBLIC_KINDS or not isinstance(event.get("text"), str)
                    or len(event["text"].encode()) > MAX_TEXT_BYTES):
                raise ValueError("invalid public event projection")
        return value["events"]

    @staticmethod
    def _page(corpus_id: str, events: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("offset must be nonnegative and limit between 1 and 50")
        page: list[dict[str, Any]] = []
        for event in events[offset:offset + limit]:
            if len(_json(page + [event])) > MAX_PAGE_BYTES:
                break
            page.append(event)
        next_offset = offset + len(page)
        return {"corpus_id": corpus_id, "events": page, "total": len(events), "offset": offset,
                "next_offset": next_offset if next_offset < len(events) else None}

    def search(self, corpus_id: str, query: str = "", *, agent_id: str | None = None,
               kind: str | None = None, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("query must be at most 1000 characters")
        if kind is not None and kind not in PUBLIC_KINDS:
            raise ValueError("unknown public event kind")
        matches = []
        for event in self._load(corpus_id):
            start = event["text"].casefold().find(query.casefold())
            if start >= 0 and (agent_id is None or event["agent_id"] == agent_id) and (
                    kind is None or event["kind"] == kind):
                matches.append({key: value for key, value in event.items() if key != "text"} | {
                    "snippet": event["text"][max(0, start - 60):start + 240], "read_tool": "read_events"})
        return self._page(corpus_id, matches, offset, limit)

    def read_events(self, corpus_id: str, *, event_ids: list[str] | None = None,
                    agent_id: str | None = None, kind: str | None = None,
                    offset: int = 0, limit: int = 20, text_offset: int = 0,
                    text_limit: int = 2000) -> dict[str, Any]:
        if (type(text_offset) is not int or text_offset < 0 or type(text_limit) is not int
                or not 1 <= text_limit <= 2000):
            raise ValueError("text_offset must be nonnegative and text_limit between 1 and 2000")
        events = self._load(corpus_id)
        if kind is not None and kind not in PUBLIC_KINDS:
            raise ValueError("unknown public event kind")
        if event_ids is not None:
            if (not isinstance(event_ids, list) or not 1 <= len(event_ids) <= 50
                    or any(not isinstance(value, str) or not _EVENT_ID.fullmatch(value) for value in event_ids)):
                raise ValueError("event_ids must contain between 1 and 50 event IDs")
            selected = set(event_ids)
            events = [event for event in events if event["event_id"] in selected]
            if selected - {event["event_id"] for event in events}:
                raise ValueError("event ID does not belong to this corpus")
        events = [event for event in events if (agent_id is None or event["agent_id"] == agent_id)
                  and (kind is None or event["kind"] == kind)]
        pages = []
        for event in events:
            text = event["text"]
            end = min(len(text), text_offset + text_limit)
            pages.append(event | {"text": text[text_offset:end], "text_offset": text_offset,
                                  "text_length": len(text), "next_text_offset": end if end < len(text) else None})
        return self._page(corpus_id, pages, offset, limit)

    def stats(self, corpus_id: str) -> dict[str, Any]:
        events = self._load(corpus_id)
        agents = Counter(event["agent_id"] for event in events if event["agent_id"] is not None)
        kinds = Counter(event["kind"] for event in events)
        edges = Counter((event["agent_id"], event["recipient_id"]) for event in events
                        if event.get("agent_id") is not None and event.get("recipient_id") is not None)
        if len(agents) > 100 or len(edges) > 100:
            raise ValueError("statistics exceed response bound; use filtered corpus searches")
        return {"corpus_id": corpus_id, "event_count": len(events), "by_agent": dict(agents),
                "by_kind": dict(kinds), "text_characters": sum(len(event["text"]) for event in events),
                "observed_edges": [{"from": sender, "to": receiver, "events": count}
                                   for (sender, receiver), count in sorted(edges.items())],
                "interpretation": "Recorded observations; communication edges do not establish causal influence."}
