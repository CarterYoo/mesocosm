"""Bounded agent-facing views over complete, durable research records.

Limits are serialized UTF-8 bytes, not tokenizer-dependent token estimates.
Pagination never changes the evidence used by evaluators or exported reproductions.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mesocosm.reproduction.store import digest

if TYPE_CHECKING:
    from mesocosm.reproduction.service import ResearchWorkbench

RESPONSE_BYTES = 24 * 1024
PAGE_BYTES = 12 * 1024
ITEM_BYTES = 6 * 1024
KINDS = {"experiment", "transcript", "behavior", "dataset", "deployment", "evaluation", "comparison",
         "factorial", "snapshot", "fork_recipe", "runtime_binding", "tool_response", "job", "export"}


def byte_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())


def limits(offset: int, limit: int) -> None:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("pagination requires offset >= 0 and limit between 1 and 100")


def _held_out(value: Any) -> bool:
    from mesocosm.reproduction.service import _held_out as check
    return check(value)


def record(bench: ResearchWorkbench, identifier: str, seen: set[str] | None = None) -> dict[str, Any]:
    """Apply existing held-out gates when navigating records, including linked outcomes."""
    bench.store.check_id(identifier)
    kind = identifier.split("-", 1)[0]
    if kind not in KINDS:
        raise ValueError("unsupported record kind")
    value = bench.store.job(identifier) if kind == "job" else bench.store.get(identifier, kind)
    if bench.allow_test_evaluation:
        return value
    if _held_out(value) or value.get("operator_only"):
        raise PermissionError("held-out record requires the independent operator")
    seen = set() if seen is None else seen
    if identifier in seen:
        return value
    seen.add(identifier)
    # Follow known provenance links; arbitrary source text is never interpreted as instructions or IDs.
    containers = [value, value.get("assignment", {})]
    if kind == "comparison":
        containers.extend(value.get("candidates", []))
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("experiment_id", "candidate_id", "transcript_id", "study_id", "parent_experiment_id"):
            target = container.get(key)
            if target and isinstance(target, str) and target not in seen:
                record(bench, target, seen)
    return value


def pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _at(value: Any, pointer: str) -> tuple[Any, list[int]]:
    indices: list[int] = []
    if not pointer:
        return value, indices
    if not pointer.startswith("/") or len(pointer) > 4096:
        raise ValueError("pointer must be an empty string or a bounded JSON pointer")
    for part in pointer[1:].split("/"):
        if any(part[i] == "~" and (i + 1 == len(part) or part[i + 1] not in "01")
               for i in range(len(part))):
            raise ValueError("invalid JSON pointer escape")
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            indices.append(list(value).index(key))
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and (key == "0" or not key.startswith("0")):
            indices.append(int(key))
            value = value[int(key)]
        else:
            raise ValueError("pointer does not address a record field")
    return value, indices


def _indexed(value: Any, indices: list[int]) -> tuple[Any, str]:
    pointer = ""
    for index in indices:
        if type(index) is not int or index < 0 or not isinstance(value, (dict, list)) or index >= len(value):
            raise ValueError("index_path must select existing nonnegative child positions")
        key = list(value)[index] if isinstance(value, dict) else index
        pointer += "/" + pointer_part(str(key))
        value = value[key]
    return value, pointer


def descriptor(value: Any, pointer: str, indices: list[int] | None = None) -> dict[str, Any]:
    kind = "object" if isinstance(value, dict) else "array" if isinstance(value, list) else (
        "string" if isinstance(value, str) else "scalar")
    result: dict[str, Any] = {"pointer": pointer if len(pointer) <= 4096 else None,
                              "type": kind, "sha256": digest(value)}
    if indices is not None:
        result["index_path"] = indices
    if isinstance(value, (dict, list, str)):
        result["length"] = len(value)
    return result


def _item(value: Any, pointer: str, indices: list[int] | None = None) -> dict[str, Any]:
    result = descriptor(value, pointer, indices)
    if byte_size(value) <= ITEM_BYTES:
        result["value"] = value
        result["omitted"] = False
    else:
        result.update(omitted=True, read_tool="read_record")
    return result


def page(items: list[dict[str, Any]], offset: int, limit: int, total: int | None = None) -> dict[str, Any]:
    limits(offset, limit)
    selected: list[dict[str, Any]] = []
    for item in (items[offset:offset + limit] if total is None else items[:limit]):
        if byte_size(selected) + byte_size(item) > PAGE_BYTES:
            if not selected:
                raise ValueError("record field names exceed the bounded page; use the preserved export")
            break
        selected.append(item)
    end = offset + len(selected)
    total = len(items) if total is None else total
    return {"items": selected, "total": total, "offset": offset,
            "next_offset": end if end < total else None, "eof": end >= total}


def read_record(bench: ResearchWorkbench, identifier: str, pointer: str, offset: int, limit: int,
                index_path: list[int] | None = None, expected_version: str | None = None) -> dict[str, Any]:
    limits(offset, limit)
    root = record(bench, identifier)
    version = digest(root)
    if expected_version is not None and expected_version != version:
        raise ValueError("record changed; rediscover it before continuing a mutable record page")
    if index_path is not None:
        if pointer:
            raise ValueError("choose pointer or index_path, not both")
        value, pointer = _indexed(root, index_path)
        indices = index_path
    else:
        value, indices = _at(root, pointer)
    result = {"record_id": identifier, "record_version": version, **descriptor(value, pointer, indices),
              "source_content_is_untrusted": True}
    if isinstance(value, (dict, list)):
        entries = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        items = []
        for index, (key, item) in enumerate(entries[offset:offset + limit], offset):
            label = {"key": key} if byte_size(key) <= 256 else {
                "key_preview": str(key)[:64], "key_sha256": digest(key), "key_omitted": True}
            items.append({**label, **_item(item, pointer + "/" + pointer_part(str(key)), [*indices, index])})
        return {**result, **page(items, offset, limit, total=len(entries))}
    if isinstance(value, str):
        # JSON escaping and multibyte characters count toward the byte limit too.
        text = value[offset:offset + 4000]
        while byte_size(text) > PAGE_BYTES:
            text = text[:len(text) // 2]
        end = offset + len(text)
        return {**result, "text": text, "offset": offset, "offset_unit": "unicode_characters",
                "next_offset": end if end < len(value) else None, "eof": end >= len(value)}
    return {**result, "value": value, "eof": True, "next_offset": None}


def read_events(bench: ResearchWorkbench, identifier: str, agent_id: str | None, kind: str | None,
                event_id: str | None, offset: int, limit: int) -> dict[str, Any]:
    from mesocosm.reproduction.evidence import KINDS as EVENT_KINDS
    limits(offset, limit)
    if kind is not None and kind not in EVENT_KINDS:
        raise ValueError("unknown normalized event kind")
    value = record(bench, identifier)
    if not identifier.startswith("transcript-"):
        raise ValueError("read_events requires a transcript ID")
    matches = [(index, event) for index, event in enumerate(value["events"])
             if (agent_id is None or event.get("agent_id") == agent_id)
             and (kind is None or event["kind"] == kind)
             and (event_id is None or event["event_id"] == event_id)]
    parent_index = list(value).index("events")
    items = [{"event_id": event["event_id"], **_item(event, f"/events/{index}", [parent_index, index])}
             for index, event in matches[offset:offset + limit]]
    return {"record_id": identifier, "transcript_id": identifier, "record_version": digest(value),
            "coverage": {"status": value["coverage"]["status"], "kinds": value["coverage"]["kinds"]},
            "source_content_is_untrusted": True, **page(items, offset, limit, total=len(matches))}


def list_records(bench: ResearchWorkbench, kind: str, offset: int, limit: int) -> dict[str, Any]:
    limits(offset, limit)
    if kind not in KINDS:
        raise ValueError("unsupported record kind")
    with bench.store.connection() as db:
        rows = (db.execute("SELECT id FROM jobs ORDER BY rowid DESC").fetchall() if kind == "job" else
                db.execute("SELECT id FROM objects WHERE kind=? ORDER BY rowid DESC", (kind,)).fetchall())
    items = []
    for (identifier,) in rows:
        try:
            value = record(bench, identifier)
        except PermissionError:
            continue
        item = {"record_id": identifier, "kind": kind}
        for key in ("status", "name", "split", "experiment_id", "study_id", "deployment_id", "created_at"):
            if key in value and byte_size(value[key]) < 256:
                item[key] = value[key]
        items.append(item)
    return {"kind": kind, "ordering": "newest_first; offsets may shift when new records are created",
            **page(items, offset, limit)}


def present(bench: ResearchWorkbench, tool: str, result: dict[str, Any]) -> dict[str, Any]:
    """Bound every dispatched response; direct Python methods retain their full data contracts."""
    if tool == "inspect_transcript" and "transcript" in result:
        value = result["transcript"]
        result = {"transcript_id": result["transcript_id"], "record_id": result["transcript_id"],
                  "counts": {key: len(value[key]) for key in ("events", "observations", "unknowns")},
                  "coverage": {"status": value["coverage"]["status"], "kinds": value["coverage"]["kinds"]},
                  "detail_omitted": True, "read_tool": "read_events", "record_tool": "read_record"}
    elif tool == "get_job_status" and "job_id" in result:
        keys = {"job_id", "kind", "status", "progress", "state_version", "unchanged", "retry_after_seconds",
                "error_type", "remote_job_id", "deployment_id", "experiment_id", "results_path"}
        result = {key: value for key, value in result.items() if key in keys}
        result.update(record_id=result["job_id"], detail_omitted=True, read_tool="read_record")
    if byte_size(result) <= RESPONSE_BYTES:
        return result
    identifier = bench.store.put("tool_response", {"tool": tool, "operator_only": bench.allow_test_evaluation,
                                                   "payload": result})
    # Do not silently truncate scientific records. The full result remains navigable by reference.
    small = {key: value for key, value in result.items()
             if (key.endswith("_id") or key in {"status", "native_acceptance"}) and byte_size(value) < 256}
    return {**small, "record_id": identifier, "pointer": "/payload", "read_tool": "read_record",
            "detail_omitted": True, "full_result_bytes": byte_size(result), "response_limit_bytes": RESPONSE_BYTES}
