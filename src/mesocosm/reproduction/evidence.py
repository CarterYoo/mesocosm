"""Normalize preserved incident evidence without treating its content as instructions.

A source hash authenticates imported bytes, not the truth or completeness of their
claims. Native tool records establish recorded execution, never an inferred state
change. Coverage is unknown unless the source explicitly declares its scope.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from mesocosm.core.records import RunBundle

NORMALIZER_VERSION = "mesocosm-evidence-v1"
MAX_SOURCE_FILE_BYTES = 256 << 20
KINDS = frozenset({"speech", "action_request", "action_execution", "tool_result", "state_change",
                   "intervention", "reward", "outcome", "model_call", "context", "observation"})


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read_source_bytes(path: Path) -> bytes:
    if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
        raise ValueError("transcript source exceeds the per-file byte limit")
    with path.open("rb") as stream:
        content = stream.read(MAX_SOURCE_FILE_BYTES + 1)
    if len(content) > MAX_SOURCE_FILE_BYTES:
        raise ValueError("transcript source grew beyond the per-file byte limit")
    return content


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _kind(value: str) -> str:
    if value in KINDS:
        return value
    if value in {"harness.reply", "harness.message", "harness.transcript", "message", "dialogue"}:
        return "speech"
    if value in {"tool.call", "tool_call", "tool.request"}:
        return "action_request"
    if value in {"tool.executed", "tool.started", "action.executed"}:
        return "action_execution"
    if value in {"tool.result", "tool_result", "tool"}:
        return "tool_result"
    if value in {"file.changed", "memory.changed", "state.diff", "judge.state_check", "verifier.world_state"}:
        return "state_change"
    if value.startswith(("conditions.", "intervention.", "run.paused", "run.resumed")):
        return "intervention"
    if value in {"learning.reward", "episode.feedback", "learning.feedback_delivered"}:
        return "reward"
    if value.startswith("verifier."):
        return "outcome"
    if value == "gateway.chat_completion":
        return "model_call"
    return "observation"


def _event(raw: dict[str, Any], location: dict[str, Any], source_sha: str,
           *, parent: dict[str, Any] | None = None) -> dict[str, Any]:
    parent = parent or {}
    identity = dict(_object(raw.get("identity")))
    fallback_kind = "tool_result" if raw.get("role") == "tool" else "message" if "role" in raw else "observation"
    source_kind = str(raw.get("kind") or raw.get("type") or fallback_kind)
    data = raw.get("data") if isinstance(raw.get("data"), dict) else dict(raw)
    timestamp = raw.get("ts", raw.get("timestamp"))
    if isinstance(timestamp, float) and not math.isfinite(timestamp):
        raise ValueError("event timestamp must be finite")
    source_id = raw.get("event_id") or raw.get("id")
    session = identity.get("session_id") or raw.get("session_id") or parent.get("session_id")
    # Never silently upgrade a data.session_id into an authoritative identity binding.
    event = {
        "event_id": "evidence_" + digest([source_sha, location]),
        "source_event_id": source_id,
        "kind": _kind(source_kind), "source_kind": source_kind,
        "ts": timestamp, "time_basis": raw.get("time_basis", "source_unspecified"),
        "agent_id": raw.get("agent_id", parent.get("agent_id")),
        "session_id": session, "identity": identity,
        "display_name": raw.get("display_name"), "principal": raw.get("principal"),
        "run_id": raw.get("run_id", parent.get("run_id")),
        "request_id": raw.get("request_id"), "links": raw.get("links", []),
        "data": data, "source_location": location, "confidence": raw.get("confidence"),
        "evidence_basis": "recorded_source", "source_sha256": source_sha,
    }
    return event


def _native_messages(payload: dict[str, Any], path: str, sha: str,
                     parent: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, message in enumerate(payload.get("messages", [])):
        if not isinstance(message, dict):
            raise ValueError("native messages must be objects")
        info = _object(message.get("info"))
        session = info.get("sessionID") or payload.get("session_id")
        base = {**parent, "session_id": session}
        for part_index, part in enumerate(message.get("parts", [])):
            if not isinstance(part, dict):
                raise ValueError("native message parts must be objects")
            location = {"path": path, "pointer": f"/messages/{index}/parts/{part_index}"}
            common = {"message_id": info.get("id"), "role": info.get("role"),
                      "part_id": part.get("id"), "synthetic": part.get("synthetic", False),
                      "ignored": part.get("ignored", False)}
            if part.get("type") == "tool":
                state = _object(part.get("state"))
                time = _object(state.get("time"))
                data = {**common, "tool": part.get("tool"), "call_id": part.get("callID"),
                        "arguments": state.get("input"), "status": state.get("status")}
                variants = [("action_request", None, data)]
                if state.get("status") in {"running", "completed"}:
                    variants.append(("action_execution", time.get("start"), data))
                if state.get("status") in {"completed", "error"}:
                    variants.append(("tool_result", time.get("end"),
                                     {**data, "output": state.get("output"), "error": state.get("error")}))
                for kind, timestamp, value in variants:
                    event = _event({"kind": kind, "ts": timestamp, "time_basis": "opencode_unix_ms",
                                    "data": value}, {**location, "component": kind}, sha, parent=base)
                    event["evidence_basis"] = "harness_tool_record"
                    result.append(event)
            else:
                kind = "speech" if part.get("type") == "text" else "context"
                result.append(_event({"kind": kind, "data": {**common, **part},
                                      "ts": _object(info.get("time")).get("created"),
                                      "time_basis": "opencode_unix_ms"}, location, sha, parent=base))
    return result


def _decode_file(path: Path, label: str) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any], str]:
    content = _read_source_bytes(path)
    sha = hashlib.sha256(content).hexdigest()
    text = content.decode("utf-8")
    metadata: dict[str, Any] = {}
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if path.suffix.lower() == ".jsonl":
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label}:{number}: expected an object")
                rows.append((value, {"path": label, "line": number}))
    elif path.suffix.lower() == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            metadata = value
            key = "events" if "events" in value else "messages" if "messages" in value else None
            entries = value[key] if key else [value]
            prefix = f"/{key}" if key else ""
        else:
            entries, prefix = value, ""
        if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
            raise ValueError("JSON transcript must contain a list of event/message objects")
        rows = [(row, {"path": label, "pointer": f"{prefix}/{index}"}) for index, row in enumerate(entries)]
    else:
        # Speaker-looking strings and commands in prose remain speech with unknown identity.
        rows = [({"kind": "speech", "data": {"text": line}}, {"path": label, "line": number})
                for number, line in enumerate(text.splitlines(), 1) if line.strip()]
    return rows, metadata, sha


def inspect_transcript(path: Path) -> dict[str, Any]:
    """Read a RunBundle or UTF-8 JSON/JSONL/text transcript; never execute its text.

    The normalized result is evaluator-side evidence, including any supplied split
    metadata. It must not be exposed as model context or used as a training export.
    """
    path = Path(path).resolve()
    is_bundle = path.is_dir()
    if is_bundle:
        RunBundle.open(path)  # Reuse the platform's bundle recognition contract.
    files: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    bundle_run_id = path.name if is_bundle else None
    inputs = [path / "events.jsonl", path / "receipts.jsonl"] if is_bundle else [path]
    for file in inputs:
        if not file.exists():
            continue
        label = file.relative_to(path).as_posix() if is_bundle else file.name
        rows, meta, sha = _decode_file(file, label)
        files.append({"path": label, "sha256": sha, "bytes": file.stat().st_size})
        if not is_bundle:
            metadata = meta
        for raw, location in rows:
            event = _event(raw, location, sha, parent={"run_id": bundle_run_id})
            if label == "receipts.jsonl":
                event["kind"] = "intervention"
            events.append(event)
            # OpenAI-format structured calls are requests, including calls never executed.
            if raw.get("role") == "assistant" and isinstance(raw.get("tool_calls"), list):
                for call_index, call in enumerate(raw["tool_calls"]):
                    if not isinstance(call, dict):
                        raise ValueError("tool_calls entries must be objects")
                    events.append(_event({"kind": "action_request", "data": call,
                                          "agent_id": event["agent_id"], "session_id": event["session_id"]},
                                         {**location, "tool_call_index": call_index}, sha))
            if is_bundle and raw.get("kind") == "harness.turn":
                data = _object(raw.get("data"))
                relative = data.get("path")
                if not isinstance(relative, str) or not relative:
                    raise ValueError("harness.turn has no native evidence path")
                native = (path / relative).resolve()
                if not native.is_relative_to(path):
                    raise ValueError("native evidence path escapes the bundle")
                content = _read_source_bytes(native)
                native_sha = hashlib.sha256(content).hexdigest()
                if not data.get("sha256") or data["sha256"] != native_sha:
                    raise ValueError("native turn hash mismatch or missing digest")
                files.append({"path": relative, "sha256": native_sha, "bytes": len(content)})
                parent = {"run_id": raw.get("run_id"), "agent_id": raw.get("agent_id"),
                          "session_id": data.get("session_id")}
                events.extend(_native_messages(json.loads(content), relative, native_sha, parent))
    if not is_bundle and metadata.get("messages") and isinstance(metadata["messages"][0].get("info"), dict):
        events = _native_messages(metadata, path.name, files[0]["sha256"], {})
    if is_bundle:
        for name in ("summary.json", "lineage.json", "evidence-coverage.json"):
            file = path / name
            if file.exists():
                content = _read_source_bytes(file)
                files.append({"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
                metadata[name.removesuffix(".json")] = json.loads(content)
    # Repeated references to the same native part are one observation, not extra trials.
    unique_events: dict[str, dict[str, Any]] = {}
    for event in events:
        previous = unique_events.get(event["event_id"])
        if previous is not None and previous != event:
            raise ValueError("conflicting identity for repeated native evidence")
        unique_events[event["event_id"]] = event
    events = list(unique_events.values())
    # A hash covers the inventory, not an arbitrary absolute path or import time.
    files = list({(f["path"], f["sha256"]): f for f in files}.values())
    source_sha = digest(sorted(files, key=lambda f: f["path"]))
    declared = _object(metadata.get("source"))
    incident_id = metadata.get("incident_id") or declared.get("incident_id") or declared.get("id") or source_sha
    coverage = _object(metadata.get("coverage", metadata.get("evidence-coverage")))
    status = coverage.get("status", "unknown")
    kinds = coverage.get("kinds", [])
    if status not in {"complete", "partial", "unknown"} or not isinstance(kinds, list) or set(kinds) - KINDS:
        raise ValueError("invalid evidence coverage declaration")
    if status == "complete" and (not kinds or not coverage.get("evidence")):
        raise ValueError("complete coverage requires explicit kinds and evidence")
    unknowns = [{"event_id": event["event_id"], "fields": [key for key in ("ts", "agent_id", "session_id")
                                                          if event.get(key) is None]}
                for event in events if any(event.get(key) is None for key in ("ts", "agent_id", "session_id"))]
    result = {
        "schema_version": 1, "normalizer_version": NORMALIZER_VERSION,
        "source": {"id": str(incident_id), "sha256": source_sha, "files": files,
                   "format": "mesocosm-run-bundle" if is_bundle else path.suffix.lower().lstrip(".") or "text"},
        "incident_id": str(incident_id), "run_id": bundle_run_id or metadata.get("run_id"),
        "session_ids": sorted({str(e["session_id"]) for e in events if e["session_id"] is not None}),
        "events": events, "coverage": {"status": status, "kinds": kinds, "evidence": coverage.get("evidence")},
        "observations": [{"event_id": e["event_id"], "source_location": e["source_location"],
                          "basis": e["evidence_basis"]} for e in events],
        "assumptions": metadata.get("assumptions", []),
        "unknowns": unknowns + metadata.get("unknowns", []), "metadata": {
            k: v for k, v in metadata.items() if k not in {"events", "messages", "assumptions", "unknowns"}},
    }
    result["transcript_sha256"] = digest(result)
    return result


def validate_transcript(transcript: dict[str, Any]) -> None:
    """Reject mutated normalized evidence before evaluating or deriving a dataset."""
    body = {k: v for k, v in transcript.items() if k != "transcript_sha256"}
    if (transcript.get("normalizer_version") != NORMALIZER_VERSION
            or transcript.get("transcript_sha256") != digest(body)):
        raise ValueError("normalized transcript digest/version mismatch")
