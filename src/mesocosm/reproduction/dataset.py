"""Frozen observation/action datasets from supplied conversations or native gateway records.

Selection is not generation. Labels are research annotations, never behavior findings.
Only preceding messages become observations; metadata, judgments and later events do not.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mesocosm.learning.checkpoints import write_json

Label = Literal["normal", "target", "recovery"]
Split = Literal["train", "development", "test"]


def _default_labels() -> list[Label]:
    return ["normal", "target", "recovery"]


def _default_splits() -> dict[Split, float]:
    return {"train": .8, "development": .1, "test": .1}


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    labels: list[Label] = Field(default_factory=_default_labels, min_length=1)
    label_overrides: dict[str, Label] = Field(default_factory=dict)
    agent_ids: list[str] = Field(default_factory=list)
    reasoning_policy: Literal["preserve", "drop"] = "preserve"
    max_context_tokens: int = Field(default=4096, ge=2)
    overflow: Literal["reject", "drop"] = "reject"
    seed: int = 0
    split_ratios: dict[Split, float] = Field(default_factory=_default_splits)
    split_assignments: dict[str, Split] = Field(default_factory=dict)
    action_ranges: dict[str, tuple[int, int]] = Field(default_factory=dict)
    max_per_label: dict[Label, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self) -> DatasetConfig:
        if set(self.split_ratios) != {"train", "development", "test"} or any(
            value < 0 for value in self.split_ratios.values()
        ) or abs(sum(self.split_ratios.values()) - 1) > 1e-9:
            raise ValueError("split_ratios must contain train/development/test and sum to one")
        if any(start < 0 or end <= start for start, end in self.action_ranges.values()):
            raise ValueError("action_ranges use nonnegative half-open assistant-turn indices")
        if any(type(value) is not int or value < 0 for value in self.max_per_label.values()):
            raise ValueError("max_per_label values must be nonnegative integers")
        return self


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _read(path: pathlib.Path, raw: bytes | None = None) -> list[dict[str, Any]]:
    text = (path.read_bytes() if raw is None else raw).decode("utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        if not isinstance(value, (dict, list)):
            raise ValueError("dataset source must contain conversation objects")
        rows = value if isinstance(value, list) else value.get("conversations", [value])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("dataset source must contain conversation objects")
    return rows


def _gateway(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("event") != "chat_completion":
        return None
    response = row.get("response") or {}
    if (row.get("status") != 200 or response.get("finish_reason") not in {"stop", "tool_calls", "length"}
            or response.get("error") or response.get("stream_errors") or response.get("unparsed_lines")
            or (row.get("stream") and response.get("stream_complete") is not True)
            or response.get("choice_indices", [0]) != [0]):
        raise ValueError(f"incomplete/ambiguous native completion: {row.get('request_id')}")
    request = row.get("request") or {}
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("native request has no recorded messages")
    context = row.get("rollout") or {}
    action = {"role": "assistant", "content": response.get("content", "")}
    if response.get("tool_calls"):
        action["tool_calls"] = response["tool_calls"]
    if response.get("reasoning"):
        action["reasoning_content"] = response["reasoning"]
    # A run fallback groups conservatively; it does not fabricate a harness session identity.
    return {"id": row["request_id"], "incident_id": row.get("run"), "task_id": context.get("task"),
            "session_id": context.get("session_id"), "agent_id": row.get("agent"),
            "messages": [*messages, action], "tools": request.get("tools"), "only_last_action": True}


def _message(value: Any, reasoning: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError("each message requires a supported role")
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    if reasoning == "preserve":
        allowed.update({"reasoning", "reasoning_content"})
    message = {key: item for key, item in value.items() if key in allowed}
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ValueError("multimodal message content is unsupported by this SFT dataset")
    return message


def _splits(rows: list[dict[str, Any]], spec: DatasetConfig) -> dict[str, str]:
    """Connected aliases prevent a shared incident, task OR session crossing splits."""
    parents = {row["id"]: row["id"] for row in rows}

    def find(key: str) -> str:
        while parents[key] != key:
            key = parents[key]
        return key

    owners: dict[tuple[str, str], str] = {}
    for row in rows:
        aliases = [(key, str(row[key])) for key in ("incident_id", "task_id", "session_id") if row.get(key)]
        if not aliases:
            raise ValueError(f"record {row['id']} requires a recorded incident, task or session identity")
        for alias in aliases:
            if alias in owners:
                parents[find(row["id"])] = find(owners[alias])
            owners[alias] = row["id"]
    groups: dict[str, list[str]] = {}
    for key in parents:
        groups.setdefault(find(key), []).append(key)
    result: dict[str, str] = {}
    for members in groups.values():
        assigned = {spec.split_assignments[key] for key in members if key in spec.split_assignments}
        if len(assigned) > 1:
            raise ValueError("split leakage: connected incident/task/session has conflicting assignments")
        if assigned:
            split = next(iter(assigned))
        else:
            value = int(digest([spec.seed, sorted(members)]), 16) / 2**256
            split = "test" if value >= spec.split_ratios["train"] + spec.split_ratios["development"] else (
                "development" if value >= spec.split_ratios["train"] else "train")
        result.update({key: split for key in members})
    return result


def prepare_dataset(sources: list[pathlib.Path], output: pathlib.Path, config: dict[str, Any]) -> dict[str, Any]:
    """Write a new immutable dataset, preserving source bytes and selection decisions.

    Native run bundle directories use native/gateway/records.jsonl. Annotate their
    request IDs with label_overrides; no success score becomes a behavior label.
    Context token limits are enforced by the pinned tokenizer during training.
    """
    spec = DatasetConfig.model_validate(config)
    if not sources:
        raise ValueError("at least one source is required")
    output = pathlib.Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    frozen, rows = [], []
    for source in sources:
        path = pathlib.Path(source).resolve()
        if path.is_dir():
            path = path / "native/gateway/records.jsonl"
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        frozen.append((path, raw, source_hash))
        for index, original in enumerate(_read(path, raw)):
            row = _gateway(original) if "event" in original and "request_id" in original else original
            if row is None:
                continue
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("every conversation requires a nonempty stable id")
            rows.append({**row, "source_sha256": source_hash, "source_row": index})
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate source record id; refusing to duplicate actions")
    for mapping in (spec.label_overrides, spec.split_assignments, spec.action_ranges):
        if set(mapping) - set(identifiers):
            raise ValueError("selection references a missing source record")
    splits = _splits(rows, spec)
    examples, decisions = [], []
    counts: dict[str, int] = {}
    for row in sorted(rows, key=lambda value: digest([spec.seed, value["id"]])):
        label = spec.label_overrides.get(row["id"], row.get("label"))
        if label not in {"normal", "target", "recovery"}:
            raise ValueError(f"explicit behavior-class annotation required for {row['id']}")
        reason = None
        if label not in spec.labels or (spec.agent_ids and row.get("agent_id") not in spec.agent_ids):
            reason = "selection"
        if counts.get(label, 0) >= spec.max_per_label.get(label, len(rows)):
            reason = "class_quota"
        if reason:
            decisions.append({"record_id": row["id"], "selected": False, "reason": reason})
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("conversation requires recorded messages")
        messages = [_message(message, spec.reasoning_policy) for message in messages]
        action_indices = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        if row.get("only_last_action"):
            action_indices = action_indices[-1:]
        start, end = spec.action_ranges.get(row["id"], (0, len(action_indices)))
        if end > len(action_indices):
            raise ValueError("action range exceeds recorded assistant turns")
        for number in range(start, end):
            index = action_indices[number]
            if index == 0:
                raise ValueError("assistant action requires a preceding observation")
            example = {"id": f"{row['id']}:{number}", "record_id": row["id"], "label": label,
                       "split": splits[row["id"]], "observation": messages[:index], "action": messages[index],
                       "tools": row.get("tools"), "agent_id": row.get("agent_id"),
                       "identities": {key: row.get(key) for key in ("incident_id", "task_id", "session_id")},
                       "source_sha256": row["source_sha256"], "source_row": row["source_row"],
                       "source_message_index": index}
            examples.append(example)
        counts[label] = counts.get(label, 0) + 1
        decisions.append({"record_id": row["id"], "selected": True, "actions": end - start})
    if not examples:
        raise ValueError("selection contains no assistant actions")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    lineage = []
    for index, (path, raw, source_hash) in enumerate(frozen):
        relative = f"sources/{index:04d}{path.suffix}"
        (output / relative).write_bytes(raw)
        lineage.append({"original_path": str(path), "path": relative, "sha256": source_hash, "bytes": len(raw)})
    data_path = output / "examples.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in examples))
    manifest = {"schema_version": 1, "kind": "observation_action", "sources": lineage,
                "examples_sha256": file_sha256(data_path), "config": spec.model_dump(mode="json"),
                "selection": decisions, "counts": {split: sum(row["split"] == split for row in examples)
                                                    for split in ("train", "development", "test")},
                "class_counts": counts, "class_count_unit": "source_records", "labels_are_annotations": True,
                "loss_mask": "latest_assistant_action_only", "context_limit_enforced_at": "pinned_tokenization",
                "leakage_checks": ["preceding_observations_only", "connected_incident_task_session_splits",
                                   "metadata_not_in_model_input"],
                "semantic_hidden_answer_detection": "unsupported"}
    manifest["dataset_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return {**manifest, "dataset_path": str(output)}


def load_dataset(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = pathlib.Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("dataset_id") != digest({key: value for key, value in manifest.items() if key != "dataset_id"}):
        raise ValueError("dataset manifest identity mismatch")
    if file_sha256(path / "examples.jsonl") != manifest["examples_sha256"]:
        raise ValueError("dataset examples changed")
    for source in manifest["sources"]:
        source_path = (path / source["path"]).resolve()
        if not source_path.is_relative_to(path) or file_sha256(source_path) != source["sha256"]:
            raise ValueError("frozen dataset source changed")
    return manifest, _read(path / "examples.jsonl")
