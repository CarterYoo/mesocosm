"""Contract tests use labeled fixture transcripts, not behavioral research data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mesocosm.reproduction.dataset import load_dataset, prepare_dataset


def conversation(identity="one", **extra):
    return {"id": identity, "session_id": identity, "label": "normal", "agent_id": "agent-1", "messages": [
        {"role": "user", "content": "inspect the recorded file"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function",
         "function": {"name": "read", "arguments": '{"path":"a"}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "original result"},
        {"role": "assistant", "content": "observed result", "reasoning_content": "recorded rationale"},
        {"role": "user", "content": "future secret"}], "hidden_answer": "never a model input", **extra}


def write_source(tmp_path, rows):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(rows))
    return path


def test_observation_action_lineage_no_future_context_and_preserved_tool_result(tmp_path):
    source = write_source(tmp_path, [conversation()])
    receipt = prepare_dataset([source], tmp_path / "dataset", {"split_assignments": {"one": "train"}})
    manifest, rows = load_dataset(tmp_path / "dataset")
    assert len(rows) == 2 and rows[0]["action"]["tool_calls"][0]["id"] == "call-1"
    assert rows[1]["observation"][-1]["role"] == "tool"
    assert rows[1]["observation"][-1]["content"] == "original result"
    assert rows[1]["action"]["reasoning_content"] == "recorded rationale"
    assert "future secret" not in json.dumps(rows) and "never a model input" not in json.dumps(rows)
    assert (tmp_path / "dataset" / manifest["sources"][0]["path"]).read_bytes() == source.read_bytes()
    assert receipt["counts"] == {"train": 2, "development": 0, "test": 0}
    with pytest.raises(FileExistsError):
        prepare_dataset([source], tmp_path / "dataset", {})


def test_connected_task_aliases_cannot_cross_splits(tmp_path):
    source = write_source(tmp_path, [conversation("one", task_id="task"),
                                     conversation("two", task_id="task", incident_id="incident"),
                                     conversation("three", incident_id="incident")])
    with pytest.raises(ValueError, match="split leakage"):
        prepare_dataset([source], tmp_path / "dataset", {"split_assignments": {"one": "train", "three": "test"}})
    assert not (tmp_path / "dataset").exists()


def test_annotations_are_explicit_and_duplicate_actions_are_rejected(tmp_path):
    row = conversation()
    del row["label"]
    source = write_source(tmp_path, [row])
    with pytest.raises(ValueError, match="annotation"):
        prepare_dataset([source], tmp_path / "dataset", {})
    prepare_dataset([source], tmp_path / "dataset", {"label_overrides": {"one": "recovery"}})
    with pytest.raises(ValueError, match="duplicate"):
        prepare_dataset([source, source], tmp_path / "duplicate", {"label_overrides": {"one": "normal"}})


def test_declared_selection_and_reasoning_removal(tmp_path):
    source = write_source(tmp_path, [conversation(), conversation("two", agent_id="agent-2", label="target")])
    receipt = prepare_dataset([source], tmp_path / "dataset", {
        "agent_ids": ["agent-1"], "action_ranges": {"one": [1, 2]}, "reasoning_policy": "drop"})
    _, rows = load_dataset(tmp_path / "dataset")
    assert len(rows) == 1 and "reasoning_content" not in rows[0]["action"]
    assert rows[0]["source_message_index"] == 3
    assert sum(not decision["selected"] for decision in receipt["selection"]) == 1


@pytest.mark.parametrize("target", ["examples.jsonl", "sources/0000.json", "manifest.json"])
def test_frozen_dataset_tampering_fails(tmp_path, target):
    source = write_source(tmp_path, [conversation()])
    prepare_dataset([source], tmp_path / "dataset", {})
    path = tmp_path / "dataset" / target
    if target == "manifest.json":
        value = json.loads(path.read_text())
        value["config"]["seed"] = 4
        path.write_text(json.dumps(value))
    else:
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed|mismatch"):
        load_dataset(tmp_path / "dataset")


def test_native_gateway_preserves_actual_request_without_retraining_history(tmp_path):
    native = tmp_path / "run/native/gateway"
    native.mkdir(parents=True)
    record = {"event": "chat_completion", "request_id": "req", "run": "run", "agent": "a", "status": 200,
              "request": {"messages": [{"role": "user", "content": "first"},
                                        {"role": "assistant", "content": "old action"},
                                        {"role": "user", "content": "next"}]},
              "response": {"content": "new action", "finish_reason": "stop", "choice_indices": [0]}}
    (native / "records.jsonl").write_text(json.dumps(record) + "\n")
    prepare_dataset([tmp_path / "run"], tmp_path / "dataset", {"label_overrides": {"req": "normal"}})
    _, rows = load_dataset(tmp_path / "dataset")
    assert len(rows) == 1 and rows[0]["action"]["content"] == "new action"
    assert rows[0]["identities"]["session_id"] is None
    assert rows[0]["identities"]["incident_id"] == "run"


def test_bad_native_completion_is_not_silently_admitted(tmp_path):
    source = write_source(tmp_path, [{"event": "chat_completion", "request_id": "broken", "status": 500}])
    with pytest.raises(ValueError, match="incomplete"):
        prepare_dataset([source], tmp_path / "dataset", {})


def test_source_is_parsed_from_the_exact_bytes_frozen_in_lineage(tmp_path, monkeypatch):
    source = write_source(tmp_path, [conversation()])
    original = Path.read_bytes

    def changing_source(path):
        data = original(path)
        if path == source:
            changed = [conversation()]
            changed[0]["messages"][3]["content"] = "changed live source"
            source.write_text(json.dumps(changed))
        return data

    monkeypatch.setattr(Path, "read_bytes", changing_source)
    prepare_dataset([source], tmp_path / "dataset", {})
    _, rows = load_dataset(tmp_path / "dataset")
    assert rows[1]["action"]["content"] == "observed result"
    assert "changed live source" in source.read_text()
