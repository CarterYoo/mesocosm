"""Software integration tests, not empirical multi-agent behavior evidence."""

from __future__ import annotations

import json
import pathlib
import zipfile

import pytest
import yaml
from typer.testing import CliRunner

from mesocosm.cli.main import app
from mesocosm.reproduction.service import TOOLS, ResearchWorkbench
from mesocosm.reproduction.store import file_hash


@pytest.fixture
def bench(tmp_path):
    return ResearchWorkbench(tmp_path / "research", allowed_workers={
        "http://127.0.0.1:8088": "MESOCOSM_WORKER_TOKEN", "http://127.0.0.1:8099": "MESOCOSM_WORKER_TOKEN"})


def world_file(root: pathlib.Path) -> pathlib.Path:
    root.mkdir()
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Original software-test instruction")
    (task / "environment/Dockerfile").write_text("FROM debian:bookworm\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    world = {"name": "contract-test", "model": {"id": "local-test-model"},
             "gateway": {"url": "http://gateway.invalid", "api_key": "DO-NOT-EXPORT",
                         "upstream_key": "UPSTREAM-SECRET"},
             "agents": [{"id": f"agent-{i}"} for i in range(3)],
             "tasks": [{"path": "task"}], "services": {"enabled": False}}
    path = root / "world.yaml"
    path.write_text(yaml.safe_dump(world))
    return path


def candidate(bench, tmp_path, split="development"):
    definition = bench.define_behavior({"name": "execution", "stages": {
        "execution": {"event": {"kind": "action_execution", "where": {"data.tool": "bash"}}}}})
    path = world_file(tmp_path / "source")
    experiment = bench.configure_experiment(str(path), definition["behavior_id"], [], split=split)
    return experiment, path


def test_freeze_preserves_inputs_after_source_changes_and_portable_export(bench, tmp_path):
    experiment, path = candidate(bench, tmp_path)
    identifier = experiment["experiment_id"]
    original = (path.parent / "task/instruction.md").read_bytes()
    (path.parent / "task/instruction.md").write_text("Later modification")
    exported = bench.export_reproducer(identifier)
    destination = pathlib.Path(exported["path"])
    with zipfile.ZipFile(destination / "world.zip") as archive:
        name = next(name for name in archive.namelist() if name.endswith("instruction.md"))
        assert archive.read(name) == original
        assert all(b"DO-NOT-EXPORT" not in archive.read(name) for name in archive.namelist())
    assert file_hash(destination / "world.zip") == exported["archive_sha256"]
    assert bench.validate_experiment(identifier)["experiment_id"] == identifier
    assert (path.parent / "task/instruction.md").read_text() == "Later modification"
    assert bench.store.audit_records()


def test_configuration_receipt_integrity(bench, tmp_path):
    from mesocosm.reproduction.store import digest

    experiment, _ = candidate(bench, tmp_path)
    config = experiment["configuration"]
    assert config["configuration_sha256"] == digest({k: v for k, v in config.items() if k != "configuration_sha256"})


def test_archive_tampering_rejected_before_run_and_export(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    archive = bench.store.directory("experiments", experiment["experiment_id"]) / "world.zip"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        bench.export_reproducer(experiment["experiment_id"])


def test_evaluation_uses_frozen_rubric_and_keeps_unknown(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    transcript = tmp_path / "account.txt"
    transcript.write_text("Someone said they would execute bash.")
    imported = bench.inspect_transcript(str(transcript))
    result = bench.evaluate_candidate(experiment["experiment_id"], imported["transcript_id"])
    assert result["evaluation"]["status"] == "unknown"
    comparison = bench.compare_candidates([result["evaluation_id"], result["evaluation_id"]])
    assert comparison["comparison"]["duplicate_evaluations_ignored"] == 1
    assert comparison["comparison"]["candidates"][0]["attempts"] == 1


def test_test_partition_not_available_to_default_agent(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path, "test")
    source = tmp_path / "test.txt"
    source.write_text("Unknown evidence")
    transcript = bench.inspect_transcript(str(source))
    with pytest.raises(ValueError, match="held-out"):
        bench.evaluate_candidate(experiment["experiment_id"], transcript["transcript_id"])


@pytest.mark.asyncio
async def test_remote_uncertainty_keeps_one_identity_and_retry_is_explicit(bench, tmp_path, monkeypatch):
    experiment, _ = candidate(bench, tmp_path)
    calls = []

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, archive, job_id, **kwargs):
            calls.append((job_id, file_hash(archive)))
            if len(calls) == 1:
                raise ConnectionError("Response lost after acceptance")
            return {"job_id": job_id, "status": "queued", "digest": kwargs["digest"]}

    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    arguments = dict(experiment_id=experiment["experiment_id"], idempotency_key="run-once",
                     worker="http://127.0.0.1:8088")
    first = await bench.run_experiment(**arguments)
    second = await bench.run_experiment(**arguments)
    assert first == second and len(calls) == 1
    assert first["status"] == "submission_uncertain"
    resolved = await bench.get_job_status(first["job_id"], "resubmit")
    assert resolved["status"] == "queued"
    assert calls[0] == calls[1]
    with pytest.raises(ValueError, match="different inputs"):
        await bench.run_experiment(**(arguments | {"worker": "http://127.0.0.1:8099"}))


def dataset_file(tmp_path):
    path = tmp_path / "conversations.jsonl"
    rows = [{"id": f"r{i}", "session_id": f"s{i}", "task_id": f"t{i}", "label": "normal",
             "messages": [{"role": "user", "content": f"question{i}"}, {"role": "assistant", "content": "answer"}]}
            for i in range(3)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.asyncio
async def test_dataset_and_queued_job_idempotency_cancel(bench, tmp_path):
    source = dataset_file(tmp_path)
    prepared = bench.prepare_dataset([str(source)], {"split_assignments": {
        "r0": "train", "r1": "development", "r2": "test"}})
    args = {"dataset_id": prepared["dataset_id"], "config": {"base_model": str(tmp_path / "not-loaded"),
            "base_sha256": "0" * 64}, "idempotency_key": "train-once", "start": False}
    first = bench.submit_training(**args)
    assert bench.submit_training(**args) == first
    cancelled = await bench.get_job_status(first["job_id"], "cancel")
    assert cancelled["status"] == "cancelled"
    assert not pathlib.Path(cancelled["output_path"]).exists()
    with pytest.raises(ValueError):
        bench.submit_training(**(args | {"config": args["config"] | {"learning_rate": 0.1}}))


def test_cli_has_all_declared_tools_and_one_json_result(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["research", "tools", "--json"])
    assert result.exit_code == 0, result.output
    assert {row["name"] for row in json.loads(result.stdout)["tools"]} == set(TOOLS)
    request = tmp_path / "request.json"
    request.write_text("{}")
    call = runner.invoke(app, ["research", "call", "list_factors", "--request", str(request),
                               "--root", str(tmp_path / "research"), "--json"])
    assert call.exit_code == 0, call.output
    assert json.loads(call.stdout)["result"]["factors"]
    bad = runner.invoke(app, ["research", "call", "__dict__", "--request", str(request),
                              "--root", str(tmp_path / "research"), "--json"])
    assert bad.exit_code == 1
    assert json.loads(bad.stdout)["error_type"] == "ValueError"
