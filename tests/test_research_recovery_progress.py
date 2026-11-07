"""Software crash-state and bounded polling checks; no model or worker is started."""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from mesocosm.learning.checkpoints import write_json as atomic_json
from mesocosm.reproduction import progress
from mesocosm.reproduction.service import ResearchWorkbench


@pytest.fixture
def world_job(tmp_path, monkeypatch):
    bench = ResearchWorkbench(tmp_path / "research", allowed_workers={
        "http://127.0.0.1:8088": "MESOCOSM_WORKER_TOKEN"})
    experiment = bench.store.put("experiment", {"package": {"digest": "f" * 64}})
    identifier = "job-" + "a" * 64
    record = {"kind": "world", "status": "completed", "experiment_id": experiment,
              "worker": "http://127.0.0.1:8088/", "token_env": "MESOCOSM_WORKER_TOKEN",
              "remote_job_id": "a" * 32}
    bench.store.save_job(identifier, record, new=True)
    calls = []

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def status(self, job_id):
            return {"status": "completed", "job_id": job_id, "digest": "f" * 64}

        async def fetch(self, job_id, destination):
            calls.append(job_id)
            return result_files(destination, job_id)

        async def submit(self, *args, **kwargs):
            pytest.fail("retrieval must never resubmit an experiment")

    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    return bench, identifier, calls


def result_files(destination, job_id, input_digest="f" * 64):
    destination.mkdir(parents=True)
    (destination / "job.json").write_text(json.dumps({"job_id": job_id, "digest": input_digest}))
    (destination / "result.txt").write_text("retained software fixture")
    metadata = {"files": {path.name: {"type": "file", "size": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in destination.iterdir()}}
    metadata["digest"] = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (destination / "package.json").write_text(json.dumps(metadata))
    return {"job_id": job_id, "digest": metadata["digest"], "destination": str(destination.resolve()),
            "sensitive": True, "redacted": False}


@pytest.mark.asyncio
async def test_download_and_repeated_fetch_use_one_remote_job(world_job):
    bench, identifier, calls = world_job
    first = await bench.get_job_status(identifier, "fetch")
    second = await bench.get_job_status(identifier, "fetch")
    assert calls == ["a" * 32]
    assert first["retrieval"] == second["retrieval"]
    assert bench._verified_results(second).is_dir()
    assert (bench.store.directory("jobs", identifier) / "retrieval.json").is_file()


@pytest.mark.asyncio
async def test_orphaned_download_is_preserved_and_refetched_not_trusted(world_job):
    bench, identifier, calls = world_job
    directory = bench.store.directory("jobs", identifier)
    destination = directory / "results"
    result_files(destination, "a" * 32)
    (destination / "result.txt").write_text("unadmitted tampered bytes")
    recovered = await bench.get_job_status(identifier, "fetch")
    assert calls == ["a" * 32]
    preserved = list(directory.glob("results.unadmitted-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "result.txt").read_text() == "unadmitted tampered bytes"
    assert (bench._verified_results(recovered) / "result.txt").read_text() == "retained software fixture"


@pytest.mark.asyncio
async def test_journal_recovers_crash_before_admission_without_redownload(world_job, monkeypatch):
    bench, identifier, calls = world_job
    original = pathlib.Path.rename

    def interrupt(path, destination):
        if path.name == "results.pending":
            raise RuntimeError("software crash after receipt journal")
        return original(path, destination)

    monkeypatch.setattr(pathlib.Path, "rename", interrupt)
    with pytest.raises(RuntimeError, match="crash"):
        await bench.get_job_status(identifier, "fetch")
    assert "retrieval" not in bench.store.job(identifier)
    monkeypatch.setattr(pathlib.Path, "rename", original)
    recovered = await bench.get_job_status(identifier, "fetch")
    assert calls == ["a" * 32]
    assert bench._verified_results(recovered).is_dir()


@pytest.mark.asyncio
async def test_journal_recovers_crash_before_job_receipt_without_redownload(world_job, monkeypatch):
    bench, identifier, calls = world_job
    original = bench.store.save_job

    def interrupt(job_id, value, **kwargs):
        if value.get("retrieval"):
            raise RuntimeError("software crash before job receipt")
        return original(job_id, value, **kwargs)

    monkeypatch.setattr(bench.store, "save_job", interrupt)
    with pytest.raises(RuntimeError, match="crash"):
        await bench.get_job_status(identifier, "fetch")
    assert "retrieval" not in bench.store.job(identifier)
    monkeypatch.setattr(bench.store, "save_job", original)
    recovered = await bench.get_job_status(identifier, "fetch")
    assert calls == ["a" * 32]
    assert bench._verified_results(recovered).is_dir()


@pytest.mark.asyncio
async def test_retained_receipt_does_not_admit_tampered_results(world_job):
    bench, identifier, calls = world_job
    result = await bench.get_job_status(identifier, "fetch")
    (pathlib.Path(result["results_path"]) / "result.txt").write_text("tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        await bench.get_job_status(identifier, "fetch")
    assert calls == ["a" * 32]


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["foreign_job", "foreign_experiment", "input_digest", "bytes"])
async def test_journal_recovery_rechecks_receipt_and_original_inputs(world_job, corruption):
    bench, identifier, calls = world_job
    job = bench.store.job(identifier)
    directory = bench.store.directory("jobs", identifier)
    destination = directory / "results"
    receipt = result_files(destination, job["remote_job_id"], "e" * 64 if corruption == "input_digest" else "f" * 64)
    if corruption == "foreign_job":
        receipt["job_id"] = "b" * 32
    if corruption == "bytes":
        (destination / "result.txt").write_text("modified after download")
    journal = {"schema_version": 1, "experiment_id": "foreign" if corruption == "foreign_experiment"
               else job["experiment_id"], "receipt": receipt}
    atomic_json(directory / "retrieval.json", journal)
    with pytest.raises(ValueError):
        await bench.get_job_status(identifier, "fetch")
    assert calls == []
    assert "retrieval" not in bench.store.job(identifier)
    assert destination.exists()


@pytest.fixture
def training_job(tmp_path):
    bench = ResearchWorkbench(tmp_path / "research")
    identifier = "job-" + "c" * 64
    output = bench.store.directory("jobs", identifier) / "output"
    output.mkdir(parents=True)
    bench.store.save_job(identifier, {"kind": "training", "status": "running", "output_path": str(output),
                                    "config": {"max_steps": 100}, "started_at": 1.0}, new=True)
    return bench, identifier, output


@pytest.mark.asyncio
async def test_training_progress_reads_actual_output_and_versions_changes(training_job):
    bench, identifier, output = training_job
    initial = await bench.get_job_status(identifier)
    assert initial["progress"] == {"availability": "not_yet_written"}
    atomic_json(output / "result.json", {"status": "running", "optimizer_steps": 100,
                                       "train_tokens": 400, "checkpoint": str(output / "checkpoint-00000100"),
                                       "losses": [{"loss": 5.0}] * 1000})
    updated = await bench.get_job_status(identifier, since_version=initial["state_version"])
    assert not updated["unchanged"]
    assert updated["progress"]["optimizer_steps"] == 100
    assert updated["progress"]["checkpoint_id"] == "checkpoint-00000100"
    assert "losses" not in updated["progress"]
    assert updated["status"] == "running" and updated["config"] == {"max_steps": 100}
    unchanged = await bench.get_job_status(identifier, since_version=updated["state_version"])
    assert unchanged == {"job_id": identifier, "status": "running", "state_version": updated["state_version"],
                         "unchanged": True, "retry_after_seconds": 5}
    assert len(json.dumps(unchanged)) < 300
    job = bench.store.job(identifier)
    bench.store.save_job(identifier, job)  # updated_at alone is not progress.
    assert (await bench.get_job_status(identifier, since_version=updated["state_version"]))["unchanged"]


@pytest.mark.asyncio
async def test_final_output_progress_does_not_claim_job_completion(training_job):
    bench, identifier, output = training_job
    atomic_json(output / "result.json", {"status": "completed", "optimizer_steps": 100,
                                       "initial_development_loss": 3.0, "development_loss": 2.0})
    result = await bench.get_job_status(identifier)
    assert result["status"] == "running"  # The supervisor has not committed its final receipt.
    assert result["progress"]["training_status"] == "completed"
    assert result["progress"]["behavioral_evaluation"] == "unmeasured"


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", ["not json", "[]", '{"optimizer_steps": -1}',
                                      '{"optimizer_steps": true}', '{"development_loss": NaN}',
                                      '{"checkpoint": "/outside/checkpoint-00000001"}'])
async def test_invalid_progress_is_explicitly_unavailable(training_job, snapshot):
    bench, identifier, output = training_job
    (output / "result.json").write_text(snapshot)
    result = await bench.get_job_status(identifier)
    assert result["progress"] == {"availability": "unavailable", "reason": "invalid_progress_snapshot"}


@pytest.mark.asyncio
async def test_progress_read_is_size_bounded(training_job, monkeypatch):
    bench, identifier, output = training_job
    monkeypatch.setattr(progress, "MAX_PROGRESS_BYTES", 64)
    (output / "result.json").write_text(" " * 65)
    assert (await bench.get_job_status(identifier))["progress"] == {
        "availability": "unavailable", "reason": "progress_size_limit"}


@pytest.mark.asyncio
async def test_symlink_progress_is_not_followed(training_job, tmp_path):
    bench, identifier, output = training_job
    external = tmp_path / "private.json"
    external.write_text('{"optimizer_steps": 100}')
    (output / "result.json").symlink_to(external)
    assert (await bench.get_job_status(identifier))["progress"]["reason"] == "unsafe_progress_path"


@pytest.mark.asyncio
async def test_poll_version_is_only_for_inspection(training_job):
    bench, identifier, _ = training_job
    with pytest.raises(ValueError, match="since_version"):
        await bench.get_job_status(identifier, "cancel", "a" * 64)
    with pytest.raises(ValueError, match="since_version"):
        await bench.get_job_status(identifier, since_version="unknown")
    assert bench.store.job(identifier)["status"] == "running"
