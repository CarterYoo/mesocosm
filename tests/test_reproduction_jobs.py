"""Bounded real-process queue checks, not native-agent or research acceptance."""

from __future__ import annotations

import fcntl
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

import pytest
from test_reproduction_training import local_training_assets as local_training_assets

from mesocosm.reproduction import jobs
from mesocosm.reproduction.service import ResearchWorkbench


def queued_job(tmp_path, *, max_seconds=2.0):
    bench = ResearchWorkbench(tmp_path / "research")
    source = tmp_path / "conversations.json"
    source.write_text(json.dumps([
        {"id": name, "session_id": name, "label": "normal", "messages": [
            {"role": "user", "content": "software fixture"}, {"role": "assistant", "content": "fixture answer"}]}
        for name in ("train", "development")]))
    dataset = bench.prepare_dataset([str(source)], {"split_assignments": {
        "train": "train", "development": "development"}})
    submitted = bench.submit_training(dataset["dataset_id"], {
        "base_model": str(tmp_path / "intentionally-not-loaded"), "base_sha256": "0" * 64,
        "max_seconds": max_seconds}, "one-job", start=False)
    return bench, bench.store.job(submitted["job_id"])


def child_env():
    env = dict(os.environ)
    src = str(pathlib.Path(jobs.__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
    return env


def test_actual_supervisor_deadline_terminates_its_child(tmp_path):
    bench, queued = queued_job(tmp_path, max_seconds=.01)
    claimed = jobs._claim(bench.store)
    assert claimed is not None and claimed["job_id"] == queued["job_id"]
    with (bench.store.root / "test-lease").open("w") as lease:
        started = time.monotonic()
        result = jobs._run_attempt(bench.store, claimed, lease.fileno())
    assert result["status"] == "budget_exhausted"
    assert 0 < result["elapsed_seconds"] < 3
    assert time.monotonic() - started < 3
    assert result["cost"]["wallclock_seconds"] == result["elapsed_seconds"]
    assert not pathlib.Path(queued["output_path"]).exists()


def test_actual_child_kernel_alarm_works_without_supervisor(tmp_path):
    bench, queued = queued_job(tmp_path)
    claimed = jobs._claim(bench.store)
    assert claimed is not None
    script = """
import pathlib, time, sys
from mesocosm.reproduction import jobs, training
def stall(*args, **kwargs):
    pathlib.Path(sys.argv[3]).write_text('entered bounded fixture')
    time.sleep(10)
training.run_training = stall
jobs._child(pathlib.Path(sys.argv[1]), sys.argv[2], .2)
"""
    marker = tmp_path / "entered"
    started = time.monotonic()
    process = subprocess.run([sys.executable, "-c", script, str(bench.store.root), queued["job_id"], str(marker)],
                             env=child_env(), capture_output=True, timeout=3, check=False)
    assert process.returncode == -signal.SIGALRM, process.stderr.decode()
    assert marker.read_text() == "entered bounded fixture"
    assert time.monotonic() - started < 3
    assert not (bench.store.directory("jobs", queued["job_id"]) / "child-result.json").exists()


@pytest.mark.asyncio
async def test_queued_cancel_is_never_claimed_or_executed(tmp_path, monkeypatch):
    bench, queued = queued_job(tmp_path)
    cancelled = await bench.get_job_status(queued["job_id"], "cancel")
    assert cancelled["status"] == "cancelled"
    monkeypatch.setattr(jobs, "_run_attempt", lambda *args: pytest.fail("cancelled job must not execute"))
    jobs.run_queue(bench.store.root)
    assert jobs._claim(bench.store) is None
    assert not pathlib.Path(queued["output_path"]).exists()


def test_frozen_dataset_change_fails_before_training(tmp_path, monkeypatch):
    from mesocosm.reproduction import training

    bench, queued = queued_job(tmp_path)
    assert jobs._claim(bench.store) is not None
    examples = pathlib.Path(queued["dataset_path"]) / "examples.jsonl"
    examples.write_text(examples.read_text() + " ")
    # Keep this unit-level ordering test inside pytest; the actual kernel alarm is
    # exercised above in a disposable subprocess, never on the test runner itself.
    monkeypatch.setattr(jobs.signal, "signal", lambda *args: None)
    monkeypatch.setattr(jobs.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(training, "run_training", lambda *args, **kwargs: pytest.fail("changed data reached trainer"))
    with pytest.raises(ValueError, match="dataset examples changed"):
        jobs._child(bench.store.root, queued["job_id"], 1)


def test_elapsed_floor_exhaustion_prevents_new_process(tmp_path, monkeypatch):
    bench, queued = queued_job(tmp_path, max_seconds=2)
    claimed = jobs._claim(bench.store)
    assert claimed is not None
    claimed["elapsed_floor_seconds"] = 2
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("budget must not be renewed"))
    with (bench.store.root / "test-lease").open("w") as lease:
        result = jobs._run_attempt(bench.store, claimed, lease.fileno())
    assert result["status"] == "budget_exhausted"
    assert not pathlib.Path(queued["output_path"]).exists()


def test_resume_preserves_original_attempt_and_accumulates_measured_elapsed(tmp_path):
    bench, original = queued_job(tmp_path, max_seconds=10)
    output = pathlib.Path(original["output_path"])
    output.mkdir()
    (output / "retained.txt").write_text("original failed-attempt evidence")
    prior = {**original, "status": "failed", "elapsed_floor_seconds": 2.5, "elapsed_seconds": 1.25}
    bench.store.save_job(prior["job_id"], prior)
    resumed = bench.submit_training(prior["dataset_id"], {**prior["config"], "resume": True},
                                    "resume-once", start=False, resume_job_id=prior["job_id"])
    job = bench.store.job(resumed["job_id"])
    assert job["elapsed_floor_seconds"] == 3.75
    assert job["output_path"] != prior["output_path"]
    assert (pathlib.Path(job["output_path"]) / "retained.txt").read_bytes() == (output / "retained.txt").read_bytes()
    assert bench.store.job(prior["job_id"])["status"] == "failed"
    with pytest.raises(ValueError, match="configuration"):
        bench.submit_training(prior["dataset_id"], {**prior["config"], "resume": True, "max_seconds": 20},
                              "budget-extension-denied", start=False, resume_job_id=prior["job_id"])


def test_two_actual_queue_processes_claim_one_job_once(tmp_path):
    bench, queued = queued_job(tmp_path)
    marker = tmp_path / "executions.jsonl"
    script = """
import pathlib, time, sys, json
from mesocosm.reproduction import jobs
def probe(store, job, lease_fd):
    with pathlib.Path(sys.argv[2]).open('a') as stream:
        stream.write(json.dumps({'job_id': job['job_id']}) + '\\n')
    time.sleep(.15)
    return {**job, 'status': 'completed', 'elapsed_seconds': .15, 'result': {'fixture': True}}
jobs._run_attempt = probe
jobs.run_queue(pathlib.Path(sys.argv[1]))
"""
    processes = [subprocess.Popen([sys.executable, "-c", script, str(bench.store.root), str(marker)],
                                 env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=4)
            assert process.returncode == 0, (stdout + stderr).decode()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
    assert [json.loads(line)["job_id"] for line in marker.read_text().splitlines()] == [queued["job_id"]]
    assert bench.store.job(queued["job_id"])["status"] == "completed"


def test_worker_recovery_never_repeats_orphaned_update(tmp_path, monkeypatch):
    bench, queued = queued_job(tmp_path)
    assert jobs._claim(bench.store) is not None
    monkeypatch.setattr(jobs, "_run_attempt", lambda *args: pytest.fail("orphan must not retry automatically"))
    jobs.run_queue(bench.store.root)
    recovered = bench.store.job(queued["job_id"])
    assert recovered["status"] == "interrupted" and recovered["error_type"] == "worker_lost"
    assert recovered["elapsed_seconds"] >= 0
    assert "conservative" in recovered["elapsed_measurement"]


def test_queue_lease_is_inherited_by_actual_child(tmp_path):
    lease_path = tmp_path / "lease"
    with lease_path.open("w") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        script = "import time; time.sleep(.2)"
        process = subprocess.Popen([sys.executable, "-c", script], pass_fds=(lease.fileno(),))
        # Closing the parent's descriptor leaves the inherited open-file lease held.
        lease.close()
        with lease_path.open("r+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process.wait(timeout=2)
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_actual_tiny_cpu_sft_completes_through_durable_queue(tmp_path, local_training_assets):
    dataset, config, _, _ = local_training_assets
    bench = ResearchWorkbench(tmp_path / "research")
    manifest = json.loads((dataset / "manifest.json").read_text())
    identifier = bench.store.put("dataset", manifest)
    shutil.copytree(dataset, bench.store.directory("datasets", identifier))
    submitted = bench.submit_training(identifier, {**config, "max_steps": 1}, "actual-cpu-fixture", start=False)
    started = time.monotonic()
    jobs.run_queue(bench.store.root)
    result = bench.store.job(submitted["job_id"])
    assert result["status"] == "completed", result
    assert result["result"]["optimizer_steps"] == 1
    assert result["result"]["losses"][0]["weight_delta_l2"] > 0
    assert result["result"]["adapter_sha256"]
    assert not result["result"]["behavioral_improvement_demonstrated"]
    assert result["elapsed_seconds"] >= result["result"]["elapsed_seconds"]
    assert time.monotonic() - started < config["max_seconds"]
