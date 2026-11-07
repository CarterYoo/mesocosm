"""Durable local training queue with bounded owned child processes.

The parent supervises cancellation and deadlines. The child inherits the queue
lease and arms a kernel alarm: losing the parent neither duplicates a job nor
removes its wall-clock limit. No cloud allocation or automatic retry occurs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

from mesocosm.reproduction.store import Store, write_json


def start_worker(store: Store) -> None:
    log = store.root / "training-worker.log"
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab") as stream:
        subprocess.Popen([sys.executable, "-m", "mesocosm.reproduction.jobs", "--root", str(store.root)],
                         stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)


def _claim(store: Store) -> dict[str, Any] | None:
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT id,body FROM jobs ORDER BY rowid").fetchall()
        for identifier, raw in rows:
            job = json.loads(raw)
            if job.get("kind") != "training" or job["status"] != "queued":
                continue
            job.update(status="running", started_at=time.time(), worker_pid=os.getpid())
            db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(job), identifier))
            return job
    return None


def _child(root: pathlib.Path, identifier: str, seconds: float) -> None:
    # Default SIGALRM terminates even while Python is inside a native forward.
    # Do not replace this kernel-level deadline with a Python exception handler.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    store = Store(root)
    job = store.job(identifier)
    if job.get("kind") != "training" or job.get("status") != "running":
        raise ValueError("child requires a claimed training job")
    directory = store.directory("jobs", identifier)
    from mesocosm.reproduction.dataset import load_dataset
    from mesocosm.reproduction.training import run_training, training_source_pins

    if job.get("training_sources") != training_source_pins():
        raise ValueError("queued training source code changed")
    if load_dataset(pathlib.Path(job["dataset_path"]))[0] != store.get(job["dataset_id"], "dataset"):
        raise ValueError("queued dataset identity changed")
    result = run_training(pathlib.Path(job["dataset_path"]), pathlib.Path(job["output_path"]), job["config"],
                          cancel_path=directory / "cancel.requested",
                          elapsed_floor_seconds=job.get("elapsed_floor_seconds", 0.0))
    write_json(directory / "child-result.json", result)
    signal.setitimer(signal.ITIMER_REAL, 0)


def _stop_owned(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _run_attempt(store: Store, job: dict[str, Any], lease_fd: int) -> dict[str, Any]:
    directory = store.directory("jobs", job["job_id"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    remaining = job["config"]["max_seconds"] - job.get("elapsed_floor_seconds", 0.0)
    started = time.monotonic()
    status, result, error_type = "failed", None, None
    process: subprocess.Popen[Any] | None = None
    if remaining <= 0:
        status = "budget_exhausted"
    elif (directory / "cancel.requested").exists():
        status = "cancelled"
    else:
        log = directory / "trainer.log"
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "ab") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-m", "mesocosm.reproduction.jobs", "--root", str(store.root),
                     "--job", job["job_id"], "--seconds", str(remaining)],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True,
                    pass_fds=(lease_fd,))
                while process.poll() is None:
                    if (directory / "cancel.requested").exists():
                        status = "cancelled"
                        _stop_owned(process)
                        break
                    if time.monotonic() - started >= remaining:
                        status = "budget_exhausted"
                        _stop_owned(process)
                        break
                    time.sleep(min(0.05, max(0.001, remaining - (time.monotonic() - started))))
                if status not in {"cancelled", "budget_exhausted"}:
                    if process.returncode == -signal.SIGALRM:
                        status = "budget_exhausted"
                    elif process.returncode == 0:
                        result = json.loads((directory / "child-result.json").read_text())
                        status = result.get("status", "failed")
                        if status not in {"completed", "budget_exhausted", "cancelled"}:
                            status = "failed"
                    else:
                        error_type = "trainer_process_failed"
        except KeyboardInterrupt:
            status, error_type = "interrupted", "operator_interrupt"
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            if process is not None:
                _stop_owned(process)
    elapsed = time.monotonic() - started
    return {**job, "status": status, "result": result, "error_type": error_type,
            "finished_at": time.time(), "elapsed_seconds": elapsed,
            "cost": {"wallclock_seconds": elapsed, "currency_amount": None,
                     "scope": "this attempt including setup/evaluation; no inferred cloud invoice"}}


def run_queue(root: pathlib.Path) -> None:
    store = Store(root)
    lease_path = store.root / "training-worker.lock"
    fd = os.open(lease_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        # Children inherit this lease, so recovery starts only after an orphaned
        # bounded child exits. Never automatically rerun interrupted updates.
        with store.connection() as db:
            previous = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
        for job in previous:
            if job.get("kind") == "training" and job["status"] == "running":
                recovered = {**job, "status": "interrupted", "error_type": "worker_lost",
                             "finished_at": time.time(),
                             "elapsed_seconds": max(0.0, time.time() - job["started_at"]),
                             "elapsed_measurement": "conservative elapsed since persisted start"}
                store.save_job(job["job_id"], recovered)
        while (job := _claim(store)) is not None:
            def terminate(_signum: int, _frame: Any) -> None:
                raise KeyboardInterrupt
            old = signal.signal(signal.SIGTERM, terminate)
            try:
                record = _run_attempt(store, job, lease.fileno())
                write_json(store.directory("jobs", job["job_id"]) / "result.json", record)
                store.save_job(job["job_id"], record)
            finally:
                signal.signal(signal.SIGTERM, old)
            if record["status"] == "interrupted":
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.job is not None:
        if args.seconds is None or not 0 < args.seconds < float("inf"):
            parser.error("child requires a finite positive deadline")
        _child(args.root, args.job, args.seconds)
    else:
        run_queue(args.root)
