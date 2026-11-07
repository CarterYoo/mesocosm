"""Recover result downloads without trusting unadmitted files or resubmitting work."""

from __future__ import annotations

import json
import os
import pathlib
import re
import uuid
from collections.abc import Callable
from typing import Any

from mesocosm.reproduction.store import Store, write_json


async def fetch_results(store: Store, job: dict[str, Any], client: Any,
                        verify: Callable[[dict[str, Any]], pathlib.Path]) -> None:
    """Journal a verified download receipt before admitting its final directory."""
    directory = store.directory("jobs", job["job_id"])
    destination = directory / "results"
    staging = directory / "results.pending"
    journal = directory / "retrieval.json"
    if any(path.is_symlink() for path in (destination, staging, journal)):
        raise ValueError("unsafe result retrieval path")
    if job.get("retrieval"):
        job["results_path"] = str(destination)
        verify(job)
        return
    if journal.exists():
        with journal.open("rb") as stream:
            content = stream.read(64 * 1024 + 1)
        if len(content) > 64 * 1024:
            raise ValueError("result retrieval journal exceeds size limit")
        value = json.loads(content)
        receipt = value.get("receipt") if isinstance(value, dict) else None
        if (not isinstance(receipt, dict) or value.get("schema_version") != 1
                or value.get("experiment_id") != job["experiment_id"]
                or receipt.get("job_id") != job["remote_job_id"]
                or not isinstance(receipt.get("digest"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", receipt["digest"])
                or receipt.get("destination") != str(destination.resolve())):
            raise ValueError("result retrieval journal identity mismatch")
        if destination.exists() and staging.exists():
            raise ValueError("ambiguous result retrieval state; preserve both directories for inspection")
        if not destination.exists():
            if not staging.is_dir():
                raise ValueError("journaled result files are missing")
            staging.rename(destination)
        job.update(retrieval=receipt, results_path=str(destination))
        verify(job)
        return
    # A directory without a durable download receipt is never admitted from its
    # self-reported hashes. Keep it and download the SAME remote job anew.
    for path in (destination, staging):
        if path.exists():
            preserved = directory / (path.name + ".unadmitted-" + uuid.uuid4().hex)
            path.rename(preserved)
            store.audit("job.results_preserved", {"job_id": job["job_id"], "path": preserved.name,
                                                  "reason": "missing_retrieval_receipt"})
    receipt = await client.fetch(job["remote_job_id"], staging)
    if (not isinstance(receipt, dict) or receipt.get("job_id") != job["remote_job_id"]
            or not isinstance(receipt.get("digest"), str) or not re.fullmatch(r"[a-f0-9]{64}", receipt["digest"])):
        raise ValueError("download receipt has invalid job identity or digest")
    receipt = {**receipt, "destination": str(destination.resolve())}
    value = {"schema_version": 1, "experiment_id": job["experiment_id"], "receipt": receipt}
    temporary = directory / ("retrieval-" + uuid.uuid4().hex + ".tmp")
    write_json(temporary, value)
    temporary.replace(journal)
    # Persist the directory entry as well as the receipt bytes before renaming.
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    staging.rename(destination)
    job.update(retrieval=receipt, results_path=str(destination))
    verify(job)
