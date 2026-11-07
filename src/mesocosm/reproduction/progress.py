"""Bounded progress views; trainer output is not behavioral evaluation evidence."""

from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

from mesocosm.reproduction.store import Store, digest

MAX_PROGRESS_BYTES = 4 * 1024 * 1024
COUNTERS = ("optimizer_steps", "train_tokens", "train_tokens_reserved", "eval_tokens_reserved",
            "eval_tokens_completed")
MEASUREMENTS = ("elapsed_seconds", "initial_development_loss", "development_loss")


def training_progress(store: Store, job: dict[str, Any]) -> dict[str, Any]:
    """Read one atomic trainer snapshot without exposing config, examples or loss history."""
    output = store.directory("jobs", job["job_id"]) / "output"
    path = output / "result.json"
    if output.is_symlink() or path.is_symlink() or pathlib.Path(job["output_path"]).resolve() != output.resolve():
        return {"availability": "unavailable", "reason": "unsafe_progress_path"}
    if not path.exists():
        return {"availability": "not_yet_written"}
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_PROGRESS_BYTES + 1)
        if len(content) > MAX_PROGRESS_BYTES:
            return {"availability": "unavailable", "reason": "progress_size_limit"}
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("invalid progress")
        progress: dict[str, Any] = {"availability": "available", "source": "training_output_result",
                                    "behavioral_evaluation": "unmeasured"}
        for key in COUNTERS:
            if key in value:
                if type(value[key]) is not int or value[key] < 0:
                    raise ValueError("invalid counter")
                progress[key] = value[key]
        for key in MEASUREMENTS:
            if key in value:
                number = value[key]
                if number is not None and (type(number) not in {int, float} or not math.isfinite(number)):
                    raise ValueError("invalid measurement")
                progress[key] = number
        if "status" in value:
            if value["status"] not in {"running", "completed", "cancelled", "budget_exhausted",
                                       "failed", "interrupted"}:
                raise ValueError("invalid training status")
            progress["training_status"] = value["status"]
        if value.get("checkpoint") is not None:
            checkpoint = pathlib.Path(value["checkpoint"])
            if (checkpoint.parent.resolve() != output.resolve()
                    or not re.fullmatch(r"checkpoint-[0-9]{8,20}", checkpoint.name)):
                raise ValueError("checkpoint outside training output")
            progress["checkpoint_id"] = checkpoint.name
        return progress
    except (OSError, ValueError, TypeError, OverflowError):
        return {"availability": "unavailable", "reason": "invalid_progress_snapshot"}


def status_view(store: Store, job: dict[str, Any], result: dict[str, Any],
                since_version: str | None) -> dict[str, Any]:
    """Keep direct Python fields while adding a compact stable polling contract."""
    progress = training_progress(store, job) if job.get("kind") == "training" else None
    # updated_at changes on remote polling even without a meaningful state change.
    state = {key: value for key, value in result.items() if key != "updated_at"}
    if progress is not None:
        state["progress"] = progress
    version = digest(state)
    running = job.get("status") in {"queued", "running", "submitting", "activating"}
    envelope = {"job_id": job["job_id"], "status": job["status"], "state_version": version,
                "unchanged": since_version == version, "retry_after_seconds": 5 if running else None}
    if since_version == version:
        return envelope
    return {**result, **({"progress": progress} if progress is not None else {}), **envelope}
