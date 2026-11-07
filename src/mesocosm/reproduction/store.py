"""Private immutable objects, idempotent requests and an independent audit chain.

This is a single trusted research operator's store, not a multi-tenant sandbox.
Content hashes establish consistency; they do not authenticate external evidence.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import time
from collections.abc import Iterator
from typing import Any

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*-[a-f0-9]{64}$")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Store:
    def __init__(self, root: str | pathlib.Path):
        given = pathlib.Path(root).absolute()
        if given.is_symlink():
            raise ValueError("research root cannot be a symlink")
        given.mkdir(parents=True, mode=0o700, exist_ok=True)
        if given.stat().st_mode & 0o077:
            raise ValueError("research root must be private (mode 0700)")
        self.root = given.resolve()
        self.path = self.root / "research.sqlite3"
        if self.path.is_symlink():
            raise ValueError("research database cannot be a symlink")
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    key TEXT PRIMARY KEY, digest TEXT NOT NULL, tool TEXT NOT NULL,
                    status TEXT NOT NULL, result TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY, body TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, body TEXT NOT NULL);
            """)
        self.path.chmod(0o600)

    @contextlib.contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA busy_timeout=30000")
            with db:
                yield db
        finally:
            db.close()

    def put(self, kind: str, value: dict[str, Any]) -> str:
        identifier = f"{kind}-{digest(value)}"
        self.check_id(identifier)
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO objects VALUES (?,?,?)", (identifier, kind, canonical(value)))
            recorded = db.execute("SELECT kind,body FROM objects WHERE id=?", (identifier,)).fetchone()
            if recorded != (kind, canonical(value)):
                raise ValueError("existing immutable object content hash mismatch")
        return identifier

    @staticmethod
    def check_id(identifier: str) -> None:
        if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
            raise ValueError("invalid research object ID")

    def get(self, identifier: str, kind: str | None = None) -> dict[str, Any]:
        self.check_id(identifier)
        with self.connection() as db:
            row = db.execute("SELECT kind, body FROM objects WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(f"unknown research object: {identifier}")
        actual_kind, body = row
        if kind is not None and actual_kind != kind:
            raise ValueError(f"expected {kind}, found {actual_kind}")
        value = json.loads(body)
        if identifier != f"{actual_kind}-{digest(value)}":
            raise ValueError("research object content hash mismatch")
        return value

    def begin(self, key: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(key, str) or not 1 <= len(key) <= 256:
            raise ValueError("idempotency_key must contain 1 to 256 characters")
        fingerprint = digest({"tool": tool, "arguments": arguments})
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,status,result,error FROM requests WHERE key=?", (key,)).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise ValueError("idempotency key was already used with different inputs")
                if row[1] == "completed":
                    return json.loads(row[2])
                raise ValueError(f"request is {row[1]}; inspect its job before any new attempt")
            db.execute("INSERT INTO requests VALUES (?,?,?,?,NULL,NULL)", (key, fingerprint, tool, "running"))
        self.audit("request.started", {"key": key, "tool": tool, "inputs_sha256": fingerprint})
        return None

    def finish(self, key: str, result: dict[str, Any] | None, error: str | None = None) -> None:
        with self.connection() as db:
            db.execute("UPDATE requests SET status=?,result=?,error=? WHERE key=?",
                       ("failed" if error else "completed", canonical(result) if result is not None else None,
                        error, key))
        self.audit("request.failed" if error else "request.completed",
                   {"key": key, "result_sha256": digest(result), "error_type": error})

    def audit(self, kind: str, data: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT sequence,sha256 FROM audit ORDER BY sequence DESC LIMIT 1").fetchone()
            sequence = previous[0] + 1 if previous else 1
            body = {"sequence": sequence, "at": time.time(), "kind": kind, "data": data,
                    "previous_sha256": previous[1] if previous else None}
            db.execute("INSERT INTO audit VALUES (?,?,?)", (sequence, canonical(body), digest(body)))

    def audit_records(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT sequence,body,sha256 FROM audit ORDER BY sequence").fetchall()
        previous = None
        result = []
        for index, (sequence, raw, fingerprint) in enumerate(rows, 1):
            body = json.loads(raw)
            if (sequence != index or body["sequence"] != index or digest(body) != fingerprint
                    or body["previous_sha256"] != previous):
                raise ValueError("research audit chain mismatch")
            result.append({**body, "sha256": fingerprint})
            previous = fingerprint
        return result

    def job(self, identifier: str) -> dict[str, Any]:
        self.check_id(identifier)
        with self.connection() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError("unknown research job")
        return json.loads(row[0])

    def save_job(self, identifier: str, value: dict[str, Any], *, new: bool = False) -> None:
        self.check_id(identifier)
        value = {**value, "job_id": identifier, "updated_at": time.time()}
        with self.connection() as db:
            if new:
                db.execute("INSERT INTO jobs VALUES (?,?)", (identifier, canonical(value)))
            else:
                if db.execute("UPDATE jobs SET body=? WHERE id=?", (canonical(value), identifier)).rowcount != 1:
                    raise KeyError("unknown research job")
        self.audit("job.status", {"job_id": identifier, "status": value["status"]})

    def directory(self, category: str, identifier: str) -> pathlib.Path:
        self.check_id(identifier)
        if category not in {"datasets", "jobs", "snapshots", "exports", "experiments", "transcripts"}:
            raise ValueError("invalid research storage category")
        result = self.root / category / identifier
        if (self.root / category).is_symlink() or result.is_symlink():
            raise ValueError("research storage directories cannot be symlinks")
        if not result.resolve().is_relative_to(self.root):
            raise ValueError("research directory escapes owned root")
        return result


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = canonical(value).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
