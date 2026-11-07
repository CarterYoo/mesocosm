"""Bounded import and local materialization of partial Mesocosm snapshots.

Only captured filesystem state is restored. No harness process, in-flight request,
optimizer or RNG continuation is claimed and no agent/model is launched here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from mesocosm.core.records import RunBundle
from mesocosm.core.snapshot import NOT_CAPTURED, SERVICES_ITEMS, LocalWorldState, restore_snapshot
from mesocosm.reproduction.evidence import digest

MAX_FILES = 10_000
MAX_BYTES = 1 << 30
MAX_METADATA_BYTES = 16 << 20
_METADATA = {"manifest.json", "coverage.json", "queue.json", "files.json"}
_EXTRA_NOT_CAPTURED = ["optimizer and training RNG state", "model/scheduler/environment RNG continuation",
                       "exact model-token or live-process replay",
                       "filesystem ownership, ACLs, extended attributes and exact permission modes"]


def _relative(name: str) -> str:
    value = PurePosixPath(name)
    if not name or "\\" in name or value.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"unsafe snapshot path: {name}")
    return value.as_posix()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot must be a real directory")
    files: dict[str, dict[str, Any]] = {}
    total = 0
    entries = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in [*dirs, *names]:
            path = Path(directory) / name
            relative = _relative(path.relative_to(root).as_posix())
            info = path.lstat()
            entries += 1
            if entries > MAX_FILES:
                raise ValueError("snapshot file-count limit exceeded")
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError(f"snapshot links and special files are unsupported: {relative}")
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ValueError("snapshot hardlinked files are unsupported")
                total += info.st_size
                if total > MAX_BYTES:
                    raise ValueError("snapshot byte limit exceeded")
                files[relative] = {"sha256": _hash(path), "bytes": info.st_size}
    return files


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("snapshot metadata size limit exceeded")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"snapshot metadata must be an object: {path.name}")
    return value


def _verify_payload(root: Path) -> dict[str, Any]:
    inventory = _inventory(root)
    if not _METADATA <= inventory.keys():
        raise ValueError("incomplete snapshot metadata")
    manifest = _json(root / "manifest.json")
    coverage = _json(root / "coverage.json")
    expected = _json(root / "files.json")
    _json(root / "queue.json")
    agents = {agent["id"] for agent in manifest.get("manifest", {}).get("agents", [])}
    if not agents or any(not isinstance(agent, str) or _relative(agent) != agent or "/" in agent for agent in agents):
        raise ValueError("snapshot manifest needs safe agent identifiers")
    actual: dict[str, str] = {}
    total_expanded = 0
    members = 0
    archived_agents = set()
    for relative, entry in inventory.items():
        parts = PurePosixPath(relative).parts
        if relative in _METADATA:
            continue
        if parts[0] == "workspaces" and len(parts) == 2 and relative.endswith(".tar.gz"):
            agent_id = parts[1][:-len(".tar.gz")]
            if agent_id not in agents:
                raise ValueError("snapshot archive names an unknown agent")
            archived_agents.add(agent_id)
            seen: set[str] = set()
            with tarfile.open(root / relative, "r|gz") as archive:
                for member in archive:
                    name = _relative(member.name.rstrip("/") if member.isdir() else member.name)
                    members += 1
                    if member.size < 0:
                        raise ValueError("negative archive member size")
                    total_expanded += member.size
                    if members > MAX_FILES or total_expanded > MAX_BYTES:
                        raise ValueError("snapshot expanded archive limit exceeded")
                    if name in seen:
                        raise ValueError("duplicate archive member")
                    seen.add(name)
                    if not member.isdir() and not member.isfile():
                        raise ValueError("archive links and special members are unsupported")
                    if member.sparse is not None:
                        raise ValueError("sparse archive members are unsupported")
                    if member.isfile():
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise ValueError("unreadable archive member")
                        member_hash = hashlib.sha256()
                        while block := stream.read(1 << 20):
                            member_hash.update(block)
                        actual[f"workspaces/{agent_id}/{name}"] = member_hash.hexdigest()
        elif parts[0] in {"services", "shared"} and len(parts) >= 2:
            if parts[0] == "services" and parts[1] not in SERVICES_ITEMS:
                raise ValueError("snapshot contains an unsupported service path")
            total_expanded += entry["bytes"]
            if total_expanded > MAX_BYTES:
                raise ValueError("snapshot expanded byte limit exceeded")
            actual[relative] = entry["sha256"]
        else:
            raise ValueError(f"unexpected snapshot content: {relative}")
    for name in expected:
        _relative(name)
    if actual != expected:
        raise ValueError("snapshot captured-state hashes do not match files.json")
    if set(NOT_CAPTURED) - set(coverage.get("not_captured", [])):
        raise ValueError("snapshot removed required coverage limitations")
    declared_agents = {item.get("agent_id") for item in coverage.get("captured", [])
                       if item.get("component") == "workspace"}
    if archived_agents != declared_agents:
        raise ValueError("snapshot workspace coverage differs from actual archives")
    if coverage.get("files") != len(actual):
        raise ValueError("snapshot coverage file count mismatch")
    return {"inventory": inventory, "coverage": coverage, "manifest": manifest,
            "expanded_bytes": total_expanded, "verified_files": len(actual)}


def _closed_parent(bundle_root: Path, payload: dict[str, Any], *, stored: bool = False) -> dict[str, dict[str, Any]]:
    bundle = RunBundle.open(bundle_root)
    coverage = payload["coverage"]
    if not stored and bundle.run_id != coverage.get("parent_run_id"):
        raise ValueError("snapshot parent run does not match supplied original bundle")
    paths = [bundle_root / name for name in ("manifest.json", "summary.json", "events.jsonl")]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("original run must have manifest, summary and event evidence")
    if _json(paths[0]) != payload["manifest"]:
        raise ValueError("snapshot manifest differs from its parent")
    summary = _json(bundle_root / "summary.json")
    if summary.get("status") not in {"ok", "stopped", "failed", "error", "cancelled"}:
        raise ValueError("parent run has no recognized terminal summary")
    if (bundle_root / "events.jsonl").stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("parent event metadata exceeds bounded import size")
    events = bundle.events()
    if any(event.run_id != coverage["parent_run_id"] for event in events):
        raise ValueError("parent event run identity mismatch")
    cleanup = [event for event in events if event.kind == "runtime.cleanup"]
    if not cleanup or cleanup[-1].data.get("status") != "complete":
        raise ValueError("parent requires original successful runtime cleanup evidence")
    closure = cleanup[-1]
    if type(coverage.get("created")) not in {int, float} or coverage["created"] < closure.ts:
        raise ValueError("snapshot predates original runtime cleanup")
    closure_index = events.index(closure)
    if any(event.kind.startswith(("harness.", "task.", "run.resumed")) for event in events[closure_index + 1:]):
        raise ValueError("parent shows execution after cleanup")
    return {path.name: {"sha256": _hash(path), "bytes": path.stat().st_size} for path in paths}


def freeze_snapshot(source_snapshot: Path, destination: Path, *, parent_bundle: Path) -> dict[str, Any]:
    """Pin an existing snapshot and its closed parent evidence into a fresh directory."""
    source_snapshot, destination, parent_bundle = Path(source_snapshot), Path(destination), Path(parent_bundle)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("snapshot destination must not exist")
    payload = _verify_payload(source_snapshot)
    parent_inventory = _closed_parent(parent_bundle, payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-import-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "frozen"
        shutil.copytree(source_snapshot, staging / "snapshot", symlinks=True)
        (staging / "parent").mkdir()
        for name in parent_inventory:
            shutil.copyfile(parent_bundle / name, staging / "parent" / name)
        copied = _verify_payload(staging / "snapshot")
        if copied["inventory"] != payload["inventory"]:
            raise ValueError("snapshot changed during import")
        for name, expected in parent_inventory.items():
            if _hash(staging / "parent" / name) != expected["sha256"]:
                raise ValueError("parent evidence changed during import")
        index = {"schema_version": 1, "kind": "partial-filesystem-snapshot",
                 "parent_run_id": payload["coverage"]["parent_run_id"],
                 "files": _inventory(staging), "verified_files": payload["verified_files"],
                 "expanded_bytes": payload["expanded_bytes"], "coverage": payload["coverage"],
                 "not_captured": [*NOT_CAPTURED, *_EXTRA_NOT_CAPTURED],
                 "limits": {"max_files": MAX_FILES, "max_bytes": MAX_BYTES},
                 "execution_started": False,
                 "capture_provenance": "operator_supplied; capture/run-state identity unverified"}
        index["index_sha256"] = digest(index)
        (staging / "index.json").write_text(json.dumps(index, indent=2))
        staging.rename(destination)
    return {"path": str(destination.resolve()), **index}


def _verify_frozen(root: Path) -> dict[str, Any]:
    inventory = _inventory(root)
    index = _json(root / "index.json")
    if index.get("schema_version") != 1 or index.get("index_sha256") != digest(
            {key: value for key, value in index.items() if key != "index_sha256"}):
        raise ValueError("frozen snapshot index hash mismatch")
    inventory.pop("index.json", None)
    if inventory != index.get("files"):
        raise ValueError("frozen snapshot inventory hash mismatch")
    payload = _verify_payload(root / "snapshot")
    _closed_parent(root / "parent", payload, stored=True)
    if (index.get("coverage") != payload["coverage"]
            or index.get("parent_run_id") != payload["coverage"]["parent_run_id"]):
        raise ValueError("frozen index coverage mismatch")
    return index


def verify_frozen_snapshot(snapshot: Path, *, expected_index_sha256: str) -> dict[str, Any]:
    """Check the detached store-owned index pin as well as every captured byte."""
    index = _json(Path(snapshot) / "index.json")
    if not expected_index_sha256 or index.get("index_sha256") != expected_index_sha256:
        raise ValueError("frozen snapshot detached index pin mismatch")
    return _verify_frozen(Path(snapshot))


def materialize_fork(snapshot: Path, destination: Path, *, expected_index_sha256: str) -> dict[str, Any]:
    """Restore captured files to fresh local paths; this prepares, never runs, a fork."""
    snapshot, destination = Path(snapshot), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("fork destination must not exist")
    index = verify_frozen_snapshot(snapshot, expected_index_sha256=expected_index_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fork-restore-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "fork"
        staging.mkdir()
        # core restore writes temporary verification directories; operate on a private copy.
        shutil.copytree(snapshot / "snapshot", staging / "source", symlinks=True)
        _verify_payload(staging / "source")
        workspaces, services, shared = (staging / name for name in ("workspaces", "services", "shared"))
        for path in (workspaces, services, shared):
            path.mkdir()
        restore = asyncio.run(restore_snapshot(staging / "source", workspaces, LocalWorldState(services, shared)))
        if restore.get("ok") is not True:
            raise ValueError(f"snapshot restore verification failed: {restore.get('mismatches')}")
        verify_frozen_snapshot(snapshot, expected_index_sha256=expected_index_sha256)
        receipt = {"schema_version": 1, "kind": "prepared-partial-fork", "execution_started": False,
                   "snapshot_index_sha256": index["index_sha256"], "parent_run_id": index["parent_run_id"],
                   "restore": restore, "coverage": index["coverage"], "not_captured": index["not_captured"],
                   "capture_provenance": index["capture_provenance"],
                   "workspace_root": str((destination / "workspaces").resolve()),
                   "services_root": str((destination / "services").resolve()),
                   "shared_root": str((destination / "shared").resolve())}
        (staging / "lineage.json").write_text(json.dumps(receipt, indent=2))
        staging.rename(destination)
    return {"path": str(destination.resolve()), **receipt}


def prepare_fork_recipe(snapshot: Path, destination: Path, *, expected_index_sha256: str,
                        world_archive: Path, expected_archive_sha256: str,
                        changes: list[dict[str, Any]], roles: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Prepare executable local CLI inputs for a fresh-session, partial-state rerun.

    Apply factors to the original portable package, not to stale worker paths.
    Adding unpinned files is rejected. Linux Docker, repository assets and model
    serving remain external prerequisites; this function starts no workload.
    """
    import yaml

    from mesocosm.core.config import absolutize_manifest
    from mesocosm.core.manifest import WorldManifest, validate_manifest
    from mesocosm.core.snapshot import fork_manifest
    from mesocosm.env.compose import service_address
    from mesocosm.execution.package import _confined, _get, _references, unpack_submission
    from mesocosm.reproduction.factors import configure_experiment

    snapshot, destination, world_archive = Path(snapshot), Path(destination).absolute(), Path(world_archive)
    index = verify_frozen_snapshot(snapshot, expected_index_sha256=expected_index_sha256)
    if world_archive.is_symlink() or _hash(world_archive) != expected_archive_sha256:
        raise ValueError("fork parent input archive hash mismatch")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("fork recipe destination must not exist")
    destination.mkdir(parents=True, mode=0o700)
    try:
        inputs = destination / "inputs"
        package = unpack_submission(world_archive, inputs)
        if package["kind"] != "world":
            raise ValueError("fork recipe requires a single frozen world package")
        original = yaml.safe_load((inputs / package["entrypoint"]).read_text())
        configured = configure_experiment(original, changes, roles)
        world = configured["manifest"]
        # Factor changes may remove references; they cannot smuggle in new local inputs.
        allowed = {_get(container, key) for container, key in _references(original)}
        for container, key in _references(world):
            reference = _get(container, key)
            if reference not in allowed:
                raise ValueError("fork factor changes require new unpinned inputs; configure a new base experiment")
            _confined(inputs, reference)
        absolutize_manifest(world, inputs)
        spec = WorldManifest.model_validate(world)
        if {agent.id for agent in spec.agents} != {
                item["agent_id"] for item in index["coverage"]["captured"] if item["component"] == "workspace"}:
            raise ValueError("fork recipe requires captured workspaces for exactly the configured agents")
        label = "fork-" + secrets.token_hex(8)
        world["name"] = label
        world["gateway"].update(url=f"http://{service_address(spec, 'gateway')}:8080",
                                api_key=secrets.token_urlsafe(32), upstream_key=None, run_label=label)
        for number, agent in enumerate(world["agents"]):
            subnet = number + 1 if spec.isolate_agents else 0
            agent["url"] = f"http://172.30.{subnet}.{10 + number}:4096"
        restored = materialize_fork(snapshot, destination / "restored", expected_index_sha256=expected_index_sha256)
        patch_path = destination / "manifest.patch.json"
        patch_path.write_text(json.dumps(world, indent=2))
        patch_path.chmod(0o600)
        check_path = destination / "validated-world.yaml"
        fork_manifest(destination / "restored" / "source", world, check_path)
        check_path.chmod(0o600)
        report = validate_manifest(check_path)
        if not report.ok:
            raise ValueError("invalid fork recipe: " + "; ".join(report.errors))
        if _hash(world_archive) != expected_archive_sha256:
            raise ValueError("parent input archive changed during preparation")
        verify_frozen_snapshot(snapshot, expected_index_sha256=expected_index_sha256)
        argv = ["mesocosm", "fork", str(destination / "restored" / "source"), "--edit", str(patch_path),
                "--out", str(destination / "execution"), "--runs-root", str(destination / "runs"),
                "--workspace-root", str(destination / "execution-workspaces"), "--json"]
        receipt = {"schema_version": 1, "kind": "local-partial-fork-recipe", "execution_started": False,
                   "snapshot_index_sha256": expected_index_sha256, "parent_archive_sha256": expected_archive_sha256,
                   "parent_package_digest": package["digest"], "factor_changes": configured["effective_changes"],
                   "argv": argv, "restoration": restored, "files": _inventory(destination),
                   "not_captured": index["not_captured"], "capture_provenance": index["capture_provenance"],
                   "semantics": "new harness/model/tool execution from captured files; not exact prefix replay",
                   "required_external_resources": ["Linux Docker with required containment",
                                                   "Mesocosm checkout as current working directory for spike assets",
                                                   "compatible pinned model serving and operator-supplied credentials"],
                   "execution_options": "CLI --vllm-host/--vllm-port select the inference endpoint; record overrides",
                   "native_acceptance": "unverified"}
        receipt["receipt_sha256"] = digest(receipt)
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2))
        (destination / "receipt.json").chmod(0o600)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {"path": str(destination), **receipt}


def verify_fork_recipe(destination: Path, *, expected_receipt_sha256: str) -> dict[str, Any]:
    """Verify a not-yet-executed recipe against its detached store-owned receipt pin."""
    destination = Path(destination)
    receipt = _json(destination / "receipt.json")
    body_sha256 = digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    if (not expected_receipt_sha256 or receipt.get("receipt_sha256") != expected_receipt_sha256
            or body_sha256 != expected_receipt_sha256):
        raise ValueError("fork recipe detached receipt pin mismatch")
    inventory = _inventory(destination)
    inventory.pop("receipt.json", None)
    if inventory != receipt.get("files"):
        raise ValueError("fork recipe input inventory changed")
    return {"path": str(destination.resolve()), **receipt}
