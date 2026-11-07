"""Software verification of bounded filesystem snapshots; no live worker acceptance."""

import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mesocosm.core.records import RunBundle
from mesocosm.core.snapshot import LocalWorldState, create_snapshot
from mesocosm.reproduction import snapshots


def make_snapshot(tmp_path, *, closed=True):
    workspace = tmp_path / "live-workspaces"
    (workspace / "a").mkdir(parents=True)
    (workspace / "a" / "answer.txt").write_text("real captured bytes")
    services, shared = tmp_path / "live-services", tmp_path / "live-shared"
    services.mkdir()
    shared.mkdir()
    (services / "board.jsonl").write_text('{"body":"recorded message"}\n')
    (shared / "data.txt").write_text("shared state")
    bundle = RunBundle.create(tmp_path / "runs", "run_parent")
    bundle.write_manifest({"name": "world", "agents": [{"id": "a"}], "model": {"id": "m"}}, "manifest-digest")
    bundle.record("task.loaded", task="task", source="manifest")
    if closed:
        bundle.record("runtime.cleanup", status="complete")
        bundle.write_summary({"status": "ok"})
    receipt = asyncio.run(create_snapshot(bundle.root, workspace, LocalWorldState(services, shared), tmp_path / "snapshots"))
    return Path(receipt["path"]), bundle


def test_freeze_and_fork_verify_actual_files_and_declare_partial_coverage(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    frozen = tmp_path / "frozen"
    receipt = snapshots.freeze_snapshot(source, frozen, parent_bundle=bundle.root)
    assert not receipt["execution_started"]
    assert "snapshot/manifest.json" in receipt["files"]
    assert "snapshot/workspaces/a.tar.gz" in receipt["files"]
    assert "parent/events.jsonl" in receipt["files"]
    inventory_before = snapshots._inventory(frozen)
    fork = snapshots.materialize_fork(frozen, tmp_path / "fork",
                               expected_index_sha256=json.loads((frozen / "index.json").read_text())["index_sha256"])
    assert fork["restore"]["ok"] and not fork["execution_started"]
    assert (Path(fork["workspace_root"]) / "a" / "answer.txt").read_text() == "real captured bytes"
    assert (Path(fork["shared_root"]) / "data.txt").read_text() == "shared state"
    assert (Path(fork["services_root"]) / "board.jsonl").exists()
    assert "optimizer and training RNG state" in fork["not_captured"]
    assert snapshots._inventory(frozen) == inventory_before  # Core restore never mutates frozen evidence.


def test_open_or_merely_paused_run_is_not_accepted(tmp_path):
    source, bundle = make_snapshot(tmp_path, closed=False)
    bundle.record("run.paused")
    with pytest.raises(ValueError, match="summary"):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)
    bundle.write_summary({"status": "ok"})
    with pytest.raises(ValueError, match="cleanup"):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)


def test_snapshot_before_cleanup_is_rejected(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    coverage = json.loads((source / "coverage.json").read_text())
    coverage["created"] = 0
    (source / "coverage.json").write_text(json.dumps(coverage))
    with pytest.raises(ValueError, match="predates"):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)


def test_restarted_parent_is_not_accepted(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    bundle.record("run.resumed")
    with pytest.raises(ValueError, match="after cleanup"):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)


@pytest.mark.parametrize("filename", ["manifest.json", "queue.json", "coverage.json", "files.json", "workspaces/a.tar.gz"])
def test_all_frozen_metadata_and_archives_are_pinned(tmp_path, filename):
    source, bundle = make_snapshot(tmp_path)
    frozen = tmp_path / "frozen"
    snapshots.freeze_snapshot(source, frozen, parent_bundle=bundle.root)
    target = frozen / "snapshot" / filename
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="inventory hash"):
        snapshots.materialize_fork(frozen, tmp_path / "fork",
                               expected_index_sha256=json.loads((frozen / "index.json").read_text())["index_sha256"])
    assert not (tmp_path / "fork").exists()


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "fifo", "bomb", "duplicate"])
def test_unsafe_archives_rejected_before_any_extraction(tmp_path, kind, monkeypatch):
    source, bundle = make_snapshot(tmp_path)
    archive_path = source / "workspaces/a.tar.gz"
    monkeypatch.setattr(snapshots, "MAX_BYTES", 8192)
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape"}.get(kind, "a"))
        member.size = 0
        if kind in {"symlink", "hardlink"}:
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../../escape"
        elif kind == "fifo":
            member.type = tarfile.FIFOTYPE
        elif kind == "bomb":
            member.size = 16384
        archive.addfile(member, io.BytesIO(b"x" * member.size))
        if kind == "duplicate":
            archive.addfile(member)
    with pytest.raises(ValueError):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)
    assert not (tmp_path / "frozen").exists() and not (tmp_path / "escape").exists()


def test_symlink_in_snapshot_tree_is_rejected(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    (source / "shared" / "leak").symlink_to(tmp_path / "runs")
    with pytest.raises(ValueError, match="links"):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)


def test_actual_captured_state_is_checked_before_freeze(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    (source / "shared" / "data.txt").write_text("changed")
    with pytest.raises(ValueError, match="captured-state hashes"):
        snapshots.freeze_snapshot(source, tmp_path / "frozen", parent_bundle=bundle.root)


def test_existing_destinations_are_never_overwritten(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        snapshots.freeze_snapshot(source, destination, parent_bundle=bundle.root)
    with pytest.raises(FileExistsError):
        snapshots.materialize_fork(source, destination, expected_index_sha256="unused")
    assert (destination / "keep").read_text() == "keep"


def test_restore_mismatch_never_publishes_partial_fork(tmp_path, monkeypatch):
    source, bundle = make_snapshot(tmp_path)
    frozen = tmp_path / "frozen"
    snapshots.freeze_snapshot(source, frozen, parent_bundle=bundle.root)

    async def broken_restore(*args):
        return {"ok": False, "mismatches": ["workspace/a/answer.txt"]}

    monkeypatch.setattr(snapshots, "restore_snapshot", broken_restore)
    with pytest.raises(ValueError, match="restore verification failed"):
        snapshots.materialize_fork(frozen, tmp_path / "fork",
                               expected_index_sha256=json.loads((frozen / "index.json").read_text())["index_sha256"])
    assert not (tmp_path / "fork").exists()


def test_existing_cli_fork_stops_before_model_run_on_bad_restore(tmp_path, monkeypatch):
    import mesocosm.cli.main as cli
    import mesocosm.core.manifest as manifests
    import mesocosm.core.run as runtime
    import mesocosm.core.snapshot as core_snapshots
    import mesocosm.env.compose as compose
    import mesocosm.env.stack as stacks

    called = []

    def fake_fork(snapshot, patch, output):
        output.parent.mkdir(parents=True)
        output.write_text("name: fork")
        return {"parent_run_id": "parent"}

    async def restore(*args):
        return {"ok": False, "mismatches": ["state"]}

    async def run(*args):
        called.append("run")
        raise AssertionError("run must not start")

    class Stack:
        def __init__(self, path):
            pass

        async def up(self):
            called.append("up")

        async def down(self):
            called.append("down")

    monkeypatch.setattr(core_snapshots, "fork_manifest", fake_fork)
    monkeypatch.setattr(core_snapshots, "restore_snapshot", restore)
    monkeypatch.setattr(manifests, "validate_manifest", lambda path: SimpleNamespace(ok=True))
    monkeypatch.setattr(manifests, "load_manifest", lambda path: SimpleNamespace(gateway=SimpleNamespace(run_label="test")))
    rendered = {}

    def render(*args, **kwargs):
        rendered.update(kwargs)
        return "services: {}"

    monkeypatch.setattr(compose, "compose_yaml", render)
    monkeypatch.setattr(compose, "write_agent_files", lambda *args, **kwargs: rendered.update(files_dir=args[2]))
    monkeypatch.setattr(stacks, "DockerStack", Stack)
    monkeypatch.setattr(runtime, "run_world", run)
    result = CliRunner().invoke(cli.app, ["fork", str(tmp_path / "snapshot"), "--out", str(tmp_path / "output"),
                                         "--runs-root", str(tmp_path / "runs"), "--workspace-root", str(tmp_path / "workspace"), "--json"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["status"] == "restore_failed"
    assert called == ["up", "down"]
    assert rendered["workspace_root"] == tmp_path / "workspace"
    assert rendered["runs_dir"] == str(tmp_path / "runs")
    assert rendered["files_dir"] == tmp_path / "runs" / "test" / "agents"


def test_detached_index_pin_rejects_rehashed_forged_metadata(tmp_path):
    source, bundle = make_snapshot(tmp_path)
    frozen = tmp_path / "frozen"
    pinned = snapshots.freeze_snapshot(source, frozen, parent_bundle=bundle.root)["index_sha256"]
    index = json.loads((frozen / "index.json").read_text())
    index["capture_provenance"] = "false claim"
    index["index_sha256"] = snapshots.digest({key: value for key, value in index.items() if key != "index_sha256"})
    (frozen / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="detached index pin"):
        snapshots.materialize_fork(frozen, tmp_path / "fork", expected_index_sha256=pinned)


def portable_snapshot(tmp_path):
    import yaml

    from mesocosm.execution.package import pack_submission

    inputs = tmp_path / "original"
    task = inputs / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("External task content, preserved verbatim.")
    (task / "environment" / "Dockerfile").write_text("FROM debian:bookworm\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    manifest = {"name": "original", "model": {"id": "model"}, "gateway": {"url": "http://gateway:8080"},
                "agents": [{"id": name} for name in ["a", "b", "c"]], "tasks": [{"path": "task"}],
                "services": {"enabled": False}}
    world = inputs / "world.yaml"
    world.write_text(yaml.safe_dump(manifest))
    archive = tmp_path / "world.zip"
    pack_submission(world, archive)
    bundle = RunBundle.create(tmp_path / "runs", "run_source")
    bundle.write_manifest(manifest, "original", path=world)
    bundle.record("runtime.cleanup", status="complete")
    bundle.write_summary({"status": "ok"})
    workspace = tmp_path / "workspace"
    for agent in ["a", "b", "c"]:
        path = workspace / agent
        path.mkdir(parents=True)
        (path / "state.txt").write_text(agent)
    snapshot = asyncio.run(create_snapshot(bundle.root, workspace, None, tmp_path / "snapshots"))
    frozen = tmp_path / "frozen"
    receipt = snapshots.freeze_snapshot(Path(snapshot["path"]), frozen, parent_bundle=bundle.root)
    return frozen, receipt["index_sha256"], archive


def test_fork_recipe_preserves_pinned_inputs_factors_and_executable_cli(tmp_path):
    import yaml
    frozen, pinned, archive = portable_snapshot(tmp_path)
    changes = [{"factor": "agent.budget.prompt_tokens", "scope": "agent", "target": "a", "value": 12345}]
    recipe = snapshots.prepare_fork_recipe(frozen, tmp_path / "recipe", expected_index_sha256=pinned,
        world_archive=archive, expected_archive_sha256=snapshots._hash(archive), changes=changes)
    assert recipe["argv"][:2] == ["mesocosm", "fork"] and not recipe["execution_started"]
    patch = json.loads((tmp_path / "recipe" / "manifest.patch.json").read_text())
    assert patch["agents"][0]["budget"]["prompt_tokens"] == 12345
    task_path = Path(patch["tasks"][0]["path"])
    assert task_path.is_relative_to(tmp_path / "recipe" / "inputs")
    assert (task_path / "instruction.md").read_text() == "External task content, preserved verbatim."
    check = yaml.safe_load((tmp_path / "recipe" / "validated-world.yaml").read_text())
    assert check["agents"][0]["budget"]["prompt_tokens"] == 12345
    assert patch["gateway"]["url"] != "http://gateway.invalid"
    assert patch["gateway"]["api_key"] not in {"mesocosm-local", ""}
    assert "fresh" in recipe["semantics"] or "new harness" in recipe["semantics"]
    assert snapshots.verify_fork_recipe(tmp_path / "recipe", expected_receipt_sha256=recipe["receipt_sha256"]) == recipe
    (task_path / "instruction.md").write_text("tamper")
    with pytest.raises(ValueError, match="inventory"):
        snapshots.verify_fork_recipe(tmp_path / "recipe", expected_receipt_sha256=recipe["receipt_sha256"])


def test_recipe_rejects_unpinned_files_and_archive_replacement(tmp_path):
    frozen, pinned, archive = portable_snapshot(tmp_path)
    with pytest.raises(ValueError, match="archive hash"):
        snapshots.prepare_fork_recipe(frozen, tmp_path / "recipe", expected_index_sha256=pinned,
            world_archive=archive, expected_archive_sha256="wrong", changes=[])
    with pytest.raises(ValueError, match="unpinned inputs"):
        snapshots.prepare_fork_recipe(frozen, tmp_path / "recipe", expected_index_sha256=pinned,
            world_archive=archive, expected_archive_sha256=snapshots._hash(archive),
            changes=[{"factor": "agent.skills", "scope": "agent", "target": "a", "value": ["/unowned/skill"]}])
    assert not (tmp_path / "recipe").exists()
