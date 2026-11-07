"""Independent regression checks for frozen inputs, authority and export integrity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_reproduction_service import candidate

from mesocosm.reproduction.service import ResearchWorkbench


@pytest.fixture
def bench(tmp_path):
    return ResearchWorkbench(tmp_path / "research")


def test_raw_transcript_bytes_survive_original_removal_and_are_exported(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    original = tmp_path / "transcript.txt"
    content = b"  Actual original bytes.\n\nA recorded statement.  \n"
    original.write_bytes(content)
    imported = bench.inspect_transcript(str(original))
    original.unlink()
    evaluation = bench.evaluate_candidate(experiment["experiment_id"], imported["transcript_id"])
    exported = bench.export_reproducer(experiment["experiment_id"], [evaluation["evaluation_id"]])
    path = Path(exported["path"]) / "sources" / imported["transcript_id"] / "transcript.txt"
    assert path.read_bytes() == content
    path.write_bytes(b"tampered preserved source")
    with pytest.raises(ValueError, match="export bytes"):
        bench.export_reproducer(experiment["experiment_id"], [evaluation["evaluation_id"]])


def test_original_operational_credentials_are_absent_from_export(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    exported = bench.export_reproducer(experiment["experiment_id"])
    for path in Path(exported["path"]).rglob("*"):
        if path.is_file():
            assert b"DO-NOT-EXPORT" not in path.read_bytes()
            assert b"UPSTREAM-SECRET" not in path.read_bytes()
    script = Path(exported["path"]) / "rerun.py"
    compile(script.read_text(), str(script), "exec")
    assert "os.O_EXCL" in script.read_text()


@pytest.mark.parametrize("filename", ["world.zip", "behavior.json", "files.json", "README.md"])
def test_repeated_export_rejects_partial_or_changed_existing_artifacts(bench, tmp_path, filename):
    experiment, _ = candidate(bench, tmp_path)
    first = bench.export_reproducer(experiment["experiment_id"])
    path = Path(first["path"]) / filename
    path.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="export"):
        bench.export_reproducer(experiment["experiment_id"])


def test_heldout_evidence_cannot_be_relabelled_as_development_candidate(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    source = tmp_path / "test.json"
    source.write_text(json.dumps({"split": "test", "events": [{"kind": "speech", "data": {"text": "held out"}}]}))
    with pytest.raises(ValueError, match="held-out"):
        bench.inspect_transcript(str(source))
    operator = ResearchWorkbench(bench.store.root, allow_test_evaluation=True)
    imported = operator.inspect_transcript(str(source))
    evaluation = operator.evaluate_candidate(experiment["experiment_id"], imported["transcript_id"])
    with pytest.raises(ValueError, match="held-out"):
        bench.evaluate_candidate(experiment["experiment_id"], imported["transcript_id"])
    with pytest.raises(ValueError, match="held-out"):
        bench.compare_candidates([evaluation["evaluation_id"]])
    with pytest.raises(ValueError, match="held-out"):
        bench.export_reproducer(experiment["experiment_id"], [evaluation["evaluation_id"]])


@pytest.mark.asyncio
async def test_unregistered_worker_cannot_receive_arbitrary_environment_credential(bench, tmp_path, monkeypatch):
    experiment, _ = candidate(bench, tmp_path)
    monkeypatch.setenv("UNRELATED_SECRET", "x" * 64)
    def forbidden(*args, **kwargs):
        pytest.fail("network client must not be constructed before operator allowlist validation")
    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", forbidden)
    with pytest.raises(ValueError, match="operator-approved"):
        await bench.run_experiment(experiment["experiment_id"], "exfil", "https://unapproved.example",
                                   token_env="UNRELATED_SECRET")


def test_snapshot_cannot_be_attached_to_an_unrelated_candidate(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    arbitrary = tmp_path / "unverified-parent"
    arbitrary.mkdir()
    with pytest.raises(ValueError, match="verified fetched run"):
        bench.snapshot_experiment(experiment["experiment_id"], str(tmp_path / "snapshot"), str(arbitrary))


@pytest.mark.asyncio
async def test_sync_dispatch_runs_outside_active_event_loop(bench, monkeypatch):
    import asyncio
    def synchronous(**kwargs):
        async def work():
            return {"verified": True}
        return asyncio.run(work())
    monkeypatch.setattr(bench, "list_factors", synchronous)
    assert await bench.call("list_factors", {}) == {"verified": True}


def test_frozen_source_tampering_fails_before_evaluation(bench, tmp_path):
    experiment, _ = candidate(bench, tmp_path)
    path = tmp_path / "record.txt"
    path.write_text("recorded data")
    imported = bench.inspect_transcript(str(path))
    frozen = bench.store.directory("transcripts", imported["transcript_id"]) / path.name
    frozen.write_text("different original bytes")
    with pytest.raises(ValueError, match="source bytes"):
        bench.evaluate_candidate(experiment["experiment_id"], imported["transcript_id"])
