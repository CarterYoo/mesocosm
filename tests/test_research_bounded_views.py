"""Software interface regressions; these records are not empirical learning data."""
import json

import pytest

from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.views import PAGE_BYTES, RESPONSE_BYTES, byte_size


@pytest.fixture
def bench(tmp_path):
    return ResearchWorkbench(tmp_path / "research")


@pytest.mark.asyncio
async def test_import_is_compact_and_filtered_pages_preserve_every_event(bench, tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text("".join(json.dumps({"kind": "speech", "agent_id": f"agent-{i % 3}",
        "session_id": "session", "ts": i, "data": {"text": "recorded software fixture"}}) + "\n"
        for i in range(1000)))
    result = await bench.call("inspect_transcript", {"path": str(source)})
    assert byte_size(result) < 1024
    assert result["counts"]["events"] == 1000
    assert "transcript" not in result
    identifier = result["transcript_id"]
    original = bench.store.get(identifier)["events"]
    seen, offset = [], 0
    while True:
        page = await bench.call("read_events", {"transcript_id": identifier, "offset": offset, "limit": 100})
        assert byte_size(page) < RESPONSE_BYTES
        seen.extend(item["value"] for item in page["items"])
        if page["eof"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert seen == original
    filtered = bench.read_events(identifier, agent_id="agent-1", kind="speech")
    assert filtered["total"] == 333
    event = original[3]
    single = bench.read_events(identifier, event_id=event["event_id"])
    assert single["total"] == 1 and single["items"][0]["value"] == event
    frozen = bench.store.directory("transcripts", identifier) / source.name
    assert frozen.read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_oversized_single_event_can_be_read_without_truncation(bench, tmp_path):
    text = "한글🙂\n" * 10000
    source = tmp_path / "huge.jsonl"
    source.write_text(json.dumps({"kind": "speech", "data": {"text": text}}, ensure_ascii=False) + "\n")
    imported = await bench.call("inspect_transcript", {"path": str(source)})
    identifier = imported["transcript_id"]
    event = bench.read_events(identifier)["items"][0]
    assert event["omitted"] is True and "value" not in event
    pointer = event["pointer"] + "/data/text"
    pieces, offset = [], 0
    while True:
        page = await bench.call("read_record", {"record_id": identifier, "pointer": pointer, "offset": offset})
        assert byte_size(page) <= RESPONSE_BYTES
        pieces.append(page["text"])
        if page["eof"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert "".join(pieces) == text


@pytest.mark.asyncio
async def test_dispatch_bounds_any_large_result_and_retains_exact_content(bench, monkeypatch):
    value = {"status": "unknown", "deep": {"a/b~c": ["x" * 100000, "actual content"]}}
    monkeypatch.setattr(bench, "validate_experiment", lambda experiment_id: value)
    result = await bench.call("validate_experiment", {"experiment_id": "software-test"})
    assert byte_size(result) < 1024 and result["detail_omitted"]
    stored = bench.store.get(result["record_id"])
    assert stored["payload"] == value
    nested = bench.read_record(result["record_id"], "/payload/deep/a~1b~0c/1")
    assert nested["text"] == "actual content"


def test_factor_discovery_is_small_and_selected_detail_matches_original_schema(bench):
    from mesocosm.reproduction.factors import list_factors
    response = bench.list_factors(prefix="agent.budget.")
    assert response["factors"] and byte_size(response) < 4000
    assert all("schema" not in row for row in response["factors"])
    detailed = bench.list_factors(names=["agent.budget.requests"], detail=True)
    original = next(row for row in list_factors() if row["name"] == "agent.budget.requests")
    assert detailed["factors"] == [original]
    seen, offset = [], 0
    while True:
        page = bench.list_factors(offset=offset, limit=100)
        assert byte_size(page) < RESPONSE_BYTES
        seen.extend(row["name"] for row in page["factors"])
        if page["eof"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert seen == [row["name"] for row in list_factors()]


def test_restart_discovers_durable_objects_and_jobs(bench):
    behavior = bench.define_behavior({"name": "record", "stages": {"onset": {"event": {"kind": "speech"}}}})
    job_id = "job-" + "1" * 64
    bench.store.save_job(job_id, {"kind": "training", "status": "queued", "config": {}}, new=True)
    resumed = ResearchWorkbench(bench.store.root)
    assert resumed.list_records("behavior")["items"][0]["record_id"] == behavior["behavior_id"]
    assert resumed.list_records("job")["items"][0]["record_id"] == job_id
    assert resumed.read_record(job_id, "/status")["text"] == "queued"


@pytest.mark.asyncio
async def test_new_navigation_never_releases_operator_only_or_linked_holdout_records(bench, tmp_path, monkeypatch):
    operator = ResearchWorkbench(bench.store.root, allow_test_evaluation=True)
    source = tmp_path / "heldout.json"
    source.write_text(json.dumps({"split": "test", "events": [{"kind": "speech", "data": {"text": "private"}}]}))
    imported = operator.inspect_transcript(str(source))
    identifier = imported["transcript_id"]
    assert bench.list_records("transcript")["items"] == []
    with pytest.raises(PermissionError):
        bench.read_events(identifier)
    with pytest.raises(PermissionError):
        bench.read_record(identifier)
    experiment = bench.store.put("experiment", {"split": "test"})
    job_id = "job-" + "2" * 64
    bench.store.save_job(job_id, {"kind": "world", "status": "completed", "experiment_id": experiment}, new=True)
    assert bench.list_records("job")["items"] == []
    with pytest.raises(PermissionError):
        bench.read_record(job_id)
    monkeypatch.setattr(operator, "validate_experiment", lambda experiment_id: {"data": "private" * 10000})
    result = await operator.call("validate_experiment", {"experiment_id": experiment})
    with pytest.raises(PermissionError):
        bench.read_record(result["record_id"])


@pytest.mark.parametrize("offset,limit", [(-1, 20), (0, 0), (0, 101), (False, 20)])
def test_invalid_pagination_is_rejected(bench, offset, limit):
    with pytest.raises(ValueError):
        bench.list_factors(offset=offset, limit=limit)


def test_empty_and_escaped_record_navigation(bench):
    identifier = bench.store.put("behavior", {"a/b~c": [], "scalar": None})
    assert bench.read_record(identifier, "/a~1b~0c")["eof"]
    assert bench.read_record(identifier, "/scalar")["value"] is None
    with pytest.raises(ValueError):
        bench.read_record(identifier, "/a~2b")
    assert PAGE_BYTES < RESPONSE_BYTES


def test_long_source_keys_have_short_lossless_navigation(bench):
    key, text = "long-key" * 10000, "recorded text" * 1000
    identifier = bench.store.put("transcript", {key: text})
    page = bench.read_record(identifier)
    child = page["items"][0]
    assert child["key_omitted"] and child["pointer"] is None
    assert byte_size(page) < 1024
    parts, offset = [], 0
    while True:
        chunk = bench.read_record(identifier, index_path=child["index_path"], offset=offset,
                                  expected_version=page["record_version"])
        parts.append(chunk["text"])
        assert byte_size(chunk) < RESPONSE_BYTES
        if chunk["eof"]:
            break
        offset = chunk["next_offset"]
    assert "".join(parts) == text


def test_only_requested_page_members_are_described(bench, monkeypatch):
    from mesocosm.reproduction import views
    identifier = bench.store.put("behavior", {"rows": list(range(10000))})
    seen = []
    original = views._item
    def item(value, *args):
        seen.append(value)
        return original(value, *args)
    monkeypatch.setattr(views, "_item", item)
    page = bench.read_record(identifier, "/rows", offset=50, limit=5)
    assert seen == [50, 51, 52, 53, 54]
    assert page["total"] == 10000 and page["next_offset"] == 55


def test_export_discovery_and_mutable_page_version(bench):
    identifier = bench.store.put("export", {"path": "private-local-artifact"})
    assert bench.list_records("export")["items"][0]["record_id"] == identifier
    job_id = "job-" + "5" * 64
    bench.store.save_job(job_id, {"kind": "training", "status": "queued"}, new=True)
    before = bench.read_record(job_id)
    bench.store.save_job(job_id, {"kind": "training", "status": "running"})
    with pytest.raises(ValueError, match="record changed"):
        bench.read_record(job_id, index_path=[0], expected_version=before["record_version"])
