"""Collector shutdown precedes native verifier teardown; real stream loss remains fatal."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from test_run_e2e import MANIFEST
from test_run_e2e import world as world

import mesocosm.core.run as runtime
from mesocosm.core.records import RunBundle
from mesocosm.core.run import run_world


async def test_three_collectors_drain_and_close_before_verifier_teardown(world, monkeypatch):  # noqa: F811
    manifest, components, root = world
    for number in (2, 3):
        manifest.agents.append(manifest.agents[0].model_copy(update={
            "id": f"agent-{number}", "url": f"http://127.0.0.1:{4095 + number}"}))
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", .02)
    comps = components()
    original_harness, original_verify = comps.harness_for, comps.verifier.verify
    snapshots = {agent.id: asyncio.Event() for agent in manifest.agents}
    closed, verified = set(), set()
    original_record = RunBundle.record

    def record(self, kind, **kwargs):
        event = original_record(self, kind, **kwargs)
        if kind == "harness.transcript":
            snapshots[kwargs["agent_id"]].set()
        return event

    monkeypatch.setattr(RunBundle, "record", record)

    def harness_for(agent):
        harness = original_harness(agent)

        async def events():
            try:
                await snapshots[agent.id].wait()
                # A queued terminal tool event arrives during the bounded drain.
                yield {"type": "message.part.updated", "properties": {"part": {
                    "sessionID": f"ses_{agent.id}_1", "type": "tool", "tool": "bash",
                    "callID": f"last-{agent.id}", "state": {"status": "completed", "output": "last"}}}}
                await asyncio.Event().wait()
            finally:
                closed.add(agent.id)

        harness.events = events
        return harness

    async def verify(task, workspace, logs_dir, script="test.sh"):
        agent_id = workspace.name
        assert agent_id in closed, "native verifier must not remove a server with a live collector"
        verified.add(agent_id)
        return await original_verify(task, workspace, logs_dir, script)

    comps.harness_for, comps.verifier.verify = harness_for, verify
    result = await run_world(manifest, MANIFEST, comps, root / "runs")
    assert result.status == "ok", result.error
    assert closed == verified == {agent.id for agent in manifest.agents}
    events = RunBundle.open(result.bundle_path).events()
    assert not any(event.kind == "harness.events_ended" for event in events)
    for agent in manifest.agents:
        own = [event for event in events if event.agent_id == agent.id]
        kinds = [event.kind for event in own]
        closure = next(event for event in own if event.kind == "harness.events_closed")
        transcript = next(event for event in own if event.kind == "harness.transcript")
        assert kinds.index("harness.transcript") < kinds.index("harness.events_closed") < kinds.index("verifier.reward")
        assert closure.data["session_id"] == transcript.data["session_id"]
        assert closure.data["task"] == "hello-world"
        assert closure.data["status"] == "closed" and closure.data["planned"] is True
        assert closure.data["reason"] == "completed_transcript_before_native_verifier"
        assert closure.data["collector_failed"] is False
        assert closure.links[0].event_id == transcript.event_id
        raw = next(event for event in own if event.kind == "raw_log.closed")
        assert raw.data["entries"] == 1, "queued final event must reach the native raw log"


@pytest.mark.parametrize("phase", ["active_turn", "final_drain"])
@pytest.mark.parametrize("failure", ["protocol_error", "clean_eof"])
async def test_unexpected_stream_error_is_scoped_and_never_graded(world, monkeypatch, phase, failure):  # noqa: F811
    manifest, components, root = world
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", .02)
    comps = components()
    original_harness = comps.harness_for
    fail_now, cancelled = asyncio.Event(), asyncio.Event()
    original_record = RunBundle.record

    def record(self, kind, **kwargs):
        event = original_record(self, kind, **kwargs)
        if kind == "harness.transcript" and phase == "final_drain":
            fail_now.set()
        return event

    monkeypatch.setattr(RunBundle, "record", record)

    def harness_for(agent):
        harness = original_harness(agent)

        async def events():
            await fail_now.wait()
            if failure == "protocol_error":
                raise httpx.RemoteProtocolError("actual premature stream loss")
            return
            yield  # mark this as the harness async event iterator

        async def send(*args, **kwargs):
            fail_now.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def abort(*args):
            return True

        harness.events, harness.abort_session = events, abort
        if phase == "active_turn":
            harness.send_message = send
        return harness

    async def verify(*args, **kwargs):
        pytest.fail("stream loss must not reach the verifier")

    comps.harness_for, comps.verifier.verify = harness_for, verify
    result = await asyncio.wait_for(run_world(manifest, MANIFEST, comps, root / "runs"), 3)
    expected = "actual premature stream loss" if failure == "protocol_error" else "unexpected clean EOF"
    assert result.status == "failed" and expected in result.error
    events = RunBundle.open(result.bundle_path).events()
    ended = [event for event in events if event.kind == "harness.events_ended"]
    assert len(ended) == 1
    assert ended[0].agent_id == "agent-1"
    assert ended[0].data["session_id"] == "ses_agent-1_1"
    assert ended[0].data["task"] == "hello-world" and ended[0].data["planned"] is False
    assert expected in ended[0].data["error"]
    assert not any(event.kind in {"verifier.reward", "harness.events_closed"} for event in events)
    if phase == "active_turn":
        assert cancelled.is_set(), "a broken observation stream must interrupt an in-flight turn"
    else:
        assert any(event.kind == "harness.transcript" for event in events)


async def test_outer_cancellation_during_collector_close_never_emits_success(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", 0)
    cancelling, released = asyncio.Event(), asyncio.Event()

    async def collect():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
            raise

    bundle = RunBundle.create(tmp_path, "cancel-close")
    collector = asyncio.create_task(collect())
    await asyncio.sleep(0)
    ctx = runtime.AgentContext(harness=None, workspace=tmp_path, collector=collector, session_id="session")
    closing = asyncio.create_task(runtime._close_task_collector(ctx, bundle, "agent-1", "task", "transcript"))
    await asyncio.wait_for(cancelling.wait(), 1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert released.is_set()
    assert not any(event.kind == "harness.events_closed" for event in bundle.events())
