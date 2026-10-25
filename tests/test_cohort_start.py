"""CPU scheduling contracts; these fixtures are not real multi-agent acceptance."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from test_run_e2e import MANIFEST, two_tasks
from test_run_e2e import world as world

import mesocosm.core.run as runtime
from mesocosm.core.control import STOP_MARKER
from mesocosm.core.manifest import TaskSpec, WorldManifest
from mesocosm.core.records import RunBundle
from mesocosm.core.run import run_world


def three_agents(manifest):
    for number in (2, 3):
        manifest.agents.append(manifest.agents[0].model_copy(update={
            "id": f"agent-{number}", "url": f"http://127.0.0.1:{4095 + number}"}))
    return {agent.id for agent in manifest.agents}


def record_ready(monkeypatch):
    first_ready = asyncio.Event()
    original_record = RunBundle.record

    def record(self, kind, **kwargs):
        event = original_record(self, kind, **kwargs)
        if kind == "task.cohort_ready":
            first_ready.set()
        return event

    monkeypatch.setattr(RunBundle, "record", record)
    return first_ready


async def test_all_three_sessions_ready_before_each_unchanged_task_prompt(world, monkeypatch):  # noqa: F811
    manifest, components, root = world
    ids = three_agents(manifest)
    manifest.task_start = "cohort"
    manifest_path = two_tasks(root)
    manifest.tasks = [TaskSpec(path=f"tasks/{name}") for name in ("first", "second")]
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", .01)
    comps = components()
    prepared = {name: set() for name in ("first", "second")}
    sessions = {name: set() for name in ("first", "second")}
    contexts, submitted = {}, []
    original_harness = comps.harness_for

    async def prepare(agent, task, workspace, **kwargs):
        await asyncio.sleep(.01 * int(agent.id[-1]))
        prepared[task.name].add(agent.id)
        contexts[agent.id] = task.name
        return {"status": "fixture_prepared"}

    def harness_for(agent):
        harness = original_harness(agent)
        create, send = harness.create_session, harness.send_message

        async def create_session(title):
            await asyncio.sleep(.01 * int(agent.id[-1]))
            session_id = await create(title)
            sessions[contexts[agent.id]].add(agent.id)
            return session_id

        async def send_message(session_id, text, **kwargs):
            task = contexts[agent.id]
            assert prepared[task] == sessions[task] == ids
            assert text == f"Write hello.txt ({task}).\n"
            submitted.append((task, agent.id, session_id, time.time()))
            return await send(session_id, text, **kwargs)

        harness.create_session, harness.send_message = create_session, send_message
        return harness

    comps.task_runtime = SimpleNamespace(prepare=prepare)
    comps.harness_for = harness_for
    result = await asyncio.wait_for(run_world(manifest, manifest_path, comps, root / "runs"), 5)
    assert result.status == "ok", result.error
    assert len(submitted) == 6 and len(result.rewards) == 6
    assert set(result.rewards.values()) == {1.0}
    events = RunBundle.open(result.bundle_path).events()
    for name in ("first", "second"):
        ready = [event for event in events if event.kind == "task.cohort_ready" and event.data["task"] == name]
        release = next(event for event in events if event.kind == "task.cohort_released" and event.data["task"] == name)
        assert {event.agent_id for event in ready} == set(release.data["participants"]) == ids
        assert {link.event_id for link in release.links} == {event.event_id for event in ready}
        assert release.data["sessions"] == {event.agent_id: event.data["session_id"] for event in ready}
        assert all(event.ts <= release.ts for event in ready)
        first_prompts = [stamp for task, _, _, stamp in submitted if task == name]
        assert len(first_prompts) == 3 and all(release.ts <= stamp for stamp in first_prompts)


@pytest.mark.parametrize("failure", ["prepare_error", "session_error", "participant_cancel", "event_eof"])
async def test_startup_failure_or_participant_cancel_releases_waiters(world, monkeypatch, failure):  # noqa: F811
    manifest, components, root = world
    three_agents(manifest)
    manifest.task_start = "cohort"
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", .01)
    first_ready = record_ready(monkeypatch)
    comps = components()
    original_harness = comps.harness_for
    preparing_finished = set()

    async def prepare(agent, task, workspace, **kwargs):
        if agent.id == "agent-3":
            try:
                await first_ready.wait()
                if failure == "prepare_error":
                    raise RuntimeError("preparation failed")
                if failure == "participant_cancel":
                    raise asyncio.CancelledError("participant preparation cancelled")
                if failure == "event_eof":
                    await asyncio.Event().wait()
            finally:
                preparing_finished.add(agent.id)
        return {"status": "fixture_prepared"}

    def harness_for(agent):
        harness = original_harness(agent)
        create = harness.create_session

        async def create_session(title):
            if agent.id == "agent-3" and failure == "session_error":
                raise RuntimeError("session creation failed")
            return await create(title)

        async def events():
            await first_ready.wait()
            return
            yield

        async def send(*args, **kwargs):
            pytest.fail("no first prompt is allowed before the cohort is ready")

        harness.create_session, harness.send_message = create_session, send
        if agent.id == "agent-1" and failure == "event_eof":
            harness.events = events
        return harness

    comps.task_runtime, comps.harness_for = SimpleNamespace(prepare=prepare), harness_for
    result = await asyncio.wait_for(run_world(manifest, MANIFEST, comps, root / "runs"), 3)
    assert result.status == "failed", result.error
    assert "agent-3" in preparing_finished
    events = RunBundle.open(result.bundle_path).events()
    assert any(event.kind == "task.cohort_ready" for event in events)
    assert sum(event.kind == "task.cohort_aborted" for event in events) == 1
    assert not any(event.kind in {"task.cohort_released", "harness.turn", "verifier.reward"} for event in events)
    if failure == "event_eof":
        assert any(event.kind == "harness.events_ended" and event.agent_id == "agent-1" for event in events)


@pytest.mark.parametrize("interrupt", ["caller_cancel", "operator_stop"])
async def test_stop_while_waiting_cleans_up_without_starting(world, monkeypatch, interrupt):  # noqa: F811
    manifest, components, root = world
    three_agents(manifest)
    manifest.task_start = "cohort"
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", .01)
    first_ready = record_ready(monkeypatch)
    comps = components()
    blocked_closed = asyncio.Event()

    async def prepare(agent, task, workspace, **kwargs):
        if agent.id == "agent-3":
            try:
                await asyncio.Event().wait()
            finally:
                blocked_closed.set()
        return {"status": "fixture_prepared"}

    comps.task_runtime = SimpleNamespace(prepare=prepare)
    running = asyncio.create_task(run_world(manifest, MANIFEST, comps, root / "runs", run_id="cohort-stop"))
    await asyncio.wait_for(first_ready.wait(), 2)
    if interrupt == "caller_cancel":
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 2)
    else:
        marker = root / "runs" / "cohort-stop" / STOP_MARKER
        marker.parent.mkdir(parents=True)
        marker.write_text("stop")
        result = await asyncio.wait_for(running, 2)
        assert result.status == "stopped", result.error
    assert blocked_closed.is_set()
    events = RunBundle.open(root / "runs" / "cohort-stop").events()
    assert any(event.kind == "task.cohort_aborted" for event in events)
    assert not any(event.kind in {"task.cohort_released", "harness.turn", "verifier.reward"} for event in events)


async def test_default_still_starts_without_waiting_for_slow_peers(world, monkeypatch):  # noqa: F811
    manifest, components, root = world
    three_agents(manifest)
    assert manifest.task_start == "as_ready"
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", .01)
    comps = components()
    first_prompt = asyncio.Event()
    components.hooks["on_message"] = lambda body: first_prompt.set()

    async def prepare(agent, task, workspace, **kwargs):
        if agent.id == "agent-3":
            await first_prompt.wait()
        return {"status": "fixture_prepared"}

    comps.task_runtime = SimpleNamespace(prepare=prepare)
    result = await asyncio.wait_for(run_world(manifest, MANIFEST, comps, root / "runs"), 3)
    assert result.status == "ok", result.error
    assert not any(event.kind.startswith("task.cohort_") for event in RunBundle.open(result.bundle_path).events())


async def test_cohort_rejects_missing_native_runtime(world):  # noqa: F811
    manifest, components, root = world
    manifest.task_start = "cohort"
    result = await run_world(manifest, MANIFEST, components(), root / "runs")
    assert result.status == "failed" and "requires the native task runtime" in result.error


@pytest.mark.parametrize("change", [
    {"mode": "persistent"}, {"execution": {"backend": "another"}},
    {"task_assignment": {"policy": "round_robin", "participants": 1}},
    {"tasks": [{"path": "tasks/first", "stagger_seconds": 1}]},
])
def test_cohort_rejects_unsupported_schedules(world, change):  # noqa: F811
    manifest, _, _ = world
    data = {**manifest.model_dump(), "task_start": "cohort", **change}
    with pytest.raises(ValueError, match="cohort task start"):
        WorldManifest.model_validate(data)
