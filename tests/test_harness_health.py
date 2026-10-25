"""Local transport/control regressions, not real-agent or deadlock-repair evidence."""
from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import socket
import subprocess
import threading
from types import SimpleNamespace

import httpx
import pytest
from test_run_e2e import MANIFEST
from test_run_e2e import world as world
from test_runtime_controls import Workloads, three_agents

import mesocosm.core.run as runtime
from mesocosm.core.control import STOP_MARKER
from mesocosm.core.manifest import load_manifest
from mesocosm.core.records import RunBundle
from mesocosm.core.rewards import EpisodeStopped
from mesocosm.env.compose import render_compose

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_controlled_curl_probe_exits_itself_when_server_accepts_but_never_replies():
    if shutil.which("curl") is None:
        pytest.skip("curl is needed for the real local healthcheck subprocess")
    manifest = load_manifest(MANIFEST)
    health = render_compose(manifest, repo_root=ROOT, run_label="probe-fixture")["services"]["agent-1"]["healthcheck"]
    command = health["test"][1]
    assert command.startswith("exec curl ")
    assert "--connect-timeout 1" in command and "--max-time 2" in command
    assert health["timeout"] == "3s"
    finished = threading.Event()
    accepted = threading.Event()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(4)

        def hold_connection():
            connection, _ = server.accept()
            with connection:
                accepted.set()
                finished.wait(5)

        thread = threading.Thread(target=hold_connection)
        thread.start()
        try:
            command = command.replace("$$", "$").replace("127.0.0.1:4096", f"127.0.0.1:{server.getsockname()[1]}")
            result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, timeout=4,
                                    env={**os.environ, "OPENCODE_SERVER_USERNAME": "fixture",
                                         "OPENCODE_SERVER_PASSWORD": "fixture", "NO_PROXY": "127.0.0.1"})
            assert accepted.is_set()
            assert result.returncode == 28  # curl's own timeout, not an external process kill
        finally:
            finished.set()
            thread.join(timeout=5)
        assert not thread.is_alive()


def fast_health(monkeypatch):
    monkeypatch.setattr(runtime, "HARNESS_HEALTH_INTERVAL_SECONDS", .01)
    monkeypatch.setattr(runtime, "HARNESS_HEALTH_TIMEOUT_SECONDS", .02)
    monkeypatch.setattr(runtime, "EVENT_GRACE_SECONDS", 0)


async def test_stalled_post_after_three_sessions_is_stopped_without_inference_or_grading(world, monkeypatch):  # noqa: F811
    fast_health(monkeypatch)
    manifest, components, root = world
    ids = three_agents(manifest)
    manifest.task_start = "cohort"
    comps = components()
    workloads = comps.workloads = Workloads()
    original = comps.harness_for
    active, cancelled, aborted = set(), set(), set()
    probe_cancelled = 0

    async def prepare(*args, **kwargs):
        return {"status": "fixture_prepared"}

    def harness_for(agent):
        nonlocal probe_cancelled
        harness = original(agent)

        async def send(*args, **kwargs):
            active.add(agent.id)
            try:
                await asyncio.Event().wait()  # accepted POST, before any gateway call
            finally:
                cancelled.add(agent.id)

        async def health():
            nonlocal probe_cancelled
            if agent.id != "agent-2":
                return {"healthy": True}
            try:
                await asyncio.Event().wait()
            finally:
                probe_cancelled += 1

        async def abort(session_id):
            aborted.add(agent.id)
            return True

        harness.send_message, harness.health, harness.abort_session = send, health, abort
        return harness

    async def never_grade(*args, **kwargs):
        pytest.fail("unresponsive incomplete turns must not be graded")

    comps.task_runtime = SimpleNamespace(prepare=prepare)
    comps.harness_for, comps.verifier.verify = harness_for, never_grade
    result = await asyncio.wait_for(runtime.run_world(manifest, MANIFEST, comps, root / "runs"), 3)
    assert result.status == "failed" and "HarnessUnhealthy" in result.error
    assert active == cancelled == aborted == set(workloads.terminated) == ids
    assert probe_cancelled == 3
    events = RunBundle.open(result.bundle_path).events()
    assert sum(event.kind == "task.cohort_released" for event in events) == 1
    probes = [event for event in events if event.kind == "harness.health" and event.agent_id == "agent-2"]
    assert [event.data["consecutive_failures"] for event in probes] == [1, 2, 3]
    assert all(event.data["session_id"] == "ses_agent-2_1" and event.data["task"] == "hello-world" for event in probes)
    assert all(event.data["ok"] is False and event.data["error_type"] == "TimeoutError" for event in probes)
    assert not any(event.kind in {"harness.turn", "verifier.reward", "harness.task_timeout"} for event in events)
    assert [record async for record in comps.gateway.records("e2e")] == []


async def test_responsive_long_turn_survives_transient_health_failures(world, monkeypatch):  # noqa: F811
    fast_health(monkeypatch)
    manifest, components, root = world
    comps = components()
    original = comps.harness_for
    count = 0

    def harness_for(agent):
        harness = original(agent)
        send = harness.send_message

        async def delayed(*args, **kwargs):
            await asyncio.sleep(.2)  # longer than several unhealthy-watch thresholds
            return await send(*args, **kwargs)

        async def health():
            nonlocal count
            count += 1
            if count % 3 != 0:
                raise httpx.ConnectError("transient fixture connection failure")
            return {"healthy": True}

        harness.send_message, harness.health = delayed, health
        return harness

    comps.harness_for = harness_for
    result = await runtime.run_world(manifest, MANIFEST, comps, root / "runs")
    assert result.status == "ok" and result.rewards == {"agent-1/hello-world": 1.0}
    assert count >= 9
    probes = [event for event in RunBundle.open(result.bundle_path).events() if event.kind == "harness.health"]
    assert any(event.data["consecutive_failures"] == 0 and event.data["ok"] is True for event in probes)
    assert max(event.data["consecutive_failures"] for event in probes) == 2


@pytest.mark.parametrize("ending", ["complete", "cancel", "operator_stop"])
async def test_pending_probe_is_cancelled_and_awaited_when_turn_ends(tmp_path, monkeypatch, ending):
    fast_health(monkeypatch)
    monkeypatch.setattr(runtime, "HARNESS_HEALTH_TIMEOUT_SECONDS", 10)
    bundle = RunBundle.create(tmp_path / "runs", "health-cancel")
    entered, closed, done = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def health():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(.01)
            closed.set()

    async def turn():
        await done.wait()
        return "completed"

    running = asyncio.create_task(runtime._interruptible_turn(turn(), bundle,
        health_monitor=runtime._watch_harness_health(SimpleNamespace(health=health), bundle, "a", "task", "session")))
    await asyncio.wait_for(entered.wait(), 1)
    if ending == "cancel":
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 1)
    elif ending == "operator_stop":
        marker = bundle.root / STOP_MARKER
        marker.parent.mkdir(parents=True)
        marker.write_text("stop")
        with pytest.raises(EpisodeStopped):
            await asyncio.wait_for(running, 1)
    else:
        done.set()
        assert await asyncio.wait_for(running, 1) == "completed"
    assert closed.is_set()
    assert not any(event.kind == "harness.health" for event in bundle.events())


@pytest.mark.parametrize("response", [{"healthy": False}, {"healthy": 1}, {}, []])
async def test_nonhealthy_or_malformed_health_is_a_failure(tmp_path, monkeypatch, response):
    fast_health(monkeypatch)
    bundle = RunBundle.create(tmp_path / "runs", "invalid-health")

    async def health():
        return response

    with pytest.raises(runtime.HarnessUnhealthy):
        await asyncio.wait_for(runtime._watch_harness_health(SimpleNamespace(health=health), bundle,
                                                           "a", "task", "session"), 1)
    probes = bundle.events()
    assert len(probes) == 3 and all(event.data["ok"] is False for event in probes)


@pytest.mark.parametrize("turn_error", [False, True])
async def test_finished_turn_wins_simultaneous_health_failure(tmp_path, turn_error):
    bundle = RunBundle.create(tmp_path / "runs", "health-race")
    loop = asyncio.get_running_loop()
    turn, health = loop.create_future(), loop.create_future()
    running = asyncio.create_task(runtime._interruptible_turn(turn, bundle, health_monitor=health))
    await asyncio.sleep(0)  # supervisor has started waiting on both futures
    health.set_exception(runtime.HarnessUnhealthy("probe failed at completion"))
    if turn_error:
        error = ValueError("original turn failure")
        turn.set_exception(error)
        with pytest.raises(ValueError) as caught:
            await running
        assert caught.value is error
    else:
        turn.set_result("actual completed turn")
        assert await running == "actual completed turn"
    assert health.exception() is not None  # observed by cleanup, never left unconsumed
