"""Software graph contracts, not native multi-agent acceptance evidence."""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil

import httpx
import pytest
from pydantic import ValidationError
from test_run_e2e import MANIFEST
from test_run_e2e import world as world
from test_services import client, settings

from mesocosm.core.access import AccessSpec
from mesocosm.core.manifest import AgentSpec, WorldManifest, load_manifest
from mesocosm.core.records import RunBundle
from mesocosm.core.run import RunResult, run_world
from mesocosm.core.structure import StructureSpec, structure_errors
from mesocosm.env.services.app import create_app
from mesocosm.env.services.client import ServicesAdmin


def graph_world():
    manifest = load_manifest(MANIFEST)
    manifest.agents = [AgentSpec(id=f"agent-{index}") for index in range(1, 4)]
    manifest.isolate_agents = True
    manifest.services.chat.enabled = True
    manifest.services.peer_review.enabled = True
    manifest.services.peer_review.protocol = "graph"
    manifest.structure = StructureSpec.model_validate({
        "reviews": {"agent-3": ["agent-1", "agent-2"]},
        "execution": {"depends_on": {"agent-3": ["agent-1", "agent-2"]}, "finalizers": ["agent-3"]}})
    manifest.services.access = AccessSpec(services={"chat": {"read": ["agent-3"], "write": ["agent-3"]}},
                                         artifacts={})
    return manifest


def test_graph_is_opt_in_and_three_graphs_remain_distinct():
    assert load_manifest(MANIFEST).structure is None
    manifest = graph_world()
    assert structure_errors(manifest) == []
    parsed = WorldManifest.model_validate(manifest.model_dump())
    assert parsed.structure.waves([member.id for member in manifest.agents]) == [["agent-1", "agent-2"], ["agent-3"]]
    assert parsed.structure.work == {} and parsed.services.access.direct_messages == {}


@pytest.mark.parametrize("change,match", [
    ({"depends_on": {"agent-1": ["agent-3"], "agent-3": ["agent-1", "agent-2"]}}, "cycle"),
    ({"depends_on": {"agent-3": ["outsider"]}}, "unknown agents"),
    ({"depends_on": {"agent-3": ["agent-1"]}}, "every reviewed subject"),
    ({"finalizers": ["outsider"]}, "unknown agents"),
    ({"finalizers": ["agent-1", "agent-3"]}, "execution sinks"),
])
def test_invalid_execution_graphs_fail_before_execution(change, match):
    raw = graph_world().model_dump()
    raw["structure"]["execution"].update(change)
    with pytest.raises(ValidationError, match=match):
        WorldManifest.model_validate(raw)


@pytest.mark.parametrize("path,value,match", [
    (("mode",), "persistent", "isolated comparison"),
    (("memory",), "carry", "isolated comparison"),
    (("task_services",), "shared", "isolated comparison"),
    (("rewards", "enabled"), True, "episode barriers"),
    (("history", "episodes"), 1, "episode barriers"),
    (("task_start",), "cohort", "as_ready"),
    (("isolate_agents",), False, "isolated agents"),
    (("services", "access", "artifacts"), None, "exactly match"),
    (("services", "access", "services", "chat", "write"), [], "chat write"),
])
def test_unsupported_conditions_and_visibility_bypasses_are_rejected(path, value, match):
    raw = graph_world().model_dump()
    target = raw
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match=match):
        WorldManifest.model_validate(raw)


class WorkCompletion:
    def __init__(self):
        self.completed = []

    async def prepare(self, agent, task, workspace, **kwargs):
        return {"status": "software_fixture"}

    async def complete_work(self, task, workspace, logs, spec):
        agent_id = workspace.name
        self.completed.append(agent_id)
        receipt = {"status": "complete", "cleanup_verified": True, "original_verifier_invoked": False,
                   "agent_id": agent_id, "task": task.name, "artifacts_sha256": None,
                   "container_id": "a" * 64, "quiesced_containers": {agent_id: "a" * 64}}
        record = logs / "completion.json"
        record.write_text(json.dumps(receipt, sort_keys=True))
        return {**receipt, "artifacts": None, "completion_sha256": hashlib.sha256(record.read_bytes()).hexdigest()}


async def run_graph(world, tmp_path, monkeypatch, *, reviews=True, block=False, cancel=False):
    _, components, _ = world
    manifest = graph_world()
    manifest.gateway.run_label = "e2e"
    comps = components()
    runtime = WorkCompletion()
    comps.task_runtime = runtime
    st = settings(tmp_path)
    st.agent_ips["172.30.0.13"] = "agent-3"
    st.access = manifest.services.access
    st.chat = manifest.services.chat.model_dump()
    st.peer_review = manifest.services.peer_review.model_dump()
    app = create_app(st)
    admin = ServicesAdmin("http://services", "test-control",
                          transport=httpx.ASGITransport(app=app, client=("172.30.0.4", 1)))
    comps.services, comps.services_log = admin, st.log_path
    original_factory = comps.harness_for
    original_verify = comps.verifier.verify
    started, graded, interrupted = [], [], asyncio.Event()

    def harness_for(agent):
        harness = original_factory(agent)
        send = harness.send_message

        async def message(*args, **kwargs):
            started.append(agent.id)
            if agent.id == "agent-3":
                assert set(runtime.completed) == {"agent-1", "agent-2"}
            if block and agent.id == "agent-1":
                interrupted.set()
                await asyncio.Event().wait()
            reply = await send(*args, **kwargs)
            if reviews and agent.id == "agent-3":
                async with client(app, "172.30.0.13", "chat") as reviewer:
                    for subject in ("agent-1", "agent-2"):
                        response = await reviewer.post("/api/verdicts", json={"about": subject, "verdict": "accept"})
                        assert response.status_code == 201
            return reply

        harness.send_message = message
        return harness

    async def verify(task, workspace, logs, script="test.sh"):
        graded.append(workspace.name)
        return await original_verify(task, workspace, logs, script)

    comps.harness_for, comps.verifier.verify = harness_for, verify
    monkeypatch.setattr("mesocosm.core.run.EVENT_GRACE_SECONDS", 0.001)
    if block and not cancel:
        source = MANIFEST.parent / "tasks" / "hello-world"
        task_root = tmp_path / "deadline-task"
        shutil.copytree(source, task_root)
        with (task_root / "task.toml").open("a") as stream:
            stream.write("\n[agent]\ntimeout_sec = 0.03\n")
        manifest.tasks[0].path = str(task_root)
    try:
        pending = asyncio.create_task(run_world(manifest, MANIFEST, comps, tmp_path / "runs"))
        if cancel:
            await asyncio.wait_for(interrupted.wait(), 3)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 5)
            root = next((tmp_path / "runs").iterdir())
            summary = json.loads((root / "summary.json").read_text())
            result = RunResult(root.name, root, summary["status"], summary["rewards"], summary["error"])
        else:
            result = await asyncio.wait_for(pending, 5)
    finally:
        await admin.aclose()
    return result, RunBundle.open(result.bundle_path).events(), started, graded, runtime


async def test_workers_finish_before_reviewer_and_only_finalizer_uses_original_verifier(world, tmp_path, monkeypatch):
    result, events, started, graded, runtime = await run_graph(world, tmp_path, monkeypatch)
    assert result.status == "ok" and graded == ["agent-3"]
    assert started[-1] == "agent-3" and set(runtime.completed) == {"agent-1", "agent-2"}
    assert result.rewards == {"agent-1/hello-world": None, "agent-2/hello-world": None, "agent-3/hello-world": 1.0}
    admitted = next(event for event in events if event.kind == "structure.stage_admitted" and event.agent_id == "agent-3")
    completed = [event for event in events if event.kind == "structure.stage_completed"
                 and event.agent_id in {"agent-1", "agent-2"}]
    assert {link.event_id for link in admitted.links} == {event.event_id for event in completed}
    assert len([event for event in events if event.kind == "verifier.reward"]) == 1
    checked = next(event for event in events if event.kind == "structure.reviews_checked")
    assert checked.data["status"] == "complete" and len(checked.data["verdicts"]) == 2


async def test_missing_real_assigned_verdict_blocks_final_grading(world, tmp_path, monkeypatch):
    result, events, _, graded, _ = await run_graph(world, tmp_path, monkeypatch, reviews=False)
    assert result.status != "ok" and not graded
    assert "required assigned review verdicts" in result.error
    assert not any(event.kind == "verifier.reward" for event in events)


@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_or_cancellation_never_admits_or_grades_successor(world, tmp_path, monkeypatch, cancel):
    result, events, started, graded, _ = await run_graph(world, tmp_path, monkeypatch, block=True, cancel=cancel)
    assert result.status == ("cancelled" if cancel else "failed_deadline")
    assert "agent-3" not in started and not graded
    assert not any(event.kind == "structure.finalization_admitted" for event in events)
