"""Strict fetched-run graph contracts; every fixture here is software-only evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil

import pytest
import yaml
from test_reproduction_runtime import fetched
from test_reproduction_runtime import loop as loop
from test_reproduction_service import world_file
from test_structure_runtime import graph_world, run_graph
from test_structure_runtime import world as world

from mesocosm.core.manifest import WorldManifest
from mesocosm.core.records import EventLink
from mesocosm.core.rewards import outcome_records
from mesocosm.reproduction.structure_lineage import verify_structure


async def test_actual_controller_records_satisfy_graph_lineage_contract(world, tmp_path, monkeypatch):
    result, events, _, _, _ = await run_graph(world, tmp_path, monkeypatch)
    assert result.status == "ok"
    manifest = graph_world()
    manifest.tasks[0].name = "hello-world"
    receipt = verify_structure(result.bundle_path, manifest, [event.model_dump(mode="json") for event in events])
    assert receipt["finalizers"] == ["agent-3"] and receipt["live_native_acceptance"] == "unverified"


def graph_candidate(loop, tmp_path):
    bench, _, bound, deployments = loop
    path = world_file(tmp_path / "graph-source")
    raw = yaml.safe_load(path.read_text())
    workers, final = [agent["id"] for agent in raw["agents"][:2]], raw["agents"][-1]["id"]
    raw["isolate_agents"] = True
    raw["services"] = {"enabled": True, "chat": {"enabled": True},
                       "peer_review": {"enabled": True, "protocol": "graph"},
                       "access": {"services": {"chat": {"read": [final], "write": [final]}}, "artifacts": {}}}
    raw["structure"] = {"reviews": {final: workers},
                        "execution": {"depends_on": {final: workers}, "finalizers": [final]}}
    path.write_text(yaml.safe_dump(raw))
    behavior = bench.store.get(bound, "experiment")["behavior_id"]
    return bench.configure_experiment(str(path), behavior, [], deployment_id=deployments["base"])["experiment_id"]


def graph_records(bundle, native):
    world = WorldManifest.model_validate(json.loads((bundle.root / "manifest.json").read_text())["manifest"])
    graph = world.structure
    original = [event.model_dump(mode="json") for event in bundle.events()]
    records = [event for event in original if event["kind"] == "gateway.chat_completion"]

    def record(kind, agent=None, links=None, **data):
        event = bundle.record(kind, agent_id=agent, links=links or [], **data).model_dump(mode="json")
        records.append(event)
        return event

    def link(event, relation="observed"):
        return EventLink(event_id=event["event_id"], relation=relation, evidence="software contract fixture")

    verdicts = []
    for task in world.tasks:
        completed = {}
        assignment = {agent.id: graph.reviews.get(agent.id, []) for agent in world.agents}
        record("peer_review.assignment", task=task.name, assignment=assignment, topology="graph")
        for agent in world.agents:
            pair = agent.id, task.name
            prior = graph.execution.depends_on.get(agent.id, [])
            final = agent.id in graph.execution.finalizers
            record("structure.stage_admitted", agent.id, task=task.name, predecessors=prior, finalizer=final,
                   links=[link(completed[parent]) for parent in prior])
            transcript = copy.deepcopy(next(event for event in original if event["kind"] == "harness.transcript"
                and (event["agent_id"], event["data"]["task"]) == pair))
            record("harness.session", agent.id, task=task.name, session_id=transcript["data"]["session_id"])
            records.extend(event for event in original if event["kind"] == "harness.turn"
                           and (event["agent_id"], event["data"]["task"]) == pair)
            records.append(transcript)
            record("harness.events_closed", agent.id, task=task.name, status="closed", planned=True,
                   collector_failed=False, session_id=transcript["data"]["session_id"], links=[link(transcript)])
            selected = []
            for subject in graph.reviews.get(agent.id, []):
                verdict = {"from": agent.id, "about": subject, "epoch": task.name, "verdict": "accept"}
                selected.append(verdict)
                verdicts.append({"service": "chat", "op": "verdict", "agent": agent.id,
                                 "about": subject, "epoch": task.name, "verdict": "accept"})
            if selected:
                record("structure.reviews_checked", agent.id, task=task.name, status="complete",
                       subjects=graph.reviews[agent.id], verdicts=selected, links=[link(transcript)])
            if final:
                record("structure.finalization_admitted", agent.id, task=task.name,
                       original_verifier=True, links=[link(transcript)])
                reward = copy.deepcopy(next(event for event in original if event["kind"] == "verifier.reward"
                    and (event["agent_id"], event["data"]["task"]) == pair))
                reward["links"] = [link(transcript, "measured").model_dump()]
                records.append(reward)
            else:
                shutil.rmtree(bundle.root / "verifier" / agent.id / task.name)
                native_receipt = {"status": "complete", "cleanup_verified": True, "original_verifier_invoked": False,
                    "agent_id": agent.id, "task": task.name, "container_id": "a" * 64,
                    "quiesced_containers": {agent.id: "a" * 64}, "artifacts_sha256": None}
                directory = bundle.root / "work" / agent.id / task.name
                directory.mkdir(parents=True)
                path = directory / "completion.json"
                path.write_text(json.dumps(native_receipt, sort_keys=True))
                receipt = {**native_receipt, "completion_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                record("structure.work_completed", agent.id, task=task.name, receipt=receipt, links=[link(transcript)])
            completed[agent.id] = record("structure.stage_completed", agent.id, task=task.name,
                                         finalizer=final, original_reward=0 if final else None)
    records.extend(event for event in original if event["kind"] == "runtime.cleanup")
    (bundle.root / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in records))
    services = bundle.root / "native/services/services.jsonl"
    services.parent.mkdir()
    services.write_text("".join(json.dumps(row) + "\n" for row in verdicts))
    summary = json.loads((bundle.root / "summary.json").read_text())
    summary["rewards"] = {f"{agent.id}/{task.name}": 0 if agent.id in graph.execution.finalizers else None
                          for agent in world.agents for task in world.tasks}
    summary.update(outcome_records(bundle.events()))
    (bundle.root / "summary.json").write_text(json.dumps(summary))


def test_fetched_graph_qualifies_without_grading_workers(loop, tmp_path):
    bench = loop[0]
    candidate = graph_candidate(loop, tmp_path)
    job, run = fetched(bench, candidate, tmp_path, graph_records)
    assignment = bench.evaluate_run(job, run)["evaluation"]["assignment"]
    assert assignment["structure"] == "verified_native_graph_records"
    assert assignment["finalizers"] == ["agent-2"]


@pytest.mark.parametrize("failure", ["missing_admission", "duplicate_completion", "predecessor_link", "cleanup",
    "completion_bytes", "missing_review", "invented_verdict", "early_verifier", "worker_grading"])
def test_fetched_graph_rejects_changed_gates_and_original_artifacts(loop, tmp_path, failure):
    bench = loop[0]
    candidate = graph_candidate(loop, tmp_path)

    def mutate(bundle, native):
        graph_records(bundle, native)
        path = bundle.root / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == "missing_admission":
            events.remove(next(event for event in events if event["kind"] == "structure.stage_admitted"))
        elif failure == "duplicate_completion":
            events.append(copy.deepcopy(next(event for event in events if event["kind"] == "structure.stage_completed")))
        elif failure == "predecessor_link":
            next(event for event in events if event["kind"] == "structure.stage_admitted"
                 and event["data"]["predecessors"])["links"] = []
        elif failure == "cleanup":
            next(event for event in events if event["kind"] == "structure.work_completed")["data"]["receipt"][
                "cleanup_verified"] = False
        elif failure == "completion_bytes":
            next(bundle.root.glob("work/*/*/completion.json")).write_text("{}")
        elif failure == "missing_review":
            events.remove(next(event for event in events if event["kind"] == "structure.reviews_checked"))
        elif failure == "invented_verdict":
            next(event for event in events if event["kind"] == "structure.reviews_checked")["data"]["verdicts"][0][
                "verdict"] = "reject"
        elif failure == "early_verifier":
            reward = next(event for event in events if event["kind"] == "verifier.reward")
            events.remove(reward)
            events.insert(0, reward)
        elif failure == "worker_grading":
            reward = copy.deepcopy(next(event for event in events if event["kind"] == "verifier.reward"))
            reward["agent_id"] = "agent-0"
            events.append(reward)
        path.write_text("".join(json.dumps(event) + "\n" for event in events))

    job, run = fetched(bench, candidate, tmp_path, mutate)
    with pytest.raises(ValueError, match="runtime lineage rejected"):
        bench.evaluate_run(job, run)
