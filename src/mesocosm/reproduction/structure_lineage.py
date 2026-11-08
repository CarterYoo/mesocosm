"""Reconcile graph admission with native lifecycle artifacts and original service verdicts."""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections import Counter
from typing import Any

from mesocosm.core.manifest import WorldManifest


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError("runtime lineage rejected: structure " + message)


def verify_structure(root: pathlib.Path, world: WorldManifest, events: list[dict[str, Any]]) -> dict[str, Any]:
    graph = world.structure
    if graph is None:
        _require(not any(event["kind"].startswith("structure.") for event in events),
                 "events exist without a declared graph")
        return {}
    agents, tasks = {agent.id for agent in world.agents}, {task.name for task in world.tasks}
    finals = set(graph.execution.finalizers)
    pairs = {(agent, task) for agent in agents for task in tasks}
    worker_pairs = {(agent, task) for agent, task in pairs if agent not in finals}
    final_pairs = pairs - worker_pairs
    event_ids = [event.get("event_id") for event in events]
    _require(all(isinstance(value, str) and value for value in event_ids)
             and len(set(event_ids)) == len(event_ids), "event identities missing or duplicated")
    order = {event["event_id"]: index for index, event in enumerate(events)}

    def inventory(kind: str, expected: set[tuple[str, str | None]]) -> dict[tuple[str, str | None], dict[str, Any]]:
        selected = [event for event in events if event["kind"] == kind]
        _require(Counter((event.get("agent_id"), event["data"].get("task")) for event in selected)
                 == Counter(expected), f"missing or duplicate {kind} records")
        return {(event["agent_id"], event["data"]["task"]): event for event in selected}

    admissions = inventory("structure.stage_admitted", pairs)
    completed = inventory("structure.stage_completed", pairs)
    workers = inventory("structure.work_completed", worker_pairs)
    finalization = inventory("structure.finalization_admitted", final_pairs)
    sessions = inventory("harness.session", pairs)
    transcripts = inventory("harness.transcript", pairs)
    closures = inventory("harness.events_closed", pairs)
    rewards = inventory("verifier.reward", final_pairs)
    required_reviews = {(agent, task) for agent, subjects in graph.reviews.items() if subjects for task in tasks}
    reviewed = inventory("structure.reviews_checked", required_reviews)

    def before(first: dict[str, Any], second: dict[str, Any]) -> bool:
        return order[first["event_id"]] < order[second["event_id"]]

    def linked(event: dict[str, Any], sources: list[dict[str, Any]], relation: str) -> bool:
        return Counter((link.get("event_id"), link.get("relation")) for link in event.get("links", [])) == Counter(
            (source["event_id"], relation) for source in sources)

    for pair in pairs:
        agent, task = pair
        admitted, ended, session, transcript, closed = (
            admissions[pair], completed[pair], sessions[pair], transcripts[pair], closures[pair])
        predecessors = graph.execution.depends_on.get(agent, [])
        previous = [completed[parent, task] for parent in predecessors]
        _require(admitted["data"].get("predecessors") == predecessors
                 and admitted["data"].get("finalizer") is (agent in finals)
                 and ended["data"].get("finalizer") is (agent in finals)
                 and linked(admitted, previous, "observed")
                 and all(before(event, admitted) for event in previous), "predecessor admission differs from graph")
        _require(before(admitted, session) and before(session, transcript) and before(transcript, closed)
                 and before(closed, ended), "stage/session/transcript/closure ordering changed")
        _require(closed["data"].get("planned") is True and closed["data"].get("collector_failed") is False
                 and closed["data"].get("status") == "closed"
                 and closed["data"].get("session_id") == transcript["data"].get("session_id")
                 and linked(closed, [transcript], "observed"), "authoritative event-stream closure is missing")
        turns = [event for event in events if event["kind"] == "harness.turn"
                 and (event.get("agent_id"), event["data"].get("task")) == pair]
        _require(turns and all(before(session, event) and before(event, transcript) for event in turns),
                 "harness turns occurred outside admitted stage")
        if pair in reviewed:
            checked = reviewed[pair]
            _require(before(closed, checked) and before(checked, ended)
                     and checked["data"].get("status") == "complete"
                     and checked["data"].get("subjects") == graph.reviews[agent]
                     and linked(checked, [transcript], "observed"), "required review gate differs from graph")
        if pair in workers:
            event, receipt = workers[pair], workers[pair]["data"].get("receipt", {})
            completion = root / "work" / agent / str(task) / "completion.json"
            _require(completion.is_file() and hashlib.sha256(completion.read_bytes()).hexdigest()
                     == receipt.get("completion_sha256"), "native work completion artifact changed")
            _require(json.loads(completion.read_text()) == {key: value for key, value in receipt.items()
                       if key != "completion_sha256"}, "native work completion contradicts controller receipt")
            container = receipt.get("container_id")
            _require(receipt.get("status") == "complete" and receipt.get("cleanup_verified") is True
                     and receipt.get("original_verifier_invoked") is False
                     and receipt.get("agent_id") == agent and receipt.get("task") == task
                     and isinstance(container, str) and re.fullmatch(r"[a-f0-9]{64}", container)
                     and receipt.get("quiesced_containers") == {agent: container}, "unverified native work quiescence")
            _require(before(closed, event) and before(event, ended) and linked(event, [transcript], "observed")
                     and ended["data"].get("original_reward") is None
                     and not (root / "verifier" / agent / str(task)).exists(),
                     "subordinate was graded or not completed")
            if world.artifacts.enabled:
                path = root / "native" / agent / str(task) / "artifacts.json"
                _require(path.is_file(), "declared work artifacts absent")
                snapshot = json.loads(path.read_text())
                _require(hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
                         == receipt.get("artifacts_sha256"), "work artifact capture differs from native completion")
            else:
                _require(receipt.get("artifacts_sha256") is None, "undeclared work artifacts")
        else:
            event, reward = finalization[pair], rewards[pair]
            _require(before(closed, event) and before(event, reward) and before(reward, ended)
                     and event["data"].get("original_verifier") is True
                     and linked(event, [transcript], "observed") and linked(reward, [transcript], "measured")
                     and ended["data"].get("original_reward") == reward["data"].get("reward"),
                     "original verifier ran before designated finalization or differs from final outcome")
            if pair in reviewed:
                _require(before(reviewed[pair], event), "finalization preceded required review gate")
    if graph.reviews:
        assignments = [event for event in events if event["kind"] == "peer_review.assignment"]
        _require(Counter(event["data"].get("task") for event in assignments) == Counter(tasks),
                 "review assignment inventory changed")
        wanted = {agent: graph.reviews.get(agent, []) for agent in agents}
        for event in assignments:
            task = event["data"]["task"]
            _require(event["data"].get("assignment") == wanted and event["data"].get("topology") == "graph"
                     and all(before(event, admissions[agent, task]) for agent in agents),
                     "review assignments were not applied before stage admission")
        path = root / "native/services/services.jsonl"
        _require(path.is_file(), "original service verdict journal missing")
        native = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        raw = [row for row in native if row.get("service") == "chat" and row.get("op") == "verdict"]
        def projection(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
            return row.get("agent", row.get("from")), row.get("about"), row.get("epoch"), row.get("verdict")
        selected = [verdict for event in reviewed.values() for verdict in event["data"].get("verdicts", [])]
        expected = Counter((agent, subject, task) for agent, subjects in graph.reviews.items()
                           for subject in subjects for task in tasks)
        _require(Counter(projection(row)[:3] for row in raw) == expected
                 and Counter(projection(row)[:3] for row in selected) == expected
                 and Counter(projection(row) for row in selected) == Counter(projection(row) for row in raw)
                 and all(row.get("verdict") in {"accept", "reject"} for row in raw),
                 "required review gate contradicts original verdict journal")
    return {"structure": "verified_native_graph_records", "finalizers": sorted(finals),
            "subordinate_measurement": "completed_ungraded", "live_native_acceptance": "unverified"}
