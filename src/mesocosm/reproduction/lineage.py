"""Admit a fetched native run only after checking its original configuration and records.

This verifies trusted-worker provenance and serving API attribution, not remote
attestation of GPU tensors or causal links between every tool and model token.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

from mesocosm.adapters.harbor.task import read_reward
from mesocosm.adapters.opencode.trajectory import TaskTurn, collect_turn
from mesocosm.core.evaluation import input_snapshot
from mesocosm.core.manifest import WorldManifest
from mesocosm.core.records import RunBundle
from mesocosm.core.rewards import outcome_records
from mesocosm.core.run import HarnessTurnIncomplete, _require_complete_turn
from mesocosm.core.serving import response_identity
from mesocosm.execution.binding import verify_runtime_manifest
from mesocosm.gateway.app import parse_sse, request_summary, summarize_response
from mesocosm.gateway.capture import MAX_BODY_BYTES, DecodedBody
from mesocosm.reproduction.deployment import serving_identity
from mesocosm.reproduction.store import digest, file_hash
from mesocosm.reproduction.structure_lineage import verify_structure


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError("runtime lineage rejected: " + message)


def _rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_wire(record: dict[str, Any]) -> None:
    capture = record["raw_capture"]
    request, response = capture["request"], capture["response"]
    _require(response["capture_complete"] is True and response["transport_complete"] is True
             and response["termination"] == "eof", "incomplete original response capture")
    raw_request = base64.b64decode(request["body_base64"], validate=True)
    _require(len(raw_request) == request["byte_count"] <= MAX_BODY_BYTES
             and hashlib.sha256(raw_request).hexdigest() == request["sha256"]
             and json.loads(raw_request) == record["request"], "original request differs from normalized request")
    pieces = []
    total = 0
    for index, chunk in enumerate(response["chunks"]):
        part = base64.b64decode(chunk["body_base64"], validate=True)
        total += len(part)
        _require(chunk["sequence"] == index and chunk["byte_count"] == len(part) and total <= MAX_BODY_BYTES,
                 "invalid response chunk inventory")
        pieces.append(part)
    raw = b"".join(pieces)
    sha = hashlib.sha256(raw).hexdigest()
    _require(len(raw) == response["byte_count"] == response["captured_byte_count"]
             and sha == response["sha256"] == response["captured_sha256"]
             and len(pieces) == response["received_chunk_count"], "original response bytes changed")
    encoding = response["content_encoding"] if response["body_representation"] == "content_encoded_http_body" else None
    decoder = DecodedBody(encoding, MAX_BODY_BYTES)
    decoder.receive(raw)
    decoder.finish()
    parsed = parse_sse(decoder.text()) if record["stream"] else summarize_response(json.loads(decoder.text()))
    _require(parsed == record["response"], "original response differs from normalized response")


def verify_run(root: pathlib.Path, packaged: dict[str, Any], package_digest: str, remote_job_id: str,
               deployment_id: str | None, deployed: dict[str, Any], packaged_path: pathlib.Path, *,
               agent_deployments: dict[str, str] | None = None) -> dict[str, Any]:
    binding = json.loads((root / "worker-binding.json").read_text())
    original = json.loads((root / "manifest.json").read_text())
    world = WorldManifest.model_validate(original["manifest"])
    summary = json.loads((root / "summary.json").read_text())
    _require(binding["schema_version"] == 1 and binding["job_id"] == remote_job_id
             and binding["package_digest"] == package_digest and binding["inputs_verified_after_run"] is True,
             "worker receipt does not bind the expected input package/job")
    _require(original["digest"] == world.digest() == binding["runtime_manifest_digest"], "runtime manifest digest")
    verify_runtime_manifest(packaged, original["manifest"], pathlib.Path(binding["input_base"]))
    expected_inputs = input_snapshot(packaged_path)
    actual_inputs = binding["input_snapshot"]
    for field in ("tasks", "external_inputs"):
        _require([{key: value for key, value in entry.items() if key != "path"} for entry in actual_inputs[field]]
                 == [{key: value for key, value in entry.items() if key != "path"} for entry in expected_inputs[field]],
                 "worker task/verifier/external input bytes differ from frozen package")
    _require(actual_inputs["integrations"] == expected_inputs["integrations"]
             and actual_inputs["manifest_digest"] == expected_inputs["manifest_digest"],
             "worker input snapshot differs from frozen package")
    _require(world.execution.backend == "docker" and world.mode == "comparison"
             and world.task_assignment.policy == "all" and len(world.agents) >= 3,
             "verified runs currently require native Docker comparison/all with at least three agents")
    _require(summary["status"] == "ok", "incomplete run remains a failed attempt, not an admitted evaluation")
    label = world.gateway.run_label or root.name
    _require(summary["run_label"] == label, "run label mismatch")
    if agent_deployments is None:
        _require(deployment_id is not None, "missing deployment binding")
        assert deployment_id is not None
        agent_deployments = {agent.id: deployment_id for agent in world.agents}
        deployments = {deployment_id: deployed}
    else:
        deployments = deployed
    _require(set(agent_deployments) == {agent.id for agent in world.agents}, "incomplete per-agent deployment binding")
    identities = {agent: serving_identity(identifier, deployments[identifier])
                  for agent, identifier in agent_deployments.items()}
    for agent in world.agents:
        expected_identity = identities[agent.id]
        deployment = deployments[agent_deployments[agent.id]]
        policy = deployment["policy"]
        model = world.model_for(agent)
        _require(model.serving_identity is not None and model.serving_identity.model_dump() == expected_identity
                 and model.id == policy["model"] and model.upstream == deployment["server"]
                 and model.revision == policy["base_revision"] and model.hf_repo == policy["base_model"]
                 and model.api_key_env == deployment["server_token_env"], "candidate model/deployment mismatch")
    events = _rows(root / "events.jsonl")
    _require(all(event.get("run_id") == root.name for event in events), "event run identity changed")
    _require(any(event["kind"] == "runtime.cleanup" and event["data"].get("status") == "complete"
                 for event in events), "missing native cleanup evidence")
    _require(not any(event["kind"] in {"run.failed", "run.aborted", "run.stopped", "harness.failed",
                     "verifier.failed", "harness.round_timeout", "harness.task_timeout", "harness.events_ended"}
                     or event["kind"] == "runtime.cleanup" and event["data"].get("status") != "complete"
                     for event in events), "contradictory failed/incomplete execution evidence")
    outcomes = outcome_records(RunBundle(root, root.name).events())
    for field in ("task_outcomes", "learning_rewards"):
        _require(summary.get(field) == outcomes[field], "summary outcomes differ from source events")
    native = _rows(root / "native/gateway/records.jsonl")
    agents = {agent.id for agent in world.agents}
    calls = [row for row in native if row.get("event") == "chat_completion"]
    admissions = [row for row in native if row.get("event") == "request_admitted"]
    call_ids = [row["request_id"] for row in calls]
    _require(call_ids and len(set(call_ids)) == len(call_ids)
             and Counter(call_ids) == Counter(row["request_id"] for row in admissions),
             "duplicate, missing or unsettled model request")
    normalized = [event for event in events if event["kind"] == "gateway.chat_completion"]
    _require(Counter((event["request_id"], event["agent_id"]) for event in normalized)
             == Counter((row["request_id"], row["agent"]) for row in calls), "normalized/native call mismatch")
    by_id = {row["request_id"]: row for row in calls}
    for event in normalized:
        record = by_id[event["request_id"]]
        expected = {"status": record.get("status"),
                    "request_summary": record.get("request_summary"), "usage": record["response"].get("usage")}
        _require(event["data"] == expected and (record.get("ts") is None or event["ts"] == record["ts"]),
                 "normalized model event differs from original projection")
    pending = {}
    for record in native:
        if record.get("event") == "request_admitted":
            pending[record["request_id"]] = record
        elif record.get("event") == "chat_completion":
            admission = pending.pop(record["request_id"], None)
            _require(admission is not None and all(admission.get(key) == record.get(key) for key in
                ("agent", "run", "identified_by", "expected_model", "serving_identity", "upstream", "model_kind")),
                "completion lacks an earlier matching admission")
    _require(set(row["agent"] for row in calls) == agents and summary["model_calls"] == len(calls),
             "missing agent calls or mismatched call count")
    for record in [*admissions, *calls]:
        _require(record["agent"] in agents, "native request names an unknown agent")
        deployment = deployments[agent_deployments[record["agent"]]]
        policy = deployment["policy"]
        _require(record["run"] == label and record["agent"] in agents and record["identified_by"] == "key"
                 and record.get("expected_model") == policy["model"]
                 and record.get("serving_identity") == identities[record["agent"]]
                 and record["upstream"].rstrip("/") == deployment["server"].rstrip("/")
                 and record["model_kind"] == "vllm",
                 "native request attribution differs from frozen deployment")
    for record in calls:
        policy = deployments[agent_deployments[record["agent"]]]["policy"]
        response = record["response"]
        _require(record["status"] == 200 and record["request"]["model"] == policy["model"]
                 and response_identity(response, policy["model"]) == "matched"
                 and record.get("model_identity_check") == "matched", "wrong or missing response model")
        _require(not response.get("stream_errors") and not response.get("unparsed_lines")
                 and (not record["stream"] or response.get("stream_complete") is True), "incomplete model stream")
        _require(not response.get("error") and response.get("choice_indices") == [0]
                 and response.get("finish_reason") in {"stop", "length", "tool_calls", "content_filter"},
                 "missing or errored model completion")
        _require(record.get("request_summary") == request_summary(record["request"]),
                 "request summary contradicts original request")
        verify_wire(record)
    task_pairs = {(agent.id, task.name) for agent in world.agents for task in world.tasks}
    transcripts = [event for event in events if event["kind"] == "harness.transcript"]
    rewards = [event for event in events if event["kind"] == "verifier.reward"]
    reward_pairs = ({pair for pair in task_pairs if pair[0] in world.structure.execution.finalizers}
                    if world.structure is not None else task_pairs)
    for records, expected_pairs in ((transcripts, task_pairs), (rewards, reward_pairs)):
        _require(Counter((event["agent_id"], event["data"].get("task")) for event in records)
                 == Counter(expected_pairs), "missing or duplicate original task/session/verifier record")
    structure = verify_structure(root, world, events)
    for event in transcripts:
        path = root / event["data"]["path"]
        _require(path.resolve().is_relative_to(root.resolve()) and file_hash(path) == event["data"]["sha256"],
                 "original harness transcript changed")
    sessions: dict[str, str] = {}
    session_by_task = {}
    history_by_task = {}
    for event in transcripts:
        session = event["data"].get("session_id")
        _require(isinstance(session, str) and session and sessions.get(session, event["agent_id"]) == event["agent_id"],
                 "missing harness session or session shared by independent agents")
        sessions[session] = event["agent_id"]
        session_by_task[event["agent_id"], event["data"]["task"]] = session
        history_by_task[event["agent_id"], event["data"]["task"]] = json.loads(
            (root / event["data"]["path"]).read_text())
    turns = [event for event in events if event["kind"] == "harness.turn"]
    _require({(event["agent_id"], event["data"].get("task")) for event in turns} == task_pairs,
             "missing original harness turns")
    for event in turns:
        path = root / event["data"]["path"]
        _require(path.resolve().is_relative_to(root.resolve()) and file_hash(path) == event["data"]["sha256"],
                 "original harness turn changed")
        turn = TaskTurn.model_validate_json(path.read_text())
        pair = event["agent_id"], event["data"]["task"]
        _require(turn.session_id == event["data"].get("session_id") == session_by_task[pair],
                 "misbound harness turn")
        _require(collect_turn(history_by_task[pair], user_message_id=turn.user_message_id,
                              reply=turn.reply) == turn, "turn contradicts original complete session history")
        try:
            _require_complete_turn(turn)
        except HarnessTurnIncomplete as exc:
            raise ValueError("runtime lineage rejected: " + str(exc)) from exc
        projection = {"session_id": turn.session_id, "user_message_id": turn.user_message_id,
            "final_message_id": turn.final_message_id, "finish_reason": turn.finish_reason,
            "messages": len(turn.messages), "text_parts": len(turn.text_parts),
            "message_error_ids": list(turn.message_errors), "length_message_ids": turn.length_message_ids,
            "unfinished_message_ids": turn.unfinished_message_ids}
        _require(all(event["data"].get(key) == value for key, value in projection.items()),
                 "normalized harness turn differs from its original")
    reward_matrix = {f"{agent}/{task}": None for agent, task in task_pairs} if world.structure is not None else {}
    reward_matrix.update({f"{event['agent_id']}/{event['data']['task']}": event["data"].get("reward")
                          for event in rewards})
    _require(summary.get("rewards") == reward_matrix, "summary verifier matrix mismatch")
    for event in rewards:
        logs = root / "verifier" / event["agent_id"] / event["data"]["task"]
        _require(logs.resolve().is_relative_to(root.resolve()) and (logs / "environment.json").is_file()
                 and (logs / "verifier.log").is_file(), "missing original verifier environment/logs")
        measured = read_reward(logs)
        _require(measured.value is not None and measured.value == event["data"].get("reward")
                 and measured.source == event["data"].get("source")
                 and measured.labels == (event["data"].get("labels") or {})
                 and measured.checkpoints == (event["data"].get("checkpoints") or {}),
                 "original verifier score or metadata mismatch")
    return {"run_id": root.name, "runtime_manifest_digest": world.digest(), "model_calls": len(calls),
            "agents": sorted(agents), "request_ids": sorted(call_ids), "deployment_id": deployment_id,
            **({"agent_deployments": agent_deployments} if deployment_id is None else {}),
            "worker_binding_sha256": file_hash(root / "worker-binding.json"),
            "gateway_records_sha256": file_hash(root / "native/gateway/records.jsonl"),
            "costs": {"model_requests": len(calls), **{
                field: sum(row["response"]["usage"][field] for row in calls) if all(
                    type((row["response"].get("usage") or {}).get(field)) is int for row in calls) else None
                for field in ("prompt_tokens", "completion_tokens")}},
            "task_set_sha256": digest(sorted((task["source"], task["tree_sha256"])
                                               for task in binding["input_snapshot"]["tasks"])),
            "binding": "verified_worker_runtime_and_serving_attribution",
            **structure,
            "numerical_weight_verification": "unverified", "exact_call_to_tool_causal_links": "unverified"}
