"""Native bundle/transport contract fixtures, never real-agent acceptance evidence."""

from __future__ import annotations

import copy
import json
import pathlib

import pytest
import yaml
from test_reproduction_service import dataset_file, world_file

from mesocosm.adapters.opencode.trajectory import collect_turn
from mesocosm.core.config import absolutize_manifest
from mesocosm.core.evaluation import input_snapshot
from mesocosm.core.manifest import WorldManifest, load_manifest
from mesocosm.core.records import RunBundle
from mesocosm.core.rewards import outcome_records
from mesocosm.core.run import _record_turn
from mesocosm.execution.package import export_results, extract_results, unpack_submission
from mesocosm.gateway.app import request_summary, summarize_response
from mesocosm.gateway.capture import BodyCapture
from mesocosm.learning.config import policy_id
from mesocosm.reproduction.deployment import serving_identity
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import file_hash


@pytest.fixture
def loop(tmp_path):
    bench = ResearchWorkbench(
        tmp_path / "research",
        allowed_workers={"http://127.0.0.1:8088": "WORKER_KEY"},
        allowed_servers={"http://127.0.0.1:8000": None},
    )
    dataset = bench.prepare_dataset(
        [str(dataset_file(tmp_path))], {"split_assignments": {"r0": "train", "r1": "development", "r2": "test"}}
    )
    deployments = {}
    for name in ("base", "trained"):
        adapter = "c" * 64 if name == "trained" else None
        policy = {
            "base_model": "/pinned/base",
            "base_revision": "a" * 40,
            "base_sha256": "b" * 64,
            "adapter_sha256": adapter,
            "adapter_path": "/pinned/adapter" if adapter else None,
            "model": name,
            "dtype": "bfloat16",
            "id": policy_id("/pinned/base", "a" * 40, "b" * 64, adapter),
        }
        deployments[name] = bench.store.put(
            "deployment",
            {
                "policy": policy,
                "server": "http://127.0.0.1:8000/",
                "server_token_env": None,
                "training_artifact": {"dataset_id": dataset["dataset_id"]} if adapter else None,
                "evidence_level": "software_fixture",
                "numerical_weight_verification": "unverified",
            },
        )
    path = world_file(tmp_path / "world-source")
    raw = yaml.safe_load(path.read_text())
    for agent in raw["agents"]:
        agent["budget"] = {"requests": 2, "completion_tokens": 256}
    path.write_text(yaml.safe_dump(raw))
    behavior = bench.define_behavior(
        {"name": "recorded-model-call", "stages": {"onset": {"event": {"kind": "model_call"}}}}
    )
    baseline = bench.configure_experiment(str(path), behavior["behavior_id"], [])
    bound = bench.configure_experiment(str(path), behavior["behavior_id"], [], deployment_id=deployments["base"])
    return bench, baseline["experiment_id"], bound["experiment_id"], deployments


def fetched(bench, experiment_id, tmp_path, mutate=None, *, owned_job_id=None, destination=None, save_job=True):
    original = tmp_path / "worker-job"
    original.mkdir(parents=True)
    package = unpack_submission(bench._archive(experiment_id), original / "input")
    source = original / "input" / package["entrypoint"]
    raw = load_manifest(source).model_dump(mode="json")
    absolutize_manifest(raw, source.parent)
    raw["name"] = "worker-fixture"
    raw["gateway"].update(
        run_label="worker-fixture", api_key="worker-key", upstream_key=None, url="http://172.30.0.2:8080"
    )
    world = WorldManifest.model_validate(raw)
    bundle = RunBundle.create(original / "runs", "run_contract")
    bundle.write_manifest(raw, world.digest())
    remote_id = owned_job_id[-32:] if owned_job_id else "d" * 32
    (bundle.root / "worker-binding.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": remote_id,
                "package_digest": package["digest"],
                "entrypoint": package["entrypoint"],
                "input_base": str(source.parent),
                "runtime_manifest_digest": world.digest(),
                "inputs_verified_after_run": True,
                "input_snapshot": input_snapshot(source),
            }
        )
    )
    candidate = bench.store.get(experiment_id, "experiment")
    bindings = candidate.get("agent_deployments") or {agent.id: candidate["deployment_id"] for agent in world.agents}
    native = []
    for i, agent in enumerate(world.agents):
        dep_id = bindings[agent.id]
        dep = bench.store.get(dep_id, "deployment")
        identity = serving_identity(dep_id, dep)
        model = world.model_for(agent)
        request = {
            "model": model.id,
            "messages": [{"role": "user", "content": "original instruction"}],
            "stream": False,
        }
        response = {
            "id": f"reply-{i}",
            "model": model.id,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "recorded reply"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
        capture = BodyCapture(json.dumps(request).encode())
        capture.response(content_type="application/json", content_encoding=None, encoded=False)
        capture.receive(json.dumps(response).encode())
        capture.finish("eof")
        base = {
            "run": "worker-fixture",
            "agent": agent.id,
            "identified_by": "key",
            "request_id": f"r-{i}",
            "expected_model": model.id,
            "serving_identity": identity,
            "upstream": dep["server"].rstrip("/"),
            "model_kind": "vllm",
        }
        native.extend(
            [
                {**base, "event": "request_admitted"},
                {
                    **base,
                    "event": "chat_completion",
                    "status": 200,
                    "request": request,
                    "request_summary": request_summary(request),
                    "response": summarize_response(response),
                    "stream": False,
                    "model_identity_check": "matched",
                    "raw_capture": capture.record(),
                },
            ]
        )
        bundle.record(
            "gateway.chat_completion",
            agent_id=agent.id,
            request_id=f"r-{i}",
            status=200,
            usage=response["usage"],
            ts=None,
            request_summary=request_summary(request),
        )
        for task in world.tasks:
            messages = [
                {
                    "info": {"role": "user", "id": f"u-{i}", "sessionID": f"session-{i}"},
                    "parts": [{"type": "text", "text": "original instruction"}],
                },
                {
                    "info": {
                        "role": "assistant",
                        "id": f"m-{i}",
                        "sessionID": f"session-{i}",
                        "parentID": f"u-{i}",
                        "finish": "stop",
                        "time": {"completed": 1},
                    },
                    "parts": [{"type": "text", "text": "recorded reply"}],
                },
            ]
            transcript = bundle.root / "native" / agent.id / task.name / "messages.json"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(json.dumps(messages))
            bundle.record(
                "harness.transcript",
                agent_id=agent.id,
                task=task.name,
                session_id=f"session-{i}",
                path=transcript.relative_to(bundle.root).as_posix(),
                sha256=file_hash(transcript),
            )
            turn = collect_turn(messages, user_message_id=f"u-{i}", reply=messages[-1])
            _record_turn(bundle, agent.id, task.name, 1, turn)
            logs = bundle.root / "verifier" / agent.id / task.name
            (logs / "verifier").mkdir(parents=True)
            (logs / "verifier/reward.txt").write_text("0")
            (logs / "environment.json").write_text('{"coverage":"software_fixture"}')
            (logs / "verifier.log").write_text("software contract fixture only")
            bundle.record(
                "verifier.reward",
                agent_id=agent.id,
                task=task.name,
                reward=0,
                source="reward.txt",
                labels={},
                checkpoints={},
            )
    bundle.record("runtime.cleanup", status="complete")
    bundle.write_summary(
        {
            "status": "ok",
            "run_label": "worker-fixture",
            "model_calls": 3,
            "rewards": {f"{agent.id}/{task.name}": 0 for agent in world.agents for task in world.tasks},
            **outcome_records(bundle.events()),
        }
    )
    gateway = bundle.root / "native/gateway/records.jsonl"
    gateway.parent.mkdir()
    gateway.write_text("".join(json.dumps(row) + "\n" for row in native))
    if mutate:
        mutate(bundle, native)
        gateway.write_text("".join(json.dumps(row) + "\n" for row in native))
    (original / "job.json").write_text(
        json.dumps(
            {"job_id": remote_id, "digest": package["digest"], "status": "completed", "cleanup": {"status": "complete"}}
        )
    )
    (original / "result.json").write_text(json.dumps({"status": "ok", "run_status": "ok", "run_id": bundle.run_id}))
    job_id = owned_job_id or "job-" + "f" * 64
    result_path = destination or bench.store.directory("jobs", job_id) / "results"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "result.zip"
    result_package = export_results(original, archive)
    extract_results(archive, result_path)
    if save_job:
        bench.store.save_job(
            job_id,
            {
                "kind": "world",
                "status": "completed",
                "experiment_id": experiment_id,
                "worker": "http://127.0.0.1:8088",
                "token_env": "WORKER_KEY",
                "remote_job_id": remote_id,
                "serving_preflight": {identifier: {"deployment_id": identifier} for identifier in set(bindings.values())},
                "results_path": str(result_path),
                "retrieval": {"digest": result_package["digest"]},
            },
            new=owned_job_id is None,
        )
    return job_id, "run_contract"


def test_collective_evaluator_qualifies_original_native_binding(loop, tmp_path):
    bench, _, _, deployments = loop
    defined = bench.define_behavior({"kind": "collective", "test_file_patterns": ["tests/*.py"],
                                     "test_function_names": ["run_tests"], "impossible_task": False,
                                     "exposure_kinds": ["observation", "tool_result"]})
    candidate = bench.configure_experiment(str(world_file(tmp_path / "collective-world")),
                    defined["behavior_id"], [], deployment_id=deployments["base"])
    job, run = fetched(bench, candidate["experiment_id"], tmp_path)
    result = bench.evaluate_collective(defined["behavior_id"], job_id=job, run_id=run)
    assert result["evaluation"]["association"] == "verified_runtime_lineage"
    assert result["evaluation"]["assignment"]["binding"]["model_calls"] == 3
    assert result["evaluation"]["prevalence"]["status"] == "unknown"
    different = bench.define_behavior({"kind": "collective", "test_file_patterns": ["other.py"],
                                       "exposure_kinds": ["observation"]})
    with pytest.raises(ValueError, match="candidate's frozen behavior"):
        bench.evaluate_collective(different["behavior_id"], job_id=job, run_id=run)

def test_collective_qualified_test_results_keep_operator_gate(loop, tmp_path):
    bench, _, _, deployments = loop
    defined = bench.define_behavior({"kind": "collective", "test_file_patterns": ["tests/*.py"],
                                     "exposure_kinds": ["observation", "tool_result"]})
    operator = ResearchWorkbench(bench.store.root, allow_test_evaluation=True,
                                  allowed_workers=bench.allowed_workers, allowed_servers=bench.allowed_servers)
    held = operator.configure_experiment(str(world_file(tmp_path / "held-collective")),
                defined["behavior_id"], [], deployment_id=deployments["base"], split="test")
    held_job, held_run = fetched(operator, held["experiment_id"], tmp_path / "held-result")
    measured = operator.evaluate_collective(defined["behavior_id"], job_id=held_job, run_id=held_run)
    assert measured["evaluation"]["operator_only"] is True
    with pytest.raises(PermissionError, match="held-out"):
        bench.read_record(measured["evaluation_id"])


def test_run_lineage_freezes_full_sources_and_keeps_zero(loop, tmp_path):
    bench, _, bound, _ = loop
    job, run = fetched(bench, bound, tmp_path)
    result = bench.evaluate_run(job, run)
    value = result["evaluation"]
    assert value["status"] == "present"
    assert value["assignment"]["model_calls"] == 3
    assert value["assignment"]["numerical_weight_verification"] == "unverified"
    assert (
        bench.compare_candidates([result["evaluation_id"]])["comparison"]["prospective_runtime_binding"] == "verified"
    )
    exported = bench.export_reproducer(bound, [result["evaluation_id"]])
    assert (
        pathlib.Path(exported["path"]) / "runtime-results" / job / "runs" / run / "native/gateway/records.jsonl"
    ).is_file()
    retained = pathlib.Path(bench.store.job(job)["results_path"]) / "runs" / run / "manifest.json"
    retained.write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        bench.compare_candidates([result["evaluation_id"]])


def mixed_candidate(loop, *, bindings=None, changes=None):
    bench, baseline, _, deployments = loop
    experiment = bench.store.get(baseline, "experiment")
    mapping = bindings if bindings is not None else {
        "agent-0": deployments["trained"], "agent-1": deployments["base"], "agent-2": deployments["base"]}
    return bench.configure_experiment(str(pathlib.Path(experiment["source_root"]) / "world.yaml"),
        experiment["behavior_id"], changes or [], agent_deployments=mapping)


def test_mixed_agent_binding_packages_aliases_lineage_and_all_deployments(loop, tmp_path):
    bench, baseline, _, deployments = loop
    source = pathlib.Path(bench.store.get(baseline, "experiment")["source_root"]) / "world.yaml"
    original = yaml.safe_load(source.read_text())
    original["models"] = [{"id": "second-original-policy"}]
    original["agents"][1]["model"] = "second-original-policy"
    source.write_text(yaml.safe_dump(original))
    candidate = mixed_candidate(loop)
    identifier = candidate["experiment_id"]
    package = unpack_submission(bench._archive(identifier), tmp_path / "mixed-input")
    world = load_manifest(tmp_path / "mixed-input" / package["entrypoint"])
    assert {agent.id: world.model_for(agent).id for agent in world.agents} == {
        "agent-0": "trained", "agent-1": "base", "agent-2": "base"}
    assert {model.id for model in world.models} == {"base", "trained"}
    for agent in world.agents:
        model = world.model_for(agent)
        assert model.serving_identity.deployment_id == deployments[model.id]
    job, run = fetched(bench, identifier, tmp_path)
    result = bench.evaluate_run(job, run)
    assert result["evaluation"]["assignment"]["agent_deployments"] == {
        "agent-0": deployments["trained"], "agent-1": deployments["base"], "agent-2": deployments["base"]}
    exported = bench.export_reproducer(identifier, [result["evaluation_id"]])
    records = json.loads((pathlib.Path(exported["path"]) / "deployments.json").read_text())
    assert set(records) == set(deployments.values())


def test_single_binding_retains_legacy_preflight_and_binding_receipt(loop, tmp_path):
    bench, _, bound, deployments = loop
    job, run = fetched(bench, bound, tmp_path)
    value = bench.store.job(job)
    value["serving_preflight"] = {"deployment_id": deployments["base"]}
    bench.store.save_job(job, value)
    result = bench.evaluate_run(job, run)
    assert result["evaluation"]["assignment"]["deployment_id"] == deployments["base"]
    assert "agent_deployments" not in result["evaluation"]["assignment"]


def test_deployment_binding_arguments_are_mutually_exclusive(loop):
    bench, baseline, _, deployments = loop
    experiment = bench.store.get(baseline, "experiment")
    with pytest.raises(ValueError, match="mutually exclusive"):
        bench.configure_experiment(str(pathlib.Path(experiment["source_root"]) / "world.yaml"),
            experiment["behavior_id"], [], deployment_id=deployments["base"],
            agent_deployments={f"agent-{i}": deployments["base"] for i in range(3)})


@pytest.mark.parametrize("bad_keys", [{"agent-0", "agent-1"}, {"agent-0", "agent-1", "unknown"}, set()])
def test_mixed_binding_requires_exact_agent_keys(loop, bad_keys):
    with pytest.raises(ValueError, match="every agent exactly once"):
        mixed_candidate(loop, bindings={key: loop[3]["base"] for key in bad_keys})


@pytest.mark.parametrize("field", ["server", "base_model", "base_revision", "base_sha256", "dtype"])
def test_mixed_binding_rejects_different_base_or_endpoint(loop, field):
    bench, _, _, deployments = loop
    changed = bench.store.get(deployments["trained"], "deployment")
    if field == "server":
        changed[field] = "http://127.0.0.1:9000"
    else:
        changed["policy"][field] = {"base_model": "other/base", "base_revision": "e" * 40,
            "base_sha256": "f" * 64, "dtype": "float16"}[field]
    mismatch = bench.store.put("deployment", changed)
    with pytest.raises(ValueError, match="must share"):
        mixed_candidate(loop, bindings={"agent-0": mismatch, "agent-1": deployments["base"],
                                       "agent-2": deployments["base"]})


@pytest.mark.parametrize("factor", ["model.sampling.seed", "agent.model", "models"])
def test_mixed_binding_locks_model_factors(loop, factor):
    with pytest.raises(ValueError, match="cannot override"):
        mixed_candidate(loop, changes=[{"factor": factor, "value": "trained"}])


def test_mixed_binding_rejects_another_agents_response_alias(loop, tmp_path):
    bench, _, _, _ = loop
    candidate = mixed_candidate(loop)

    def mutate(bundle, native):
        call = next(row for row in native if row["event"] == "chat_completion" and row["agent"] == "agent-2")
        call["response"]["model"] = "trained"

    job, run = fetched(bench, candidate["experiment_id"], tmp_path, mutate)
    with pytest.raises(ValueError, match="response model"):
        bench.evaluate_run(job, run)


async def test_mixed_submission_preflights_each_distinct_deployment_once(loop, monkeypatch):
    bench, _, _, deployments = loop
    candidate = mixed_candidate(loop)
    verified = []

    async def verify(identifier):
        verified.append(identifier)
        return {"deployment_id": identifier}

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, archive, job_id, **kwargs):
            return {"job_id": job_id, "digest": kwargs["digest"], "status": "queued"}

    monkeypatch.setattr(bench, "_verify_deployment", verify)
    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    result = await bench.run_experiment(candidate["experiment_id"], "mixed", "http://127.0.0.1:8088", "WORKER_KEY")
    assert sorted(verified) == sorted(deployments.values())
    assert bench.store.job(result["job_id"])["serving_preflight"] == {
        identifier: {"deployment_id": identifier} for identifier in deployments.values()}


@pytest.mark.parametrize(
    "failure",
    [
        "response_alias",
        "missing_alias",
        "missing_call",
        "duplicate_call",
        "wrong_agent",
        "wrong_deployment",
        "wire_changed",
        "incomplete",
        "manifest",
        "verifier",
        "missing_transcript",
        "worker_binding",
    ],
)
def test_wrong_or_incomplete_runtime_never_admitted(loop, tmp_path, failure):
    bench, _, bound, _ = loop

    def mutate(bundle, rows):
        record = rows[1]
        if failure == "response_alias":
            record["response"]["model"] = "wrong"
        elif failure == "missing_alias":
            record["response"]["model"] = None
        elif failure == "missing_call":
            rows.pop()
        elif failure == "duplicate_call":
            rows.append(copy.deepcopy(record))
        elif failure == "wrong_agent":
            record["agent"] = "outsider"
        elif failure == "wrong_deployment":
            record["serving_identity"] = {}
        elif failure == "wire_changed":
            record["raw_capture"]["request"]["sha256"] = "0" * 64
        elif failure == "incomplete":
            record["raw_capture"]["response"]["capture_complete"] = False
        elif failure == "manifest":
            path = bundle.root / "manifest.json"
            value = json.loads(path.read_text())
            value["manifest"]["agents"][0]["budget"]["requests"] = 999
            path.write_text(json.dumps(value))
        elif failure == "verifier":
            next(bundle.root.glob("verifier/*/*/verifier/reward.txt")).write_text("1")
        elif failure == "missing_transcript":
            next(bundle.root.glob("native/*/*/messages.json")).unlink()
        elif failure == "worker_binding":
            path = bundle.root / "worker-binding.json"
            value = json.loads(path.read_text())
            value["job_id"] = "0" * 32
            path.write_text(json.dumps(value))

    job, run = fetched(bench, bound, tmp_path, mutate)
    with pytest.raises((ValueError, OSError)):
        bench.evaluate_run(job, run)


def test_imported_run_cannot_upgrade_to_verified_assignment(loop, tmp_path):
    bench, _, bound, _ = loop
    job, run = fetched(bench, bound, tmp_path)
    imported = bench.inspect_transcript(str(pathlib.Path(bench.store.job(job)["results_path"]) / "runs" / run))
    evaluated = bench.evaluate_candidate(bound, imported["transcript_id"])
    assert "operator_supplied" in evaluated["evaluation"]["assignment"]["binding"]


def planned(loop):
    bench, baseline, _, deployments = loop
    return bench.plan_factorial(
        baseline,
        deployments["base"],
        deployments["trained"],
        [{"factor": "agent.tools", "scope": "global", "value": {"todowrite": False}}],
        "four-cells",
        repetitions=2,
        order_seed=17,
    )


def test_factorial_plan_freezes_all_cells_budgets_and_denominator(loop):
    bench, _, _, _ = loop
    result = planned(loop)
    assert planned(loop) == result
    study = result["study"]
    assert len(study["cells"]) == 4 and len(study["trials"]) == 8
    assert study["request_budget_upper_bound"] == 48
    report = bench.report_factorial(result["study_id"])
    assert report["denominator"] == 8
    assert all(cell["behavior"]["unknown"] == 2 for cell in report["cells"].values())


@pytest.mark.parametrize(
    "factor,value",
    [("agent.budget.requests", 10), ("model.sampling.seed", 8), ("tasks", []), ("observe.activations", True)],
)
def test_factorial_refuses_confounding_axes(loop, factor, value):
    bench, baseline, _, deployments = loop
    with pytest.raises(ValueError, match="preserve"):
        bench.plan_factorial(
            baseline,
            deployments["base"],
            deployments["trained"],
            [{"factor": factor, "scope": "global", "value": value}],
            "bad-axis",
        )


async def test_factorial_reconnect_never_duplicates_submission(loop, monkeypatch):
    bench, _, _, _ = loop
    result = planned(loop)
    calls = []

    async def verify(identifier):
        return {"deployment_id": identifier}

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, archive, job_id, **kwargs):
            calls.append(job_id)
            self.__class__.digest = kwargs["digest"]
            raise ConnectionError("accepted but acknowledgement lost")

        async def status(self, job_id):
            assert job_id == calls[0]
            return {"job_id": job_id, "digest": self.digest, "status": "running"}

    monkeypatch.setattr(bench, "_verify_deployment", verify)
    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    await bench.advance_factorial(result["study_id"], "http://127.0.0.1:8088", "WORKER_KEY")
    await bench.advance_factorial(result["study_id"], "http://127.0.0.1:8088", "WORKER_KEY")
    assert len(calls) == 1
    assert bench.report_factorial(result["study_id"])["trials"][0]["status"] == "running"


def rewrite_events(bundle, change):
    path = bundle.root / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    change(events)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.mark.parametrize(
    "failure",
    [
        "normalized_status",
        "normalized_usage",
        "normalized_request_summary",
        "admission_agent",
        "admission_order",
        "verifier_labels",
        "summary",
        "run_failed",
        "cleanup_failed",
        "empty_turn",
        "missing_reply",
        "session_history",
        "unfinished_tool",
    ],
)
def test_contradictory_original_evidence_rejected(loop, tmp_path, failure):
    bench, _, bound, _ = loop

    def mutate(bundle, rows):
        if failure == "admission_agent":
            rows[0]["agent"], rows[2]["agent"] = rows[2]["agent"], rows[0]["agent"]
        elif failure == "admission_order":
            rows[0], rows[1] = rows[1], rows[0]
        elif failure == "summary":
            path = bundle.root / "summary.json"
            value = json.loads(path.read_text())
            value["rewards"] = {}
            path.write_text(json.dumps(value))
        elif failure == "run_failed":
            bundle.record("run.failed")
        elif failure == "cleanup_failed":
            bundle.record("runtime.cleanup", status="failed")
        else:

            def change(events):
                if failure.startswith("normalized_"):
                    event = next(event for event in events if event["kind"] == "gateway.chat_completion")
                    event["data"][failure.removeprefix("normalized_")] = "contradiction"
                elif failure == "verifier_labels":
                    event = next(event for event in events if event["kind"] == "verifier.reward")
                    event["data"]["labels"] = {"invented": True}
                elif failure == "session_history":
                    event = next(event for event in events if event["kind"] == "harness.transcript")
                    path = bundle.root / event["data"]["path"]
                    value = json.loads(path.read_text())
                    value[-1]["parts"][0]["text"] = "a different original reply"
                    path.write_text(json.dumps(value))
                    event["data"]["sha256"] = file_hash(path)
                else:
                    event = next(event for event in events if event["kind"] == "harness.turn")
                    path = bundle.root / event["data"]["path"]
                    value = json.loads(path.read_text())
                    if failure == "empty_turn":
                        value["messages"] = []
                    elif failure == "missing_reply":
                        value["reply"] = None
                    else:
                        # Change both original session and turn consistently: native completion
                        # validation, not a hash mismatch, must catch the outstanding tool.
                        message = value["messages"][-1]
                        message["parts"].append({"type": "tool", "state": {"status": "running"}})
                        value["reply"] = message
                        original = next(
                            e
                            for e in events
                            if e["kind"] == "harness.transcript" and e["agent_id"] == event["agent_id"]
                        )
                        original_path = bundle.root / original["data"]["path"]
                        original_path.write_text(json.dumps(value["messages"]))
                        original["data"]["sha256"] = file_hash(original_path)
                    path.write_text(json.dumps(value))
                    event["data"]["sha256"] = file_hash(path)

            rewrite_events(bundle, change)

    job, run = fetched(bench, bound, tmp_path, mutate)
    with pytest.raises(ValueError):
        bench.evaluate_run(job, run)


@pytest.mark.parametrize(
    "changes",
    [
        [{"factor": "memory", "scope": "global", "value": "reset"}],
        [
            {"factor": "memory", "scope": "global", "value": "carry"},
            {"factor": "memory", "scope": "global", "value": "reset"},
        ],
    ],
)
def test_factorial_rejects_no_effective_environment_change(loop, changes):
    bench, baseline, _, deployments = loop
    with pytest.raises(ValueError, match="no effective difference"):
        bench.plan_factorial(baseline, deployments["base"], deployments["trained"], changes, "noop")


async def test_interrupted_preflight_is_recoverable_only_at_same_id(loop, monkeypatch):
    import asyncio

    bench, _, bound, _ = loop

    async def interrupted(_):
        raise asyncio.CancelledError()

    monkeypatch.setattr(bench, "_verify_deployment", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await bench.run_experiment(bound, "crashed", "http://127.0.0.1:8088", "WORKER_KEY")
    receipt = await bench.run_experiment(bound, "crashed", "http://127.0.0.1:8088", "WORKER_KEY")
    job = await bench.get_job_status(receipt["job_id"])
    assert job["status"] == "submitting" and not job.get("submission_started")
    calls = []

    async def verified(identifier):
        return {"deployment_id": identifier}

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, archive, job_id, **kwargs):
            calls.append(job_id)
            return {"job_id": job_id, "digest": kwargs["digest"], "status": "queued"}

    monkeypatch.setattr(bench, "_verify_deployment", verified)
    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    resumed = await bench.get_job_status(receipt["job_id"], "resubmit")
    assert resumed["status"] == "queued" and calls == [job["remote_job_id"]]
    with pytest.raises(ValueError, match="uncertain"):
        await bench.get_job_status(receipt["job_id"], "resubmit")


@pytest.mark.parametrize("terminal", ["completed", "failed", "failed_cleanup", "quarantined"])
async def test_factorial_full_schedule_and_failure_accounting(loop, tmp_path, monkeypatch, terminal):
    bench, baseline, _, deployments = loop
    planned = bench.plan_factorial(
        baseline,
        deployments["base"],
        deployments["trained"],
        [{"factor": "agent.tools", "scope": "global", "value": {"todowrite": False}}],
        "schedule",
        repetitions=1,
    )
    submitted, fetched_jobs = {}, []

    async def verify(identifier):
        return {"deployment_id": identifier}

    class Client:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def submit(self, archive, job_id, **kwargs):
            assert job_id not in submitted
            submitted[job_id] = {
                "job_id": job_id,
                "digest": kwargs["digest"],
                "status": terminal if terminal in {"failed", "failed_cleanup"} else "completed",
            }
            return submitted[job_id]

        async def status(self, job_id):
            return submitted[job_id]

        async def fetch(self, job_id, destination):
            fetched_jobs.append(job_id)
            with bench.store.connection() as db:
                jobs = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
            owned = next(j for j in jobs if j.get("remote_job_id") == job_id)
            if terminal in {"completed", "quarantined"}:

                def mutation(bundle, rows):
                    if terminal == "quarantined":
                        rows[1]["response"]["model"] = "wrong"

                fetched(bench, owned["experiment_id"], tmp_path / job_id, mutation,
                        owned_job_id=owned["job_id"], destination=destination, save_job=False)
                receipt = json.loads((destination / "package.json").read_text())
                return {"job_id": job_id, "digest": receipt["digest"], "destination": str(destination)}
            source = tmp_path / job_id
            source.mkdir()
            (source / "job.json").write_text(json.dumps(submitted[job_id]))
            (source / "result.json").write_text(json.dumps({"status": terminal}))
            archive = source.with_suffix(".zip")
            package = export_results(source, archive)
            extract_results(archive, destination)
            return {"job_id": job_id, "digest": package["digest"], "destination": str(destination)}

    monkeypatch.setattr(bench, "_verify_deployment", verify)
    monkeypatch.setattr("mesocosm.reproduction.service.ExecutionClient", Client)
    for _ in range(6):
        report = await bench.advance_factorial(planned["study_id"], "http://127.0.0.1:8088", "WORKER_KEY")
        if report["status"] in {"completed", "halted"}:
            break
    assert report["denominator"] == 4
    assert len(fetched_jobs) == len(submitted)
    if terminal in {"failed_cleanup", "quarantined"}:
        assert report["status"] == "halted" and len(submitted) == 1
        assert sum(t["status"] == "not_run" for t in report["trials"]) == 3
    else:
        assert report["status"] == "completed" and len(submitted) == 4
    expected = "evaluated" if terminal == "completed" else "quarantined" if terminal == "quarantined" else "failed"
    assert report["trials"][0]["status"] == expected
    if terminal == "completed":
        assert all(
            [e["data"]["reward"] for e in t["evaluation"]["task_outcomes"]] == [0, 0, 0] for t in report["trials"]
        )
