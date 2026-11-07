"""Real tiny SFT artifact → HTTP lifecycle contract; the vLLM server is a software fixture."""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest
from test_reproduction_training import local_training_assets  # noqa: F401

from mesocosm.learning.checkpoints import checkpoint_sha256
from mesocosm.learning.serving import VLLMPolicyServer
from mesocosm.reproduction.deployment import training_artifact
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.training import run_training


@pytest.fixture
def external_assets(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"architectures":["FixtureForCausalLM"]}')
    (base / "model.safetensors").write_bytes(b"base fixture weights")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base), "r": 32}))
    (adapter / "adapter_model.safetensors").write_bytes(b"external fixture weights")
    return dict(server="http://127.0.0.1:8000", model="hacker", base_model=str(base),
        base_revision="a" * 40, base_sha256=checkpoint_sha256(base), idempotency_key="external",
        external_adapter={"repo": "publisher/adapters", "revision": "d" * 40, "subfolder": "seed-42",
                          "path": str(adapter), "sha256": checkpoint_sha256(adapter, adapter=True)})


@pytest.mark.parametrize("failure", ["hash", "base", "owned_and_external", "revision"])
async def test_external_adapter_invalid_inputs_never_reach_server(tmp_path, external_assets, monkeypatch, failure):
    bench = ResearchWorkbench(tmp_path / "research", allowed_servers={"http://127.0.0.1:8000": None})
    args = external_assets
    if failure == "hash":
        args["external_adapter"]["sha256"] = "f" * 64
    elif failure == "base":
        config = pathlib.Path(args["external_adapter"]["path"]) / "adapter_config.json"
        config.write_text('{"base_model_name_or_path":"different/base","r":32}')
        args["external_adapter"]["sha256"] = checkpoint_sha256(config.parent, adapter=True)
    elif failure == "owned_and_external":
        args["training_job_id"] = "nonexistent"
    else:
        args["external_adapter"]["revision"] = "main"
    monkeypatch.setattr(bench, "_server", lambda _: pytest.fail("invalid adapter reached serving API"))
    with pytest.raises(ValueError, match={"hash": "hash mismatch", "base": "base model mismatch",
            "owned_and_external": "mutually exclusive", "revision": "revision"}[failure]):
        await bench.deploy_policy(**args)


def external_transport(args, *, maximum=32, lose_ack=False):
    entries = [{"id": "base", "root": args["base_model"], "parent": None}]
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": entries})
        if request.url.path == "/server_info":
            lora = {"lora_dtype": "auto"}
            if maximum is not None:
                lora["max_lora_rank"] = maximum
            return httpx.Response(200, json={"vllm_config": {
                "model_config": {"model": args["base_model"], "revision": args["base_revision"],
                    "dtype": "bfloat16", "quantization": None, "hf_overrides": {}, "hf_config": {}},
                "cache_config": {"cache_dtype": "auto"}, "lora_config": lora}})
        body = json.loads(request.content)
        if request.url.path == "/v1/load_lora_adapter":
            entries.append({"id": body["lora_name"], "root": body["lora_path"], "parent": "base"})
            if lose_ack:
                raise httpx.ReadError("load succeeded but acknowledgement was lost")
            return httpx.Response(200, json={"status": "loaded"})
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, json={"id": "probe", "model": body["model"],
            "choices": [{"text": "reply", "finish_reason": "length"}], "usage": {"completion_tokens": 1}})

    return httpx.MockTransport(handle), calls


@pytest.mark.parametrize("maximum", [None, 16])
async def test_external_adapter_requires_observed_server_rank_capacity(
    tmp_path, external_assets, monkeypatch, maximum
):
    bench = ResearchWorkbench(tmp_path / "research", allowed_servers={"http://127.0.0.1:8000": None})
    transport, calls = external_transport(external_assets, maximum=maximum)
    monkeypatch.setattr(bench, "_server", lambda url: VLLMPolicyServer(url, transport=transport))
    with pytest.raises(ValueError, match="unknown|exceeds"):
        await bench.deploy_policy(**external_assets)
    assert "/v1/load_lora_adapter" not in calls


async def test_external_adapter_uncertain_load_reconciles_without_reload_and_rehashes(
    tmp_path, external_assets, monkeypatch
):
    bench = ResearchWorkbench(tmp_path / "research", allowed_servers={"http://127.0.0.1:8000": None})
    transport, calls = external_transport(external_assets, lose_ack=True)
    monkeypatch.setattr(bench, "_server", lambda url: VLLMPolicyServer(url, transport=transport))
    initial = await bench.deploy_policy(**external_assets)
    assert initial["status"] == "activation_uncertain"
    reconciled = await bench.get_job_status(initial["job_id"], "reconcile")
    assert reconciled["status"] == "ready"
    assert await bench.get_job_status(initial["job_id"], "reconcile") == reconciled
    assert calls.count("/v1/load_lora_adapter") == 1
    record = bench.store.get(reconciled["deployment_id"], "deployment")
    assert record["training_artifact"] is None
    assert record["external_adapter"] == external_assets["external_adapter"]
    assert record["training_provenance"] == "unverified"
    assert record["evidence_level"] == "external_adapter_hash_and_serving_api"
    assert record["receipt"]["external_adapter_capacity"]["max_lora_rank"] == 32
    await bench._verify_deployment(reconciled["deployment_id"])
    assert calls.count("/v1/load_lora_adapter") == 1
    pathlib.Path(record["external_adapter"]["path"], "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        await bench._verify_deployment(reconciled["deployment_id"])


async def test_real_completed_sft_connects_to_deployment_and_uncertain_load_reconciliation(
    tmp_path, local_training_assets, monkeypatch  # noqa: F811
):
    _, config, _, _ = local_training_assets
    bench = ResearchWorkbench(tmp_path / "research", allowed_servers={"http://127.0.0.1:8000": None})
    dataset = bench.prepare_dataset(
        [str(tmp_path / "source.json")],
        {"split_assignments": {"train-1": "train", "train-2": "train", "development": "development", "test": "test"}},
    )
    queued = bench.submit_training(
        dataset["dataset_id"], {**config, "dtype": "bfloat16", "max_steps": 1}, "train", start=False
    )
    job = bench.store.job(queued["job_id"])
    result = run_training(pathlib.Path(job["dataset_path"]), pathlib.Path(job["output_path"]), job["config"])
    assert result["optimizer_steps"] == 1 and result["losses"][0]["weight_delta_l2"] > 0
    job.update(status="completed", result=result)
    bench.store.save_job(job["job_id"], job)
    assert training_artifact(bench.store, job["job_id"])["adapter_sha256"] == result["adapter_sha256"]
    entries = [{"id": "base", "root": config["base_model"], "parent": None}]
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": entries})
        if request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={
                    "vllm_config": {
                        "model_config": {
                            "model": config["base_model"],
                            "revision": "a" * 40,
                            "dtype": "bfloat16",
                            "quantization": None,
                            "hf_overrides": {},
                            "hf_config": {"head_dtype": "model"},
                        },
                        "cache_config": {"cache_dtype": "auto"},
                        "lora_config": {"lora_dtype": "auto"},
                    }
                },
            )
        body = json.loads(request.content)
        if request.url.path == "/v1/load_lora_adapter":
            entries.append({"id": body["lora_name"], "root": body["lora_path"], "parent": "base"})
            raise httpx.ReadError("load succeeded but acknowledgement was lost")
        assert request.url.path == "/v1/completions"
        return httpx.Response(
            200,
            json={
                "id": "probe",
                "model": body["model"],
                "choices": [{"text": "reply", "finish_reason": "length"}],
                "usage": {"completion_tokens": 1},
            },
        )

    monkeypatch.setattr(
        "mesocosm.learning.serving.VLLMPolicyServer",
        lambda url, key: VLLMPolicyServer(url, key, transport=httpx.MockTransport(handle)),
    )
    args = {
        "server": "http://127.0.0.1:8000",
        "model": "base",
        "base_model": config["base_model"],
        "base_revision": "a" * 40,
        "base_sha256": config["base_sha256"],
        "idempotency_key": "base",
    }
    base = await bench.deploy_policy(**args)
    assert base["status"] == "ready"
    args.update(model="trained", idempotency_key="trained", training_job_id=job["job_id"])
    trained = await bench.deploy_policy(**args)
    assert trained["status"] == "activation_uncertain"
    assert await bench.deploy_policy(**args) == trained
    reconciled = await bench.get_job_status(trained["job_id"], "reconcile")
    assert reconciled["status"] == "ready"
    assert await bench.get_job_status(trained["job_id"], "reconcile") == reconciled
    assert calls.count("/v1/load_lora_adapter") == 1
    assert reconciled["numerical_weight_verification"] == "unverified"
    await bench._verify_deployment(reconciled["deployment_id"])
    pathlib.Path(result["adapter_path"], "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        await bench._verify_deployment(reconciled["deployment_id"])


async def test_unapproved_server_and_unsupported_dtype_never_load(tmp_path):
    bench = ResearchWorkbench(tmp_path / "research", allowed_servers={"http://127.0.0.1:8000": None})
    args = {
        "server": "http://127.0.0.1:8000",
        "model": "base",
        "base_model": "missing",
        "base_revision": "a" * 40,
        "base_sha256": "b" * 64,
        "idempotency_key": "reject",
    }
    with pytest.raises(ValueError, match="operator-approved"):
        await bench.deploy_policy(**(args | {"server": "https://unapproved.invalid"}))
    with pytest.raises(ValueError, match="dtype conversion"):
        await bench.deploy_policy(**(args | {"dtype": "float32"}))


@pytest.mark.parametrize("mutation", ["choices", "error", "id", "tokens", "finish"])
def test_workbench_requires_complete_serving_probe(mutation):
    from mesocosm.learning.config import PolicyIdentity, policy_id
    from mesocosm.reproduction.deployment import verify_receipt

    policy = PolicyIdentity(
        id=policy_id("base", "a" * 40, "b" * 64, None),
        model="base",
        base_model="base",
        base_revision="a" * 40,
        base_sha256="b" * 64,
    )
    probe = {
        "id": "probe-1",
        "model": "base",
        "choices": [{"text": "", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
    }
    receipt = {
        "policy_id": policy.id,
        "model": "base",
        "probe_response": probe,
        "registration": {"root": "base", "parent": None},
        "configuration": {"model": "base"},
    }
    verify_receipt(policy, receipt)
    if mutation == "choices":
        probe["choices"] = [{}]
    elif mutation == "error":
        probe["error"] = {"message": "failed"}
    elif mutation == "id":
        probe["id"] = None
    elif mutation == "tokens":
        probe["usage"] = {"completion_tokens": 0}
    else:
        probe["choices"][0]["finish_reason"] = None
    with pytest.raises(ValueError, match="probe"):
        verify_receipt(policy, receipt)


async def test_concurrent_alias_activation_and_reconciliation_are_leased(tmp_path, monkeypatch):
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    bench = ResearchWorkbench(tmp_path / "research", allowed_servers={"http://127.0.0.1:8000": None})
    monkeypatch.setattr("mesocosm.reproduction.deployment.verify_artifacts", lambda _: None)

    class Server:
        async def verify(self, policy):
            entered.set()
            await release.wait()
            return {
                "policy_id": policy.id,
                "model": policy.model,
                "registration": {"root": "base", "parent": None},
                "configuration": {"model": "base"},
                "probe_response": {
                    "id": "probe",
                    "model": policy.model,
                    "choices": [{"text": "", "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                },
            }

        async def aclose(self):
            pass

    monkeypatch.setattr(bench, "_server", lambda _: Server())
    args = dict(
        server="http://127.0.0.1:8000",
        model="base",
        base_model="base",
        base_revision="a" * 40,
        base_sha256="b" * 64,
        idempotency_key="initial",
    )
    pending = asyncio.create_task(bench.deploy_policy(**args))
    await entered.wait()
    try:
        with pytest.raises(ValueError, match="another caller"):
            await bench.deploy_policy(**(args | {"idempotency_key": "other"}))
        with bench.store.connection() as db:
            job_id = db.execute("SELECT id FROM jobs").fetchone()[0]
        with pytest.raises(ValueError, match="another caller"):
            await bench.get_job_status(job_id, "reconcile")
    finally:
        release.set()
    result = await pending
    assert await bench.get_job_status(job_id, "reconcile") == result
    assert await bench.deploy_policy(**args) == result
