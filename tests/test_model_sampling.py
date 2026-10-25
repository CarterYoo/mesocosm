"""CPU wire-contract tests; fixtures do not establish live vLLM acceptance or replay."""

from __future__ import annotations

import base64
import json
import pathlib

import httpx
import pytest
from pydantic import ValidationError

from mesocosm.core.manifest import ModelSpec, WorldManifest
from mesocosm.core.sampling import SamplingSpec
from mesocosm.env.compose import gateway_upstreams, render_compose
from mesocosm.gateway.app import Settings, create_app
from mesocosm.gateway.client import GatewayAdmin
from mesocosm.gateway.rollout import SAMPLING
from mesocosm.learning.config import PolicyIdentity, policy_id

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMETERS = {"temperature": 0.6, "top_p": 0.95, "seed": 42, "top_k": -1, "min_p": 0.0}


def world(**overrides):
    return WorldManifest.model_validate({
        "name": "sampling-contract", "model": {"id": "sampled", "sampling": PARAMETERS},
        "gateway": {"url": "http://localhost:8080"},
        "agents": [{"id": "agent-1"}, {"id": "agent-2"}, {"id": "agent-3"}], **overrides,
    })


def records(path):
    return [json.loads(line) for line in (path / "gateway.jsonl").read_text().splitlines()]


def gateway(tmp_path, upstreams, **settings):
    sent = []

    def upstream(request):
        sent.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=(
                b'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n'),
                headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "ok"},
                                                     "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    app = create_app(Settings(upstream="http://default", upstream_key="default-secret", log_dir=tmp_path,
        upstreams=upstreams, agent_keys={"key": {"agent": "agent-1", "run": "world"}}, **settings),
        transport=httpx.MockTransport(upstream))
    return app, sent


async def request(app, body, *, path="/v1/chat/completions"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client:
        return await client.post(path, headers={"Authorization": "Bearer key"}, json=body)


async def close(app):
    for client in app.state.gateway.clients.values():
        await client.aclose()


def test_explicit_contract_defaults_ranges_and_absence():
    assert ModelSpec(id="plain").sampling is None
    spec = SamplingSpec(temperature=0, top_p=1, seed=0)
    assert spec.model_dump() == {"temperature": 0.0, "top_p": 1.0, "seed": 0, "top_k": -1, "min_p": 0.0}
    assert SamplingSpec(temperature=2, top_p=1, seed=2**63 - 1, top_k=32, min_p=1).top_k == 32
    assert gateway_upstreams(world(model={"id": "plain"})) == {}


@pytest.mark.parametrize("key,value", [
    ("temperature", -0.1), ("temperature", 2.1), ("temperature", True), ("temperature", "0.6"),
    ("temperature", None), ("temperature", float("nan")), ("temperature", float("inf")),
    ("top_p", 0), ("top_p", 1.01), ("top_p", False), ("top_p", None), ("top_p", float("nan")),
    ("seed", -1), ("seed", 2**63), ("seed", 1.0), ("seed", True), ("seed", None),
    ("top_k", -2), ("top_k", 0), ("top_k", 2.0), ("top_k", False), ("top_k", None),
    ("min_p", -0.1), ("min_p", 1.01), ("min_p", True), ("min_p", None), ("min_p", float("inf")),
    ("unknown", 1),
])
def test_manifest_and_direct_gateway_reject_invalid_contract(key, value):
    invalid = PARAMETERS | {key: value}
    with pytest.raises(ValidationError):
        ModelSpec(id="m", sampling=invalid)
    with pytest.raises(ValidationError):
        Settings(upstreams={"m": {"sampling": invalid}})


@pytest.mark.parametrize("key", ["temperature", "top_p", "seed"])
def test_required_fields_cannot_fall_back_to_harness_or_server(key):
    with pytest.raises(ValidationError):
        SamplingSpec.model_validate({k: v for k, v in PARAMETERS.items() if k != key})


@pytest.mark.parametrize("duplicate", [{"id": "sampled"}, {"id": "sampled", "sampling": PARAMETERS | {"seed": 43}}])
def test_duplicate_catalog_ids_cannot_erase_or_replace_a_sampling_contract(duplicate):
    with pytest.raises(ValidationError, match="unique model ids"):
        world(models=[duplicate])


def test_non_vllm_contract_and_training_compose_fail_before_execution():
    with pytest.raises(ValidationError, match="vllm route"):
        ModelSpec(id="m", kind="openai", sampling=PARAMETERS)
    with pytest.raises(ValueError, match="vllm route"):
        Settings(upstreams={"m": {"kind": "openai", "sampling": PARAMETERS}})
    with pytest.raises(ValueError, match="rollout-bound training"):
        Settings(upstreams={"m": {"sampling": PARAMETERS}}, require_rollout_binding=True)
    with pytest.raises(ValueError, match="rollout-bound training"):
        render_compose(world(), repo_root=ROOT, run_label="train", require_rollout_binding=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("supplied", [{}, PARAMETERS])
async def test_compose_default_route_reaches_actual_wire_and_exact_evidence(tmp_path, monkeypatch, stream, supplied):
    config = render_compose(world(), repo_root=ROOT, run_label="sampling")["services"]["gateway"]["environment"]
    monkeypatch.setenv("MESOCOSM_UPSTREAMS", config["MESOCOSM_UPSTREAMS"])
    upstreams = Settings.from_env().upstreams
    assert upstreams["sampled"]["url"] is None
    app, sent = gateway(tmp_path, upstreams, token_logprobs=True, top_logprobs=20)
    body = {"model": "sampled", "messages": [], "stream": stream, **supplied}
    result = await request(app, body)
    assert result.status_code == 200
    actual = json.loads(sent[0].content)
    assert actual == {**body, **PARAMETERS, "logprobs": True, "top_logprobs": 20, "return_token_ids": True,
                      **({"stream_options": {"include_usage": True}} if stream else {})}
    assert sent[0].url.host == "default" and sent[0].headers["authorization"] == "Bearer default-secret"
    completed = next(r for r in records(tmp_path) if r["event"] == "chat_completion")
    assert completed["request"] == actual
    assert base64.b64decode(completed["raw_capture"]["request"]["body_base64"]) == sent[0].content
    assert completed["sampling_contract"] == {"source": "model.sampling", "model": "sampled",
        "parameters": PARAMETERS, "seed_scope": "fixed_per_request"}
    assert next(r for r in records(tmp_path) if r["event"] == "request_admitted")["sampling_contract"] == completed["sampling_contract"]
    await close(app)


async def test_multiple_routes_keep_distinct_contracts_keys_and_unknown_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OTHER_KEY", "other-secret")
    other = PARAMETERS | {"temperature": 0.0, "top_p": 1.0, "seed": 7}
    manifest = world(models=[{"id": "other", "upstream": "http://other", "api_key_env": "OTHER_KEY", "sampling": other},
                             {"id": "api", "kind": "openai", "upstream": "http://api"}])
    app, sent = gateway(tmp_path, gateway_upstreams(manifest))
    for model in ("sampled", "other", "unknown", "api", "sampled"):
        assert (await request(app, {"model": model, "messages": []})).status_code == 200
    assert [r.url.host for r in sent] == ["default", "other", "default", "api", "default"]
    assert sent[1].headers["authorization"] == "Bearer other-secret"
    assert "authorization" not in sent[3].headers
    assert json.loads(sent[1].content)["seed"] == 7
    assert json.loads(sent[0].content)["seed"] == json.loads(sent[4].content)["seed"] == 42
    for index in (2, 3):
        assert set(json.loads(sent[index].content)) == {"model", "messages"}
    completed = [r for r in records(tmp_path) if r["event"] == "chat_completion"]
    assert "sampling_contract" not in completed[2] and "sampling_contract" not in completed[3]
    await close(app)


@pytest.mark.parametrize("extra", [
    {"temperature": 1}, {"temperature": True}, {"temperature": None}, {"top_p": 1}, {"top_p": "0.95"},
    {"seed": 41}, {"seed": None}, {"top_k": 0}, {"top_k": 50}, {"min_p": 0.01}, {"min_p": False},
    {"n": 2}, {"n": True}, {"stream": "true"}, {"best_of": 2}, {"use_beam_search": True},
    {"extra_body": {"temperature": 0.6}}, {"logit_bias": {}}, {"logits_processors": []},
    {"structured_outputs": {}}, {"response_format": {}}, {"future_sampler": 1},
])
async def test_conflicts_and_unsupported_modes_never_dispatch_or_consume_budget(tmp_path, extra):
    app, sent = gateway(tmp_path, {"sampled": {"sampling": PARAMETERS}})
    body = {"model": "sampled", "messages": [], **extra}
    response = await request(app, body)
    assert response.status_code == 400 and not sent
    assert app.state.gateway.usage_for("agent-1")["requests"] == 0
    log = records(tmp_path)
    assert log[-1]["event"] == "invalid_request" and log[-1]["request"] == body
    assert not any(r["event"] == "request_admitted" for r in log)
    await close(app)


async def test_unsupported_endpoint_and_absent_contract_do_not_gain_sampling(tmp_path):
    app, sent = gateway(tmp_path, {"sampled": {"sampling": PARAMETERS}})
    for path in ("/v1/responses", "/responses", "/v1/completions"):
        assert (await request(app, {"model": "sampled"}, path=path)).status_code == 404
    assert not sent
    body = {"model": "legacy", "messages": [], "temperature": 0.2, "future_sampler": {"enabled": True}}
    assert (await request(app, body)).status_code == 200
    assert json.loads(sent[0].content) == body
    await close(app)


@pytest.mark.parametrize("message", [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example/image"}}]},
    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}]},
    {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://example/video"}}]},
    {"role": "user", "content": [{"type": "text", "text": "x", "image_url": "hidden"}]},
    {"role": "user", "content": {"type": "text", "text": "x"}},
    {"role": "assistant", "content": None, "audio": {"id": "audio-response"}},
    {"role": "user", "content": "x", "prompt_embeds": [[1]]},
])
async def test_opt_in_multimodal_or_unknown_message_extensions_fail_closed(tmp_path, message):
    app, sent = gateway(tmp_path, {"sampled": {"sampling": PARAMETERS}})
    response = await request(app, {"model": "sampled", "messages": [message]})
    assert response.status_code == 400 and not sent
    assert app.state.gateway.usage_for("agent-1")["requests"] == 0
    await close(app)


async def test_text_parts_and_tool_history_are_preserved(tmp_path):
    app, sent = gateway(tmp_path, {"sampled": {"sampling": PARAMETERS}})
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Use a tool."}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "/candidate"},
    ]
    response = await request(app, {"model": "sampled", "messages": messages, "n": 1})
    assert response.status_code == 200 and json.loads(sent[0].content)["messages"] == messages
    await close(app)


async def test_rollout_binding_rejects_sampling_and_request_defense_preserves_training(tmp_path):
    identity = {"base_model": "test/model", "base_revision": "a" * 40, "base_sha256": "b" * 64,
                "adapter_sha256": None}
    policy = PolicyIdentity(id=policy_id(**identity), model="sampled", **identity)
    app, sent = gateway(tmp_path, {"sampled": {"sampling": PARAMETERS}})
    admin = GatewayAdmin("http://gateway", transport=httpx.ASGITransport(app))
    with pytest.raises(httpx.HTTPStatusError) as error:
        await admin.bind_rollout("agent-1", "world", "episode", "task", 1, "session", policy)
    assert error.value.response.status_code == 422 and not app.state.gateway.rollout_bindings
    # Defense against in-memory configuration changes after a valid binding.
    app.state.gateway.settings.upstreams = {}
    await admin.bind_rollout("agent-1", "world", "episode", "task", 1, "session", policy)
    app.state.gateway.settings.upstreams = {"sampled": {"sampling": PARAMETERS}}
    assert (await request(app, {"model": "sampled", "messages": []})).status_code == 400 and not sent
    app.state.gateway.settings.upstreams = {}
    assert (await request(app, {"model": "sampled", "messages": [], "temperature": 0.2})).status_code == 200
    actual = json.loads(sent[0].content)
    assert all(actual[key] == value for key, value in SAMPLING.items())
    assert "sampling_contract" not in records(tmp_path)[-1]
    await admin.aclose()
    await close(app)
