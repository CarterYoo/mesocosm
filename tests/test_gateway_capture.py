"""CPU transport fixtures test evidence retention, not real-model acceptance."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import zlib

import httpx
import pytest

from mesocosm.gateway.app import Settings, create_app
from mesocosm.gateway.capture import BodyCapture, CaptureLimitError
from mesocosm.gateway.client import GatewayAdmin
from mesocosm.gateway.rollout import SAMPLING
from mesocosm.learning.config import PolicyIdentity, policy_id


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, *, fail=False):
        self.chunks, self.fail = chunks, fail

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadError("interrupted fixture")


def records(path):
    return [json.loads(line) for line in (path / "gateway.jsonl").read_text().splitlines()]


def captured_bytes(record):
    capture = record["raw_capture"]
    response = capture["response"]
    chunks = response["chunks"]
    assert [c["sequence"] for c in chunks] == list(range(len(chunks)))
    assert [c["elapsed_ns"] for c in chunks] == sorted(c["elapsed_ns"] for c in chunks)
    assert all(c["received_at"] >= capture["request"]["dispatch_at"] for c in chunks)
    assert capture["timestamp_semantics"] == "gateway_receive_time_not_generation_time"
    assert capture["chunk_semantics"] == "http_body_delivery_not_token_or_sse_event"
    content = b"".join(base64.b64decode(c["body_base64"]) for c in chunks)
    assert len(content) == response["byte_count"] == sum(c["byte_count"] for c in chunks)
    assert hashlib.sha256(content).hexdigest() == response["sha256"]
    return content


def gateway(tmp_path, upstream, **settings):
    return create_app(Settings(upstream="http://upstream", log_dir=tmp_path, return_token_ids=True,
                               upstream_key="private-upstream-key",
                               agent_keys={"private-agent-key": {"agent": "a", "run": "r"}}, **settings),
                      transport=httpx.MockTransport(upstream))


async def request(app, *, stream=False):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client:
        return await client.post("/v1/chat/completions", headers={"Authorization": "Bearer private-agent-key"},
                                 json={"model": "open-weight", "messages": [], "stream": stream})


async def test_capture_preserves_exact_forwarded_request_and_unknown_json_fields_in_bundle_export(tmp_path):
    raw = (' {"choices":[{"message":{"content":"안녕","reasoning":"a"},"logprobs":'
           '{"content":[{"token":"안","bytes":[236,149,136],"logprob":-0.2,"top_logprobs":[]}]}}],'
           '"unknown_future_field":{"nested":[1,true]},"usage":{"prompt_tokens":1,"completion_tokens":1}}\n').encode()
    sent = []

    def upstream(req):
        sent.append(req.content)
        return httpx.Response(200, stream=Chunks([raw[:25], raw[25:]]), headers={"content-type": "application/json"})

    app = gateway(tmp_path, upstream)
    result = await request(app)
    assert result.content == raw
    record = records(tmp_path)[-1]
    assert captured_bytes(record) == raw
    assert record["response"]["reasoning"] == "a"
    capture_request = record["raw_capture"]["request"]
    assert base64.b64decode(capture_request["body_base64"]) == sent[0]
    assert hashlib.sha256(sent[0]).hexdigest() == capture_request["sha256"]
    assert json.loads(sent[0])["return_token_ids"] is True
    assert record["raw_capture"]["response"]["body_representation"] == "content_encoded_http_body"
    assert record["raw_capture"]["response"]["termination"] == "eof"
    assert "private-agent-key" not in json.dumps(record) and "private-upstream-key" not in json.dumps(record)
    admin = GatewayAdmin("http://gateway", transport=httpx.ASGITransport(app))
    exported = [item async for item in admin.records("r")]
    assert exported[-1]["raw_capture"] == record["raw_capture"]
    await admin.aclose()
    await app.state.gateway.client.aclose()


@pytest.mark.parametrize("status,raw", [
    (200, b'data: {"choices":[{"delta":{"reasoning":"a"},"unknown":7}]}\r\n\r\n'
          b'event: custom\ndata: {"future":[1,2]}\n\ndata: [DONE]\n\n'),
    (429, b'{"error":{"message":"' + b'x' * 7000 + b'"},"extra":"retained"}'),
    (200, b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'),
    (200, b'data: []\n\n\xff\x00'),
])
async def test_stream_capture_preserves_unknown_fields_invalid_bytes_and_large_errors(tmp_path, status, raw):
    chunks = [raw[:13], raw[13:44], raw[44:]]
    app = gateway(tmp_path, lambda _: httpx.Response(status, stream=Chunks(chunks),
                                                   headers={"content-type": "text/event-stream"}))
    result = await request(app, stream=True)
    assert result.status_code == status and result.content == raw
    record = records(tmp_path)[-1]
    assert captured_bytes(record) == raw
    response = record["raw_capture"]["response"]
    assert len(response["chunks"]) == len([c for c in chunks if c])
    assert response["body_representation"] == "content_encoded_http_body"
    assert response["transport_complete"] is True
    if b"partial" in raw:
        assert record["response"]["stream_complete"] is False  # EOF is not the SSE completion marker.
    if b"data: []" in raw:
        assert record["response"]["normalization_error"] == "AttributeError"
    await app.state.gateway.client.aclose()


@pytest.mark.parametrize("stream", [False, True])
async def test_content_encoding_has_explicit_capture_layer(tmp_path, stream):
    raw = b'data: [DONE]\n\n' if stream else b'{"choices":[]}'
    compressed = gzip.compress(raw)
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([compressed]),
                                                   headers={"content-encoding": "gzip"}))
    result = await request(app, stream=stream)
    assert result.content == raw
    record = records(tmp_path)[-1]
    assert captured_bytes(record) == compressed
    assert record["raw_capture"]["response"]["content_encoding"] == "gzip"
    if not stream:
        assert "content-encoding" not in result.headers
    await app.state.gateway.client.aclose()


@pytest.mark.parametrize("stream", [False, True])
async def test_prebuffered_transport_explicitly_records_decoded_bytes(tmp_path, stream):
    raw = b'data: [DONE]\n\n' if stream else b'{"choices":[]}'
    app = gateway(tmp_path, lambda _: httpx.Response(200, content=gzip.compress(raw),
                                                   headers={"content-encoding": "gzip"}))
    result = await request(app, stream=stream)
    record = records(tmp_path)[-1]
    assert result.content == raw and "content-encoding" not in result.headers
    assert captured_bytes(record) == raw
    assert record["raw_capture"]["response"]["body_representation"] == "decoded_http_body"
    await app.state.gateway.client.aclose()


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
async def test_compressed_stream_is_normalized_without_losing_usage_or_token_ids(tmp_path, encoding):
    raw = (b'data: {"choices":[{"index":0,"delta":{"reasoning":"step","content":"answer"},'
           b'"token_ids":[2],"finish_reason":"stop"}],"prompt_token_ids":[1],'
           b'"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n')
    compressed = gzip.compress(raw) if encoding == "gzip" else zlib.compress(raw)
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([compressed[:15], compressed[15:]]),
                                                   headers={"content-encoding": encoding}))
    result = await request(app, stream=True)
    record = records(tmp_path)[-1]
    assert result.content == raw and captured_bytes(record) == compressed
    assert record["response"]["reasoning"] == "step" and record["response"]["content"] == "answer"
    assert record["response"]["stream_complete"] is True and record["missing_usage"] is False
    assert record["response"]["prompt_token_ids"] == [1] and record["response"]["token_ids"] == [2]
    assert record["usage_total"] == {"requests": 1, "prompt_tokens": 1, "completion_tokens": 1}
    await app.state.gateway.client.aclose()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("usage", [{"prompt_tokens": "bad", "completion_tokens": 1},
                                   {"prompt_tokens": 3, "completion_tokens": "bad"},
                                   {"prompt_tokens": -1, "completion_tokens": 1}, "bad"])
async def test_invalid_usage_never_loses_raw_evidence_or_partially_updates_counters(tmp_path, stream, usage):
    payload = json.dumps({"choices": [], "usage": usage}).encode()
    raw = b"data: " + payload + b"\n\ndata: [DONE]\n\n" if stream else payload
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([raw])))
    result = await request(app, stream=stream)
    record = records(tmp_path)[-1]
    assert result.status_code == 200 and captured_bytes(record) == raw
    assert record["missing_usage"] is True
    assert record["usage_total"] == {"requests": 1, "prompt_tokens": 0, "completion_tokens": 0}
    await app.state.gateway.client.aclose()


def test_capture_limits_keep_prefix_and_hash_all_received_bytes_without_unbounded_chunks():
    capture = BodyCapture(b"{}", max_bytes=5, max_chunks=4)
    capture.receive(b"abc")
    with pytest.raises(CaptureLimitError, match="byte_limit"):
        capture.receive(b"defg")
    capture.finish("capture_limit")
    saved = capture.record()["response"]
    assert saved["byte_count"] == 7 and saved["captured_byte_count"] == 5
    assert b"".join(base64.b64decode(c["body_base64"]) for c in saved["chunks"]) == b"abcde"
    assert saved["sha256"] == hashlib.sha256(b"abcdefg").hexdigest()
    assert saved["captured_sha256"] == hashlib.sha256(b"abcde").hexdigest()
    assert saved["capture_complete"] is False and saved["truncation_reason"] == "byte_limit"
    chunks = BodyCapture(b"{}", max_bytes=100, max_chunks=2)
    chunks.receive(b"a")
    chunks.receive(b"b")
    with pytest.raises(CaptureLimitError, match="chunk_limit"):
        chunks.receive(b"c")
    saved = chunks.record()["response"]
    assert len(saved["chunks"]) == 2 and saved["received_chunk_count"] == 3
    assert saved["captured_byte_count"] == 2 and saved["byte_count"] == 3


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["byte_limit", "chunk_limit", "decoded_byte_limit"])
async def test_capture_limit_stops_upstream_records_failure_and_releases_lease(tmp_path, stream, kind):
    cap = 256
    raw = b"x" * (cap + 1)
    parts = [raw] if kind == "byte_limit" else [b"a", b"b", b"c"]
    headers = {}
    if kind == "decoded_byte_limit":
        parts, headers = [gzip.compress(raw)], {"content-encoding": "gzip"}

    class Tracked(Chunks):
        closed = False

        async def aclose(self):
            self.closed = True

    upstream_body = Tracked(parts)
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=upstream_body, headers=headers),
                  capture_max_bytes=cap, capture_max_chunks=2)
    if stream:
        with pytest.raises(CaptureLimitError, match=kind):
            await request(app, stream=True)
    else:
        result = await request(app)
        assert result.status_code == 502
    record = next(r for r in records(tmp_path) if "raw_capture" in r)
    saved = record["raw_capture"]["response"]
    assert saved["termination"] == "capture_limit" and saved["truncation_reason"] == kind
    assert saved["transport_complete"] is False and saved["capture_complete"] is False
    assert saved["captured_byte_count"] <= cap and len(saved["chunks"]) <= 2
    assert saved["sha256"] == hashlib.sha256(b"".join(parts)).hexdigest()
    assert upstream_body.closed and not app.state.gateway.request_locks["a"].locked()
    assert "a" in app.state.gateway.missing_usage
    if stream:
        assert record["response"]["stream_complete"] is False
        assert "evidence_capture_limit" in record["response"]["stream_errors"]
    await app.state.gateway.client.aclose()


async def test_oversized_request_is_rejected_before_upstream(tmp_path):
    def upstream(_):
        pytest.fail("oversized request must not reach inference")

    app = gateway(tmp_path, upstream, capture_max_bytes=16)
    response = await request(app)
    assert response.status_code == 400 and records(tmp_path)[-1]["reason"] == "request_byte_limit"
    await app.state.gateway.client.aclose()


def test_observation_and_capture_settings_from_environment(monkeypatch):
    for key, value in {"MESOCOSM_TOKEN_LOGPROBS": "1", "MESOCOSM_TOP_LOGPROBS": "7",
                       "MESOCOSM_CAPTURE_MAX_BYTES": "4096", "MESOCOSM_CAPTURE_MAX_CHUNKS": "32"}.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.token_logprobs is True and settings.top_logprobs == 7
    assert settings.capture_max_bytes == 4096 and settings.capture_max_chunks == 32
    with pytest.raises(ValueError):
        Settings(top_logprobs=1)
    with pytest.raises(ValueError):
        Settings(token_logprobs=True, top_logprobs=21)
    with pytest.raises(ValueError):
        Settings(capture_max_chunks=0)


@pytest.mark.parametrize("kind", ["vllm", "openai"])
async def test_observation_requests_logprobs_without_changing_decoding(tmp_path, kind):
    sent = []

    def upstream(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    app = gateway(tmp_path, upstream, token_logprobs=True, top_logprobs=5,
                  upstreams={"open-weight": {"url": "http://upstream", "kind": kind}})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client:
        await client.post("/v1/chat/completions", headers={"Authorization": "Bearer private-agent-key"},
                          json={"model": "open-weight", "messages": [], "temperature": 0.7, "top_p": 0.9})
    assert sent[0]["temperature"] == 0.7 and sent[0]["top_p"] == 0.9
    if kind == "vllm":
        assert sent[0]["logprobs"] is True and sent[0]["top_logprobs"] == 5 and sent[0]["return_token_ids"] is True
    else:
        assert "logprobs" not in sent[0] and "top_logprobs" not in sent[0]
    await app.state.gateway.client.aclose()


async def test_observation_logprobs_do_not_override_bound_training_sampling(tmp_path):
    sent = []

    def upstream(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    app = gateway(tmp_path, upstream, token_logprobs=True, top_logprobs=5)
    identity = {"base_model": "test/model", "base_revision": "a" * 40, "base_sha256": "b" * 64,
                "adapter_sha256": None}
    policy = PolicyIdentity(id=policy_id(**identity), model="open-weight", **identity)
    admin = GatewayAdmin("http://gateway", transport=httpx.ASGITransport(app))
    await admin.bind_rollout("a", "r", "episode", "task", 1, "session", policy)
    response = await request(app)
    assert response.status_code == 200
    assert all(sent[0][key] == value for key, value in SAMPLING.items())
    assert sent[0]["top_logprobs"] == 0
    await admin.aclose()
    await app.state.gateway.client.aclose()


async def test_unadvertised_encoding_is_rejected_with_raw_evidence(tmp_path):
    raw = b"encoded-body"
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([raw]),
                                                   headers={"content-encoding": "unsupported"}))
    response = await request(app)
    record = next(r for r in records(tmp_path) if "raw_capture" in r)
    assert response.status_code == 502 and captured_bytes(record) == raw
    assert record["raw_capture"]["response"]["error_type"] == "DecodingError"
    assert record["raw_capture"]["response"]["transport_complete"] is False
    await app.state.gateway.client.aclose()


async def test_nonstream_read_failure_preserves_partial_body_before_proxy_error(tmp_path):
    partial = b'{"choices":[{"message":{"reasoning":"unfinished'
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([partial], fail=True)))
    result = await request(app)
    assert result.status_code == 502
    record = next(r for r in records(tmp_path) if r["event"] == "upstream_error")
    assert captured_bytes(record) == partial
    assert record["raw_capture"]["response"]["termination"] == "transport_error"
    assert record["raw_capture"]["response"]["transport_complete"] is False
    assert record["raw_capture"]["response"]["error_type"] == "ReadError"
    await app.state.gateway.client.aclose()


async def test_stream_transport_failure_keeps_partial_capture_and_normalized_reasoning(tmp_path):
    partial = b'data: {"choices":[{"delta":{"reasoning":"partial"}}]}\n\n'
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([partial], fail=True)))
    with pytest.raises(httpx.ReadError, match="interrupted fixture"):
        await request(app, stream=True)
    record = records(tmp_path)[-1]
    assert captured_bytes(record) == partial
    assert record["raw_capture"]["response"]["termination"] == "stream_error"
    assert record["raw_capture"]["response"]["error_type"] == "ReadError"
    assert record["response"]["reasoning"] == "partial"
    await app.state.gateway.client.aclose()


async def test_truncated_gzip_after_done_is_never_a_complete_stream(tmp_path):
    raw = (b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
           b'data: [DONE]\n\n')
    truncated = gzip.compress(raw)[:-8]
    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Chunks([truncated]),
                                                   headers={"content-encoding": "gzip"}))
    with pytest.raises(httpx.DecodingError, match="ended before EOF"):
        await request(app, stream=True)
    record = records(tmp_path)[-1]
    assert captured_bytes(record) == truncated
    assert record["response"]["stream_complete"] is False
    assert "transport_stream_error" in record["response"]["stream_errors"]
    assert record["raw_capture"]["response"]["transport_complete"] is False
    assert record["missing_usage"] is True
    await app.state.gateway.client.aclose()


async def test_connection_failure_still_retains_forwarded_request(tmp_path):
    def unavailable(_):
        raise httpx.ConnectError("unavailable fixture")

    app = gateway(tmp_path, unavailable)
    result = await request(app)
    assert result.status_code == 502
    record = next(r for r in records(tmp_path) if r["event"] == "upstream_error")
    assert captured_bytes(record) == b""
    assert record["raw_capture"]["response"]["termination"] == "transport_error"
    assert record["raw_capture"]["response"]["error_type"] == "ConnectError"
    assert json.loads(base64.b64decode(record["raw_capture"]["request"]["body_base64"])) == record["request"]
    await app.state.gateway.client.aclose()


async def test_nonstream_invalid_utf8_error_keeps_all_bytes(tmp_path):
    raw = b'\xff\x00' + b'x' * 5000
    app = gateway(tmp_path, lambda _: httpx.Response(500, stream=Chunks([raw])))
    result = await request(app)
    assert result.status_code == 500 and result.content == raw
    assert captured_bytes(records(tmp_path)[-1]) == raw
    await app.state.gateway.client.aclose()


async def test_cancelled_nonstream_request_keeps_received_prefix(tmp_path):
    partial, waiting = b'{"choices":[', asyncio.Event()

    class Waiting(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield partial
            waiting.set()
            await asyncio.Event().wait()

    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Waiting()))
    pending = asyncio.create_task(request(app))
    await asyncio.wait_for(waiting.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    record = next(r for r in records(tmp_path) if r["event"] == "upstream_error")
    assert captured_bytes(record) == partial
    assert record["raw_capture"]["response"]["termination"] == "cancelled"
    assert record["raw_capture"]["response"]["transport_complete"] is False
    await app.state.gateway.client.aclose()


async def test_disconnect_retains_byte_exact_partial_stream(tmp_path):
    partial = b'data: {"choices":[{"delta":{"reasoning":"unfinished"}}]}\n\n'

    class Waiting(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield partial
            await asyncio.Event().wait()

    app = gateway(tmp_path, lambda _: httpx.Response(200, stream=Waiting()))
    disconnected, body_sent = asyncio.Event(), False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": json.dumps({"model": "open-weight", "messages": [], "stream": True}).encode(),
                    "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body") == partial:
            disconnected.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "POST", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "query_string": b"", "scheme": "http", "root_path": "", "server": ("gateway", 80),
             "client": ("127.0.0.1", 1234),
             "headers": [(b"authorization", b"Bearer private-agent-key"), (b"content-type", b"application/json")]}
    await asyncio.wait_for(app(scope, receive, send), 2)
    record = records(tmp_path)[-1]
    assert captured_bytes(record) == partial
    assert record["raw_capture"]["response"]["termination"] == "cancelled"
    assert record["raw_capture"]["response"]["transport_complete"] is False
    assert record["response"]["reasoning"] == "unfinished"
    await app.state.gateway.client.aclose()


async def test_stream_retains_selected_and_top_twenty_token_evidence(tmp_path):
    alternatives = [{"token": f"option-{i}", "bytes": list(f"option-{i}".encode()), "logprob": -0.1-i}
                    for i in range(20)]
    chosen = {"token": "chosen", "bytes": list(b"chosen"), "logprob": -0.1, "top_logprobs": alternatives}
    chunk = {"prompt_token_ids": [10, 11], "choices": [{"index": 0, "token_ids": [12],
             "delta": {"content": "chosen"}, "finish_reason": "stop", "logprobs": {"content": [chosen]}}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
    raw = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
    sent = []

    def upstream(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, stream=Chunks([raw[:27], raw[27:311], raw[311:]]),
                              headers={"content-type": "text/event-stream"})

    app = gateway(tmp_path, upstream, token_logprobs=True, top_logprobs=20)
    response = await request(app, stream=True)
    record = records(tmp_path)[-1]
    assert response.content == captured_bytes(record) == raw
    assert sent[0]["logprobs"] is True and sent[0]["top_logprobs"] == 20
    assert record["response"]["token_ids"] == [12]
    assert record["response"]["token_logprobs"] == [-0.1]
    recovered = json.loads(captured_bytes(record).splitlines()[0][6:])
    assert recovered["choices"][0]["logprobs"]["content"] == [chosen]
    assert record["raw_capture"]["response"]["capture_complete"] is True
    assert record["raw_capture"]["response"]["transport_complete"] is True
    await app.state.gateway.client.aclose()
