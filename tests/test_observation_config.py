"""Explicit open-weight token-probability collection and gateway-only wiring."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from mesocosm.core.manifest import ObserveSpec, WorldManifest
from mesocosm.env.compose import render_compose

ROOT = pathlib.Path(__file__).resolve().parents[1]


def world(observe=None, **overrides):
    return WorldManifest.model_validate({
        "name": "observation-contract", "model": {"id": "base-model"},
        "gateway": {"url": "http://localhost:8080"},
        "agents": [{"id": "agent-1"}, {"id": "agent-2"}, {"id": "agent-3"}],
        "observe": observe or {}, **overrides,
    })


def test_probability_collection_is_opt_in():
    spec = ObserveSpec()
    assert spec.token_logprobs is False and spec.top_logprobs == 0 and spec.activations is False
    assert ObserveSpec(token_logprobs=True).top_logprobs == 0


@pytest.mark.parametrize("top", [-1, 21, True, 1.5])
def test_alternative_count_requires_bounded_integer(top):
    with pytest.raises(ValidationError):
        ObserveSpec(token_logprobs=True, top_logprobs=top)


def test_alternatives_require_explicit_collection():
    with pytest.raises(ValidationError, match="requires observe.token_logprobs"):
        ObserveSpec(top_logprobs=1)
    assert ObserveSpec(token_logprobs=True, top_logprobs=20).top_logprobs == 20


def test_assigned_api_agents_cannot_silently_miss_requested_probabilities():
    with pytest.raises(ValidationError, match="requires vllm models"):
        world({"token_logprobs": True}, model={"id": "external", "kind": "openai"})
    with pytest.raises(ValidationError, match="agent-2"):
        world({"token_logprobs": True}, models=[{"id": "external", "kind": "openai"}],
              agents=[{"id": "agent-1"}, {"id": "agent-2", "model": "external"}, {"id": "agent-3"}])
    assert world(models=[{"id": "external", "kind": "openai"}]).observe.token_logprobs is False
    # An unused catalog route does not assert observation coverage for any actual agent.
    assert world({"token_logprobs": True}, models=[{"id": "monitor", "kind": "openai"}]).observe.token_logprobs


def test_compose_requests_ids_and_probabilities_only_in_gateway():
    manifest = world({"token_logprobs": True, "top_logprobs": 5})
    services = render_compose(manifest, repo_root=ROOT, run_label="observation-contract")["services"]
    environment = services["gateway"]["environment"]
    assert environment["MESOCOSM_RETURN_TOKEN_IDS"] == "1"
    assert environment["MESOCOSM_TOKEN_LOGPROBS"] == "1"
    assert environment["MESOCOSM_TOP_LOGPROBS"] == "5"
    assert environment["MESOCOSM_CAPTURE_MAX_BYTES"] == str(8 * 1024 * 1024)
    assert environment["MESOCOSM_CAPTURE_MAX_CHUNKS"] == "65536"
    for agent in manifest.agents:
        assert "MESOCOSM_TOKEN_LOGPROBS" not in services[agent.id].get("environment", {})
        assert "MESOCOSM_TOP_LOGPROBS" not in services[agent.id].get("environment", {})


def test_default_and_training_compose_keep_existing_sampling_contract():
    manifest = world()
    standard = render_compose(manifest, repo_root=ROOT, run_label="plain")["services"]["gateway"]["environment"]
    assert standard["MESOCOSM_TOKEN_LOGPROBS"] == "0"
    assert standard["MESOCOSM_TOP_LOGPROBS"] == "0"
    assert standard["MESOCOSM_RETURN_TOKEN_IDS"] == "0"
    training = render_compose(manifest, repo_root=ROOT, run_label="train",
                              require_rollout_binding=True)["services"]["gateway"]["environment"]
    assert training["MESOCOSM_RETURN_TOKEN_IDS"] == "1"
    assert training["MESOCOSM_TOKEN_LOGPROBS"] == "0"
    assert training["MESOCOSM_REQUIRE_ROLLOUT_BINDING"] == "1"
