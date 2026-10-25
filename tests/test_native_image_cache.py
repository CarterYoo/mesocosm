from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_task_runtime import DockerCommands, task

from mesocosm.env.native_image_cache import (
    PREFIX,
    PreparedImageError,
    expected_labels,
    find_prepared_image,
    image_fingerprint,
    load_import_receipt,
)
from mesocosm.env.task_runtime import (
    TaskEnvironmentError,
    environment_fingerprint,
    environment_key,
    harness_dockerfile,
    startup_script,
)

BASE_ID = "sha256:" + "a" * 64
AGENT_ID = "sha256:" + "b" * 64


def images(labels):
    base = {"Id": BASE_ID, "Os": "linux", "Architecture": "arm64", "RootFS": {"Layers": ["sha256:base"]},
            "Config": {"User": "root", "WorkingDir": "/candidate", "Entrypoint": [], "Cmd": [], "Env": []}}
    agent = copy.deepcopy(base)
    agent.update(Id=AGENT_ID, RootFS={"Layers": ["sha256:base", "sha256:harness"]})
    agent["Config"]["Labels"] = {**labels, PREFIX + "base-image-id": BASE_ID}
    return base, agent


class Inspect:
    def __init__(self, base, agent):
        self.base, self.agent, self.calls = base, agent, []

    async def __call__(self, *args):
        self.calls.append(args)
        assert args[:3] == ("docker", "image", "inspect")
        return json.dumps([self.base if args[-1] == BASE_ID else self.agent]).encode()


def labels():
    return expected_labels("c" * 64, "d" * 64, "1.19.3", "linux/arm64")


async def lookup(command, wanted=None, imported=None):
    return await find_prepared_image(command, tag="mesocosm-task-agent:test", labels=wanted or labels(),
                                     platform="linux/arm64", imported=imported, import_sha256="e" * 64)


async def test_validated_labels_return_immutable_image_and_original_base_without_build():
    command = Inspect(*images(labels()))
    result = await lookup(command)
    assert result["agent"]["Id"] == AGENT_ID and result["base"]["Id"] == BASE_ID
    assert result["evidence"]["kind"] == "validated_labels"
    assert len(command.calls) == 2


async def test_unlabeled_tag_is_not_a_trusted_cache_hit():
    base, agent = images(labels())
    agent["Config"].pop("Labels")
    assert await lookup(Inspect(base, agent)) is None


@pytest.mark.parametrize("field", ["environment-sha256", "recipe-sha256", "opencode-version", "platform"])
async def test_source_recipe_version_and_platform_drift_fail_closed(field):
    base, agent = images(labels())
    agent["Config"]["Labels"][PREFIX + field] = "changed"
    with pytest.raises(PreparedImageError, match="labels differ"):
        await lookup(Inspect(base, agent))


@pytest.mark.parametrize("mutation", ["base_id", "layers", "architecture", "agent_id"])
async def test_immutable_image_and_layer_lineage_are_verified(mutation):
    base, agent = images(labels())
    if mutation == "base_id":
        base["Id"] = "sha256:" + "f" * 64
    elif mutation == "layers":
        agent["RootFS"]["Layers"][0] = "sha256:other-base"
    elif mutation == "architecture":
        agent["Architecture"] = "amd64"
    else:
        agent["Id"] = "mutable:tag"
    with pytest.raises(PreparedImageError):
        await lookup(Inspect(base, agent))


def import_entry(base, agent, wanted=None):
    wanted = wanted or labels()
    return {"environment_sha256": wanted[PREFIX + "environment-sha256"], "labels": wanted,
            "base_image_id": BASE_ID, "agent_image_id": AGENT_ID,
            "base_fingerprint": image_fingerprint(base), "agent_fingerprint": image_fingerprint(agent),
            "proof": {"path": "proof.json", "sha256": hashlib.sha256(b"{}\n").hexdigest()}}


async def test_explicit_frozen_import_can_adopt_unlabeled_image_but_not_drifted_configuration():
    base, agent = images(labels())
    agent["Config"].pop("Labels")
    imported = {"entries": [import_entry(base, agent)]}
    result = await lookup(Inspect(base, agent), imported=imported)
    assert result["agent"]["Id"] == AGENT_ID
    assert result["evidence"]["kind"] == "frozen_import"
    agent["Config"]["Env"].append("UNEXPECTED=value")
    with pytest.raises(PreparedImageError, match="fingerprint mismatch"):
        await lookup(Inspect(base, agent), imported=imported)


async def test_import_cannot_match_another_task_or_another_final_image():
    base, agent = images(labels())
    entry = import_entry(base, agent)
    with pytest.raises(PreparedImageError, match="exact task"):
        await lookup(Inspect(base, agent), imported={"entries": []})
    agent["Id"] = "sha256:" + "f" * 64
    with pytest.raises(PreparedImageError, match="agent image ID mismatch"):
        await lookup(Inspect(base, agent), imported={"entries": [entry]})


def test_receipt_and_nested_proof_hashes_are_both_required(tmp_path, monkeypatch):
    base, agent = images(labels())
    payload = {"schema_version": 1, "kind": "mesocosm.native-image-import.v1", "entries": [import_entry(base, agent)]}
    receipt = tmp_path / "import.json"
    receipt.write_text(json.dumps(payload))
    proof = tmp_path / "proof.json"
    proof.write_text("{}\n")
    monkeypatch.setenv("MESOCOSM_PREPARED_IMAGE_RECEIPT", str(receipt))
    monkeypatch.delenv("MESOCOSM_PREPARED_IMAGE_SHA256", raising=False)
    with pytest.raises(PreparedImageError, match="both"):
        load_import_receipt()
    monkeypatch.setenv("MESOCOSM_PREPARED_IMAGE_SHA256", hashlib.sha256(receipt.read_bytes()).hexdigest())
    assert load_import_receipt()[0] == payload
    proof.write_text("changed")
    with pytest.raises(PreparedImageError, match="proof hash"):
        load_import_receipt()
    receipt.write_text("{}")
    with pytest.raises(PreparedImageError, match="receipt hash"):
        load_import_receipt()


class PreparedCommands(DockerCommands):
    def __init__(self, root, native):
        super().__init__(root)
        recipe = harness_dockerfile("native-image", "1.19.3", "native-user")
        recipe += startup_script({"Cmd": ["native-command"]})
        self.cache_labels = expected_labels(environment_fingerprint(native, "1.19.3"),
                                           hashlib.sha256(recipe.encode()).hexdigest(), "1.19.3", None)
        self.cached_base, self.cached_agent = images(self.cache_labels)

    async def command(self, *args, **kwargs):
        if args[:3] == ("docker", "image", "inspect"):
            self.calls.append(args)
            return json.dumps([self.cached_base if args[-1] == BASE_ID else self.cached_agent]).encode()
        return await super().command(*args, **kwargs)


async def test_fresh_runtime_reuses_persisted_image_without_any_pull_build_or_tag(tmp_path):
    native = task(tmp_path)
    runtime = PreparedCommands(tmp_path, native)
    result = await runtime.native_image(native)
    assert result.image == AGENT_ID and result.base_id == BASE_ID
    assert result.config["WorkingDir"] == "/candidate"
    assert not any(call[:2] in [("docker", "build"), ("docker", "pull"), ("docker", "tag")]
                   for call in runtime.calls)
    assert runtime.receipt("container", result)["prepared_image"]["kind"] == "validated_labels"


async def test_new_build_labels_bind_full_source_fingerprint_and_exact_base(tmp_path):
    native = task(tmp_path)
    runtime = DockerCommands(tmp_path)
    await runtime.native_image(native)
    build = next(call for call in runtime.calls if call[:2] == ("docker", "build"))
    assert PREFIX + "environment-sha256=" + environment_fingerprint(native, "1.19.3") in build
    assert PREFIX + "base-image-id=sha256:native-base" in build
    assert environment_key(native, "1.19.3") == environment_fingerprint(native, "1.19.3")[:24]


async def test_unexpected_prepared_source_drift_is_explicit_runtime_error(tmp_path):
    native = task(tmp_path)
    runtime = PreparedCommands(tmp_path, native)
    runtime.cached_agent["Config"]["Labels"][PREFIX + "recipe-sha256"] = "different"
    with pytest.raises(TaskEnvironmentError, match="prepared-image verification"):
        await runtime.native_image(native)
