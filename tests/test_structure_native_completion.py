"""Owned Docker command contracts; no live containers or real-agent acceptance are claimed."""
from __future__ import annotations

import pytest
from test_task_runtime import DockerCommands, task

from mesocosm.core.artifacts import ArtifactSpec
from mesocosm.core.manifest import AgentSpec
from mesocosm.env.task_runtime import TaskEnvironmentError


async def test_native_work_quiesces_captures_and_removes_owned_container_without_original_grading(tmp_path):
    runtime = DockerCommands(tmp_path)
    native, workspace = task(tmp_path), tmp_path / "workspace"
    await runtime.prepare(AgentSpec(id="agent-1"), native, workspace, reset=True)
    before = len(runtime.calls)
    receipt = await runtime.complete_work(native, workspace, tmp_path / "logs",
                                          ArtifactSpec(enabled=True, paths=["native-source.txt"]))
    assert receipt["status"] == "complete" and receipt["cleanup_verified"]
    assert receipt["original_verifier_invoked"] is False
    assert receipt["artifacts"]["files"]["native-source.txt"]["text"] == "source from the task image"
    assert receipt["artifacts"]["capture_evidence"]["boundary"] == "quiesced_completed_work_without_grading"
    calls = runtime.calls[before:]
    pause = next(index for index, call in enumerate(calls) if call[:3] == ("docker", "container", "pause"))
    remove = next(index for index, call in enumerate(calls) if call[:3] == ("docker", "container", "rm"))
    assert pause < remove and not runtime.containers and not runtime.active
    assert not any(call[:2] == ("docker", "commit") or call[:2] == ("docker", "run") for call in calls)


@pytest.mark.parametrize("failure", ["pause", "cleanup"])
async def test_uncertain_quiescence_or_cleanup_never_returns_completion_receipt(tmp_path, failure):
    runtime = DockerCommands(tmp_path)
    native, workspace = task(tmp_path), tmp_path / "workspace"
    await runtime.prepare(AgentSpec(id="agent-1"), native, workspace, reset=True)
    runtime.pause_error = failure == "pause"
    runtime.cleanup_error = failure == "cleanup"
    with pytest.raises(TaskEnvironmentError, match="quiesced|cleanup failed"):
        await runtime.complete_work(native, workspace, tmp_path / "logs")
    if failure == "pause":
        assert not runtime.containers
    else:
        assert runtime.containers
        runtime.cleanup_error = False
        await runtime.finish()


async def test_work_completion_rejects_side_verifier_and_absent_preparation(tmp_path):
    runtime = DockerCommands(tmp_path)
    native, workspace = task(tmp_path), tmp_path / "workspace"
    with pytest.raises(TaskEnvironmentError, match="no native task environment"):
        await runtime.complete_work(native, workspace, tmp_path / "logs")
    await runtime.prepare(AgentSpec(id="agent-1"), native, workspace, reset=True)
    (native.path / "tests" / "side_test.sh").write_text("exit 0\n")
    native.has_side_tests = True
    with pytest.raises(TaskEnvironmentError, match="cannot skip side-task grading"):
        await runtime.complete_work(native, workspace, tmp_path / "logs")
    await runtime.finish()
