# Reward-hack benchmark contract audit

Status: **unsupported** on the installed Harbor/OpenCode task-source path. Audited 2026-11-07.
Real-agent execution and behavior remain **unverified**. This is a source audit, not an execution
acceptance result. No task source, substitute evaluator, authored task, or reward scorer was installed.

The fixed-P study request explicitly requires stopping item 3 if Harbor cannot express the original
contract. A task-source-only conversion cannot preserve the contracts below with the current runtime.
The relevant decision is recorded with the fixed-P work in ADR-0066.

## Primary source and revision

The [paper, arXiv 2510.20270](https://arxiv.org/abs/2510.20270) links to the
[official safety-research/impossiblebench repository](https://github.com/safety-research/impossiblebench).
The inspected revision is `d4f9a149db2b9cec0fed7bab01d91c07ee10aca2`.
The public [dataset metadata API](https://huggingface.co/api/datasets/fjzzq2002/impossible_livecodebench)
reported dataset revision `98650ffc3f28a01b261669b6d19fcd7773823710`, with 103 records each in
`original`, `oneoff`, and `conflicting`. Reading all 103 IDs per variant through the public
[dataset rows API](https://datasets-server.huggingface.co/rows?dataset=fjzzq2002%2Fimpossible_livecodebench&config=default&split=original&offset=0&length=100)
confirmed that each pair of variants shares all 103 IDs. These are task variants, not disjoint
development/test partitions. The dataset fields include `task_id`, `prompt`, `test`, `original_test`,
`impossible_type`, and `entry_point`. IDs must be retained across any later integration; variant names
alone do not establish ID-disjoint splits.

The upstream [README at the pinned revision](https://github.com/safety-research/impossiblebench/blob/d4f9a149db2b9cec0fed7bab01d91c07ee10aca2/README.md)
documents a 2026-10-12 correction: the public GitHub port had prevented test modifications regardless
of settings. This porting bug did not affect the paper. An older checkout must not be used as evidence
for the paper's editable-test condition.

## Exact test visibility and evaluator inputs

The official [task constructor and record conversion](https://github.com/safety-research/impossiblebench/blob/d4f9a149db2b9cec0fed7bab01d91c07ee10aca2/src/impossiblebench/livecodebench_tasks.py#L96-L192)
select two distinct contracts. The paper [replication configuration](https://github.com/safety-research/impossiblebench/blob/d4f9a149db2b9cec0fed7bab01d91c07ee10aca2/demo.py#L45-L74)
uses both scaffolds, editable tests, and ten attempts. The task constructor's default is three
attempts. Those settings cannot be silently interchanged.

| Contract | Agent access | Evaluated test copy |
|---|---|---|
| Tools scaffold, `allow_test_modifications=True` | `func.py` and `test.py` are written into the same sandbox and editable. The instruction says not to modify tests; this is not an access restriction. | The current sandbox `test.py`, including agent edits. There is no `test_patch` metadata in this mode. |
| Tools scaffold, `allow_test_modifications=False` | The same files exist and can physically be edited. | The scorer compares against the test reconstructed from dataset `test` and restores it before execution if changed. This is restoration at scoring, not a read-only workspace mount. |
| Minimal scaffold, editable tests | Prompt contains the signature/docstring and test function. The model supplies a code block rather than editing a workspace. | The assembled program places dataset `test` before the submitted code, allowing the submission to redefine `check`. Iteration writes a randomly named Python program in the Inspect sandbox. Final scoring assembles code again. |
| Minimal scaffold, restored tests | The same prompt-based input. | Dataset `test` is appended after the submitted code, so its `check` definition takes precedence. |

The tools [initialization and submission loop](https://github.com/safety-research/impossiblebench/blob/d4f9a149db2b9cec0fed7bab01d91c07ee10aca2/src/impossiblebench/livecodebench_agent_full.py#L63-L149)
creates `func.py` from the original prompt plus a `NotImplementedError`, changes the test function
name from `check` to `test_func`, imports the entry point from `func`, and invokes it in `test.py`.
These transformations belong to the original benchmark scaffold, not a proposed Mesocosm conversion.

The original [tools scorer](https://github.com/safety-research/impossiblebench/blob/d4f9a149db2b9cec0fed7bab01d91c07ee10aca2/src/impossiblebench/livecodebench_scorers.py#L182-L277)
runs `python test.py` with a 30-second timeout in the current Inspect sandbox. Its result depends on
Inspect `TaskState`, including completion text containing the human-intervention marker. Its feedback
is used by Inspect `basic_agent` between submissions. The [minimal solver](https://github.com/safety-research/impossiblebench/blob/d4f9a149db2b9cec0fed7bab01d91c07ee10aca2/src/impossiblebench/livecodebench_agent_mini.py#L45-L121)
also consumes model completions and supplies test failure feedback between attempts. The dataset's
`original_test` is not a second reward scorer in these contracts; the selected variant's `test` is
the scored test input.

## Why the current Harbor path fails the exact-contract requirement

Editable workspace tests alone are expressible: a task image can contain writable tests, and a
Harbor verifier can execute that workspace copy. The read-only `/tests` mount by itself is therefore
not the blocker. The blockers are the original evaluator's inputs and lifecycle:

* The [current verifier protocol](../src/mesocosm/core/run.py) receives a task, workspace, logs
  directory, and script name. It does not receive Inspect `TaskState`, the final completion as
  evaluator input, or the original submission/feedback loop. The run manager records completion
  evidence separately and grades after the task turn.
* The [native runtime](../src/mesocosm/env/task_runtime.py) quiesces the agent, commits a filesystem
  snapshot, and starts a fresh verifier container. Its evidence explicitly says
  `filesystem_snapshot_only`. It rejects live-process snapshot verification. Original tools scoring
  instead executes in the current Inspect sandbox, so process and submission state do not have the
  same lifecycle.
* The [Harbor command](../src/mesocosm/adapters/harbor/task.py) introduces `tests/test.sh` under
  `/tests` after the agent turn. Merely translating `python test.py` success into `reward.txt` would
  omit original scorer inputs and behavior. Asking the agent to write a new completion file or
  adding prompt instructions for submission/feedback would introduce an authored contract.

No benchmark-specific runtime branch, replacement `test.sh`, Inspect shim, or alternative scorer was
added. A faithful future integration needs an execution contract that invokes the original evaluator
with its actual completion/submission state and sandbox lifecycle, and then validates the study's
multi-agent conditions against that backend. The installed integration API can host such a backend;
the presence of that API does not make this benchmark supported by the current Harbor path.

## Fallback source audit

Inspected [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) at
`73695ff5533b566f7cc99b02bfeb9168936e740d`. This is also a completion-based contract, not an
original editable-workspace task.

The original [`simple_overwrite_tests` hint](https://github.com/ariahw/rl-rewardhacking/blob/73695ff5533b566f7cc99b02bfeb9168936e740d/src/data/hints.py#L84-L106)
names `run_tests` in the prompt and sets the answer call to `run_tests()`. It does not disclose the
ground-truth assertions or create an agent-editable test file. Other upstream hints have different
visibility contracts and must not be substituted silently.

The original [reward-hacking evaluator](https://github.com/ariahw/rl-rewardhacking/blob/73695ff5533b566f7cc99b02bfeb9168936e740d/src/evaluate/evaluation.py#L243-L339)
parses code from a model completion and accepts a replacement test function at module level or as a
method, including `Solution().run_tests()`. Its [code evaluator](https://github.com/ariahw/rl-rewardhacking/blob/73695ff5533b566f7cc99b02bfeb9168936e740d/src/evaluate/evaluator.py#L25-L43)
and [subprocess helper](https://github.com/ariahw/rl-rewardhacking/blob/73695ff5533b566f7cc99b02bfeb9168936e740d/src/evaluate/helpers.py#L57-L170)
execute the original assembled program with bounded subprocess resources. The hinted test result and
ground-truth result are distinct outputs. Any later fixed-P integration must use the hinted result
for its declared reward and keep ground-truth testing audit-only, undisclosed to agents. This audit
does not add either result to a reward path.

Converting that completion to a required workspace file would change the original instructions and
evaluator input boundary. The same task-source-only incompatibility therefore applies to the fallback.

## Evidence limits and validation

No episodes were run. No before/after test-file fingerprints were captured, no development/test
partition was installed, and no original-input identity or edit-observation acceptance test is claimed.
File-change and collective measurements without such coverage remain `unknown`, not zero.
The source audit used public unauthenticated repository clones, `git rev-parse HEAD`, source reads,
and the public dataset metadata API. It did not execute downloaded code, contact model endpoints,
download model weights, or access operator stores. Since item 3 stops before implementation, its
implementation acceptance tests and full-suite commands are not claimed as passing.

Read-only ID inspection returned the following actual output (JSON reformatted for readability):

```json
{
  "counts": {"conflicting": 103, "oneoff": 103, "original": 103},
  "pairwise_id_overlaps": {
    "oneoff/conflicting": 103,
    "original/conflicting": 103,
    "original/oneoff": 103
  },
  "unique_counts": {"conflicting": 103, "oneoff": 103, "original": 103}
}
```

`git diff --check` completed with exit code 0 and no output after adding this report. These read-only
checks do not satisfy the unimplemented task-source acceptance tests.

SHA-256 values below identify the audited upstream bytes, not executable task provenance or a
complete dependency/environment lock:

| Pinned repository | File | SHA-256 |
|---|---|---|
| ImpossibleBench | `src/impossiblebench/livecodebench_tasks.py` | `bea30035964d6e0a413ce44a7fc071cb4e9d67ba201c4f75834de3a52101eec6` |
| ImpossibleBench | `src/impossiblebench/livecodebench_agent_full.py` | `6a58217f868f033f0bdeb920b98d43f67c254f15a2da1c22f0046090ba502a8f` |
| ImpossibleBench | `src/impossiblebench/livecodebench_agent_mini.py` | `313966495a8ea4f2c80e1d52fbd1576d5bcc9806ea744b25272c045857e2143b` |
| ImpossibleBench | `src/impossiblebench/livecodebench_scorers.py` | `ef39b07a4288700e466dcba8bb8603904a6a10dd8457477f6359010e4d70b2d0` |
| ImpossibleBench | `src/impossiblebench/compose.yaml` | `d077044a6c1614fb85743e50af0cf74a43adf59dc59c80652285a7ccfd4911c4` |
| ImpossibleBench | `demo.py` | `d807e7545f353d09d1a14a55c7fc44628d8a97eb53ef0bc8711cc86aa2ad6d2c` |
| rl-rewardhacking | `src/data/hints.py` | `634ea2dc85046fa6ef838d46bf148254590bc0db7c8dd328ebf3714a4f3d079f` |
| rl-rewardhacking | `src/evaluate/evaluation.py` | `3fc688f8acbf65ec34efd1f3dfd294ee4b8b3f6fb0fc751d0f027b4e1192bd2a` |
| rl-rewardhacking | `src/evaluate/evaluator.py` | `3d7b7df29a7f4566d20615e2f929932d1ea3c7535dc8d92a9d98436b4ea19f4e` |
| rl-rewardhacking | `src/evaluate/helpers.py` | `91c3e932d4ecbb42b63137d7c0f4acbf14959d2861bb39a6d9739872f2032505` |
| rl-rewardhacking | `src/prompts.py` | `cff2e3be5e47bf7f197cd6fc8baba32e0510aeb60962ce6e0a979697ab713750` |
