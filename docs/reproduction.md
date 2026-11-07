# Research-agent workbench

The workbench connects incident evidence, experimental factors, demonstration learning,
real-world execution and behavior evaluation. It exposes 19 Python/JSON tools over Mesocosm's
existing runtime. It does not insert scripted outcomes into the world.

This is a software interface, not a new claim of successful behavior reproduction. The
[185-item requirements audit](reproduction-requirements.md) maps every requested item to code,
tests and remaining limitations. A requirement marked **partial** is not complete.

For SFT adapter deployment, verified runtime lineage and repeated four-condition experiments,
see [From a trained adapter to a measured experiment](verified-research-loop.md). This path keeps
serving API attribution separate from numerical weight verification and real-agent acceptance.

## Start with a frozen question

```bash
uv run mesocosm research tools --json
uv run mesocosm research call list_factors --request request.json --root runs/research --json
```

`request.json` contains a tool's argument object, for example `{}` for `list_factors`.
Every command emits one JSON object. The private store contains immutable content-addressed
objects, a durable job table, idempotency records and a hash-chained operator audit.

The Python interface exposes the same tools:

```python
from mesocosm.reproduction.service import ResearchWorkbench

workbench = ResearchWorkbench(
    "runs/research",
    allowed_workers={"https://your-worker.example": "MESOCOSM_WORKER_TOKEN"},
)
goal = workbench.define_behavior({
    "name": "observed-shell-execution",
    "version": "1",
    "stages": {
        "execution": {
            "event": {"kind": "action_execution", "where": {"data.tool": "bash"}}
        }
    },
    "required_stages": ["execution"],
})
candidate = workbench.configure_experiment(
    base_world="/absolute/path/to/your/native-world.yaml",
    behavior_id=goal["behavior_id"],
    changes=[{
        "factor": "agent.budget.requests",
        "scope": "role",
        "target": "reviewers",
        "value": 30,
    }],
    roles={"reviewers": ["agent-1", "agent-2"]},
)
validation = workbench.validate_experiment(candidate["experiment_id"])
```

The example rubric tests an observed execution category only. A shell request, an executed command,
successful execution, a changed file and an eventual task outcome are different findings. Define
the stages your research actually needs before running candidates.

`configure_experiment` copies explicitly referenced original task inputs into a portable package.
Later edits to the source directory do not silently change that candidate. The built-in worker
requires research containment, finite comparison worlds and at least three agents. Models still
need an operator-provided serving endpoint; declaring a model revision does not verify the server.

## Tool contracts

| Tool | Result and boundary |
|---|---|
| `inspect_transcript` | Imports JSON, JSONL, text or a recorded bundle, retaining evidence provenance and unknowns. Text alone does not prove execution. |
| `define_behavior` | Freezes a versioned rubric for onset, transmission, adoption, execution and/or outcome. |
| `list_factors` | Reports allowlisted fields, types, scopes, defaults, current values and enforcement boundaries. |
| `configure_experiment` | Applies ordered global/role/agent changes and freezes original executable inputs with the rubric. |
| `validate_experiment` | Revalidates the frozen package and compatibility. Live model and Docker readiness remain separate. |
| `prepare_dataset` | Freezes recorded observation/action examples, annotations, selection and group-disjoint splits. |
| `submit_training` | Queues actual local SFT/LoRA using pinned local model files; no cloud resource is allocated. |
| `run_experiment` | Submits one frozen candidate to an existing authenticated Linux worker. |
| `get_job_status` | Inspects, cancels, retrieves results or reconciles an uncertain submission without a new run identity. |
| `snapshot_experiment` | Verifies and imports an existing closed-run snapshot with explicit filesystem coverage. |
| `fork_experiment` | Restores captured files into an isolated fork and records factor changes. Preparation is distinct from execution. |
| `evaluate_candidate` | Applies the candidate's frozen rubric with present/absent/unknown judgments and cited events. |
| `compare_candidates` | Compares incident-clustered observations; failures, missing outcomes and unknown costs remain visible. |
| `export_reproducer` | Exports portable original inputs, rubric, evaluations and evidence with content hashes. |
| `deploy_policy` | Verifies a base or loads an owned completed SFT adapter on an approved shared-filesystem vLLM endpoint. |
| `evaluate_run` | Recovers the candidate from a fetched job and checks runtime/model/harness/verifier evidence before evaluation. |
| `plan_factorial` | Freezes base/trained × baseline/designed candidates, budgets and every scheduled repetition. |
| `advance_factorial` | Reconciles a durable current trial and submits at most one next trial without outcome-selected retries. |
| `report_factorial` | Retains all scheduled trials and reports unknown-aware descriptive condition differences. |

Use `research tools --json` for current argument names and required/default values. Unknown tool
names and arguments are rejected. Unsupported factors are rejected rather than converted into prompts.
Configuration changes are pre-run; existing budget conditions remain the supported live factor API.

## Recorded behavior and uncertainty

Rubric rules support `event`, `all`, `any`, ordered `sequence` and `not`. Event selectors use an
exact event `kind`, dotted `where` field equality and optional `min_count`. Each stage retains the
matched event IDs. Definitions and normalized transcripts are hashed before evaluation.

Absence requires evidence coverage. An incomplete transcript with no matching event produces
`unknown`, not `absent`. Coverage declarations are source claims, not independent proof that no
activity escaped collection. Native tool errors do not establish execution: a before-tool hook
could have rejected the request. Recorded links and ordering do not establish causal transmission.

Comparisons deduplicate reimported evidence and consolidate fragments of the same execution.
Sessions are grouped by incident/source. Descriptive bootstrap intervals use these clusters,
not tokens as independent samples. Unknown-outcome bounds and missing/conflicting costs are
reported separately. A single source cannot establish cross-incident generalization.

## Demonstration learning

Install the existing optional `learning` dependencies on the selected training host. Provide
local base weights and tokenizer; the trainer never downloads them or allocates a GPU.

Input conversation rows contain `id`, `session_id`, optional `incident_id`, `task_id`, `agent_id`,
an explicit `label` (`normal`, `target`, `recovery`) and `messages`. Native gateway records can also
be imported with explicit label overrides. An annotation is not a measured behavioral finding.

Each assistant action becomes one example using only its preceding messages. Connected incident,
task and session groups stay in one split. Explicit split assignments that cut such a group fail.
Source bytes, selections and hashes are retained. This prevents structural leakage; it cannot
automatically identify a hidden answer already embedded in a user's input text.

```python
dataset = workbench.prepare_dataset(
    sources=["/absolute/path/to/recorded-conversations.jsonl"],
    config={"seed": 42, "reasoning_policy": "preserve", "max_context_tokens": 4096},
)
job = workbench.submit_training(
    dataset_id=dataset["dataset_id"],
    idempotency_key="study-a-sft-001",
    config={
        "backend": "sft_lora",
        "base_model": "/absolute/path/to/local-base-model",
        "base_sha256": "<checkpoint_sha256 of that directory>",
        "device": "cpu",
        "max_steps": 100,
        "max_seconds": 300,
        "max_train_tokens": 100000,
        "max_eval_tokens": 10000,
        "lora_rank": 8,
        "target_modules": ["q_proj", "v_proj"],
    },
)
```

Obtain the model digest with `mesocosm.learning.checkpoints.checkpoint_sha256(Path(...))`.
A separate tokenizer also needs a matching tokenizer digest. Templates must preserve supplied
tool and reasoning information or fail explicitly. Loss applies only to the selected assistant
action. Context length is checked with the pinned tokenizer; truncation is not silently applied.

Checkpoints retain adapter, optimizer, scheduler, RNG state and data cursor. Resuming is an explicit
new job referencing a stopped prior job with the same configuration; the original attempt remains
intact. Reserved training/evaluation token budgets are not refunded after interrupted computation.
These token budgets count assistant targets, not every forward-pass input token. Setup, input
hashing and model loading also consume wall time. Training from a CLI invocation can continue after
the submitting client exits; `start=false` queues without launching a worker.

Queued training runs in an owned child process. A supervisor and an independent child alarm bound
wall time; a hung model forward cannot bypass the batch-level checks. Cancellation stops the owned
process group. A lost parent leaves the bounded child holding the execution lease, preventing a
second worker from repeating its updates. Recovery records an interrupted attempt without retrying
it. Resumed attempts retain the original time allowance and pinned training implementation sources.

SFT/LoRA updates actual weights. QLoRA, full fine-tuning, preference learning and joint multi-agent
optimization are currently unsupported in this trainer. The existing verified-world REINFORCE
implementation remains available through `mesocosm train`; it has a separate rollout contract.
Tiny random-model unit tests verify gradient and resume mechanics, not research performance.

## Real execution and retry behavior

```python
# Run inside an async caller. Set MESOCOSM_WORKER_TOKEN in the process environment.
job = await workbench.run_experiment(
    experiment_id=candidate["experiment_id"],
    idempotency_key="study-a-native-001",
    worker="https://your-worker.example",
)
status = await workbench.get_job_status(job["job_id"])
# Once terminal:
results = await workbench.get_job_status(job["job_id"], action="fetch")
```

The existing Linux worker owns containment, execution, original evaluation and cleanup. A lost
HTTP response can leave a job `submission_uncertain`. Inspect it; an explicit `resubmit` uses the
same remote job ID and frozen package. It does not create a replacement trial. Reusing an idempotency
key with different inputs fails. Unknown invoice cost stays unknown, not zero.

Worker endpoints and credential environment-variable names are fixed by the operator when creating
the workbench. A tool request cannot redirect credentials to an arbitrary endpoint. For the CLI,
the operator supplies `--worker URL --worker-token-env MESOCOSM_WORKER_TOKEN` outside the request JSON.
Keep these operator flags outside the tools exposed to the research agent.

## Snapshot, fork and reproducibility limits

First use the existing snapshot interface against actual captured state. Workbench import checks
the original parent's terminal summary and successful cleanup evidence, file/tar hashes and safe
bounded paths. A caller's statement that a world is paused is insufficient. All copied state is
restored into fresh paths and verified before a fork can run.

This covers workspaces and supplied service/shared files. Harness sessions, requests in flight,
gateway counters, uncaptured container files and RNG state are not an exact prefix checkpoint.
The fork starts fresh sessions. Remote fresh-world submission refuses partial-state fork objects
instead of silently dropping restored state. See the returned local fork recipe and
[snapshot contract](adr/0015-persistent-operation-and-snapshots.md).

Exported packages include executable task inputs; GPU weights and worker connection credentials
are supplied separately. They require a compatible installed backend and explicitly supplied model
serving. A new run is re-execution, not deterministic replay. Original evidence and task inputs are
unredacted and may themselves contain sensitive content; review them before sharing an export.

Imported source bytes are retained separately from normalized events and rehashed before evaluation
and export. The export includes these originals, known attempt records and a rerun script that saves
its job identity before submitting. Its integrity index is checked on repeated export calls.

## Trust boundary and held-out evaluation

The store belongs to one trusted research operator and uses mode `0700`. The normal tool interface
rejects declared test candidates and known test evidence, including comparison/export of their
evaluations; independent operators can explicitly enable it.
An agent with unrestricted filesystem/process access can read that operator's data or launch a new
interface. This is not an OS sandbox, automatic semantic test-set detector or multi-tenant ACL.
Keep held-out data outside search agents' filesystem permissions and expose only allowed tools.

Hashes detect consistency failures; they do not authenticate forged external records. The operator
must review source provenance and original evaluator results before admitting research observations.
An imported transcript associated by the operator remains an annotation unless its execution lineage
is independently bound to the candidate. A rubric frozen before that annotation is not automatically
evidence of prospective preregistration before the historical event.

## Original workbench verification — 2026-11-06

These counts describe the initial workbench at `4bfe55e`, before the verified-loop extension.
Current extension checks and limits are in [the verified-loop guide](verified-research-loop.md).

- Full repository suite: **1,698 passed**, with four existing staged-scenario deprecation warnings.
- New research-workbench suite: **149 tests**, included in that full run.
- Mypy: **112 source files**, no errors. Ruff: all checks passed.
- Actual local tiny-model LoRA updates, optimizer/RNG resume and supervised child-process deadlines
  were exercised. These are software fixtures, not benchmark learning data or behavior evidence.
- No new GPU allocation, model download, native Docker run or three-agent acceptance was performed.

The work was tested from the `codex/research-workbench` checkout, based on local runtime commit
`bd2c598ae9345d7c9beec6730745d56473ec3553`. Existing unrelated activation work was left unchanged.
