# From a trained adapter to a measured experiment

The workbench can connect an owned SFT checkpoint, its serving registration, a frozen candidate,
the worker's executed configuration, original model calls and a behavior evaluation. It can also
prepare and advance a repeated **base/trained model × baseline/designed environment** comparison.
These connections have software-contract tests. A new real three-agent run remains **unverified**.

## Deployment boundary

Run the workbench's training/deployment control process on a host whose checkpoint filesystem is
shared with vLLM. `deploy_policy` hashes existing local base/adapter files and uses the existing
`VLLMPolicyServer` load/verify APIs. It does **not** upload Mac files to a remote server, download
weights, allocate a GPU, or start a worker. A path string on two unrelated hosts is not shared storage.
The Linux task worker can be a different host, provided its gateway can reach the same serving URL.

For Python callers as well as the CLI, approving a serving endpoint asserts that these shared
checkpoint paths are available to both processes. Allowlist both endpoints in the operator-owned
constructor:

```python
workbench = ResearchWorkbench(
    "/private/research",
    allowed_workers={WORKER_URL: "MESOCOSM_WORKER_TOKEN"},
    allowed_servers={SERVING_URL: "VLLM_API_KEY"},  # None for a trusted keyless endpoint
)
```

The equivalent CLI flags are `--worker`, `--worker-token-env`, `--server`,
`--server-token-env`, and `--shared-checkpoint-paths`. Keep these operator options outside an
untrusted research agent's tool arguments. The worker must independently allow the model's credential
variable and have its value. Credentials are not shipped in candidates.

Deploy the already-served base, then the completed workbench SFT job:

```python
base = await workbench.deploy_policy(
    server=SERVING_URL, model=BASE_ALIAS, base_model=PINNED_BASE,
    base_revision=HF_COMMIT, base_sha256=BASE_SHA256,
    dtype="bfloat16", idempotency_key="base-deployment-001",
)
trained = await workbench.deploy_policy(
    server=SERVING_URL, model=NEW_ADAPTER_ALIAS, base_model=PINNED_BASE,
    base_revision=HF_COMMIT, base_sha256=BASE_SHA256,
    dtype="bfloat16", training_job_id=COMPLETED_SFT_JOB,
    idempotency_key="trained-deployment-001",
)
```

Use the actual pinned 40-character revision and checkpoint content hash. No revision is inferred
from a directory name. SFT deployment currently requires a completed, nonzero update with the base
tokenizer and matching serving dtype. The existing SFT backend's float32 output is not silently cast
to bfloat16; direct float32 serving deployment is **unsupported** in this path. Existing serving
configuration checks reject quantized weights and incompatible overrides.

Adapter names cannot replace existing registrations. A lost load acknowledgement leaves an
`activation_uncertain` job; `get_job_status(job_id, action="reconcile")` verifies the same alias and
artifact without issuing another load. A per-server/alias lease prevents concurrent activation or
reconciliation; repeated reconciliation of a ready job returns the same deployment. Failed or
ambiguous attempts remain in the private store.
Reusing the same deployment request returns its original receipt; readiness is checked again before
every candidate submission.

## What “verified” means

| Check | Evidence | Limit |
|---|---|---|
| SFT artifact | Owned completed job, frozen dataset/config/source identity, committed checkpoint files and adapter hash | Does not establish learning effectiveness |
| Serving activation | Shared local artifact hashes, load API, registered root, resolved server configuration and a real one-token probe | Trusts the operator's shared filesystem and model server |
| Runtime configuration | Worker input digest, executed manifest, original task/external-input hashes before and after execution | Applies to the native Docker single-world path |
| Agent model attribution | Authenticated agent key, assigned alias, deployment identity, original request/response bytes and observed response alias | Not cryptographic GPU weight attestation or numerical logprob parity |
| Evaluation | Matching job/run, complete model-call accounting, original harness turns/transcripts and original verifier scores | No automatic semantic judgment of deception or collusion |

The manifest's `model.serving_identity` is a declaration until matched to an owned deployment and
worker result. The gateway's identity-only binding preserves sampling settings. It does not enable
the separate REINFORCE sampling contract. An opted-in agent cannot select another alias through the
default route. Conflicting or missing streamed model IDs invalidate lineage. Streaming tokens may
already have reached the harness before final validation; this is an evaluation admission check.

The result explicitly retains `numerical_weight_verification: unverified`. The existing RL numerical
parity verifier has a different constrained contract; this feature does not claim to have run it.

## Evaluate the actual run

`configure_experiment(..., deployment_id=...)` binds the shared policy for every agent. Heterogeneous
per-agent deployment binding is not yet supported by this workbench method. Submit with
`run_experiment`, fetch with `get_job_status(..., action="fetch")`, and use:

```python
evaluation = workbench.evaluate_run(job_id=EXECUTION_JOB, run_id=FETCHED_RUN_ID)
```

The candidate is recovered from the job, not assigned by the caller. The validator compares the
packaged world with the actual runtime manifest, permitting only the worker's documented path and
endpoint relocation. It reconciles all admitted calls, their original captured bytes and normalized
events, requires calls and independent harness sessions for all intended agents, and checks recorded
task rewards against the original verifier output files. A genuine score of zero is valid.

This initial admission path requires native Docker comparison mode with `task_assignment.policy: all`
and at least three agents. Missing records, failed execution and cleanup failure cannot become admitted
behavior evaluations. They remain trial records. Comparison/export rechecks the retained inputs and
result inventory. Exports include complete original fetched result bundles and deployment receipts.

`inspect_transcript` plus `evaluate_candidate` remains available for external/historical evidence.
Its operator-supplied candidate association is **unverified**, even if its filenames and IDs resemble
a known run. Never mix that annotation with verified runtime provenance silently.

Behavior coverage remains explicit. A nonmatching rule is `unknown` unless the relevant evidence
coverage supports absence. Successful execution alone does not make every kind of behavior observable.
Start with a narrowly defined recorded action or original task outcome and inspect its cited evidence.

## Freeze a four-condition study

First freeze a baseline candidate with original external tasks, a fixed rubric, reset memory, and
finite request/completion-token budgets for every agent. Then:

```python
study = workbench.plan_factorial(
    baseline_experiment_id=BASELINE_CANDIDATE,
    base_deployment_id=base["deployment_id"],
    trained_deployment_id=trained["deployment_id"],
    environment_changes=ENVIRONMENT_FACTORS,
    repetitions=3, order_seed=42, idempotency_key="study-001",
)
```

The trained deployment must be the same pinned base/dtype plus an owned SFT adapter, on the same
serving endpoint. The plan freezes four executable packages and a randomized order within each
repetition. All four are validated before any submission. A same-value or cancelling environment change is rejected. An enabled monitor needs a separate fixed alias outside the base/trained policy aliases. Task/verifier bytes, rubric, agent count,
policy sampling/context/output limits and model budgets stay fixed. Environment factors cannot
rewrite tasks or provide prompt-based substitutes. Declared training task IDs are checked for overlap;
semantic/content disjointness across differently named tasks remains **unverified**.

An external controller or researcher advances the durable plan:

```python
status = await workbench.advance_factorial(study["study_id"], WORKER_URL)
report = workbench.report_factorial(study["study_id"])
```

Each advance reconciles the active trial and submits at most one new trial. The worker continues
accepted work after client disconnection. Call again to progress the fixed schedule; this tool is
not a background autonomous search agent. Reconnection reuses the same child job ID. Concurrent
advancement is locked. An uncertain submission is inspected at that identity; it is not blindly
retried. Explicit same-ID reconciliation remains available through `get_job_status(action="resubmit")`,
including a client that died during preflight. Request and owned job creation is atomic. An interrupted
submission is never replaced with a new trial identity.

Every scheduled trial remains in the denominator. Absent, unknown, failed, quarantined and unrun
trials are distinct. Valid task failures/zero scores are not replaced. Cleanup failure, serving
preflight failure or rejected runtime lineage halts scheduling and preserves remaining unrun trials.
A new generation after repair must be declared explicitly.

Reports include original measurements, observed model usage, four cell counts and descriptive
presence-difference/interaction bounds. These bounds account for unknown observations; they are
**not confidence intervals**. Repeated runs on one frozen task set do not become independent task
samples. Study agent quotas exclude the one-token control-plane serving probes and are not a cloud
invoice or GPU-allocation budget. Task/runtime deadlines remain the existing native runner's contract.

Automatic hypothesis search, ablation selection, arbitrary remote adapter transfer, numerical GPU
weight verification and claims of behavioral improvement remain outside this implementation.

## Verification

See `tests/test_reproduction_deployment.py` for a real tiny local SFT update connected to a simulated
HTTP serving lifecycle, including a lost load acknowledgement and read-only reconciliation.
`tests/test_serving_identity.py` checks actual gateway transport behavior without changing sampling.
`tests/test_reproduction_runtime.py` exercises bundle reconciliation, original zero scores, rejected
model/record/config/verifier mismatches, four-cell planning and duplicate-submission prevention.
These fixtures are software checks, not new benchmark data or native multi-agent acceptance.

No GPU allocation, model download, Docker experiment or new real-agent behavior measurement was
performed for this change. The operator confirmed that a live server is not yet available.

Verification on 2026-11-06, based on `4bfe55e`:

- Full repository suite: **1,754 passed**, four existing staged-scenario deprecation warnings.
- Final model-list routing correction and CLI wording: **289 affected tests passed**, including
  the additional deployment-server model-list regression. These checks cover the gateway, RL gateway,
  sampling, workbench execution/lineage and Docker worker contracts.
- Final Ruff check passed; Mypy passed for **117 source files**.
- Read-only reviews checked serving activation, runtime lineage and factorial crash/failure handling;
  their findings were repaired and covered by the local regressions.
