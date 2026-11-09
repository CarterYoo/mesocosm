# Designer development execution

The optional execution profile lets a designer prepare a validated P/S/E candidate, run bounded
development trials on an existing worker and compare their compact measurements. It uses the
native operator workbench and original task/verifier lifecycle. It does not allocate servers or
provide a substitute task, model response or evaluator when a backend is unavailable.

## Operator preparation

Prepare a compatible Linux Docker worker and its model deployments using the
[execution guide](execution.md) and [policy guide](https://github.com/khs1010/mesocosm/blob/b6652daab8ab868e8f76eca045233a1f93540150/docs/designer-policy.md). Keep its credentials,
native store and private settings outside the designer sandbox. An approved worker URL uses
HTTPS, or loopback HTTP for local operator use; the public designer never selects this endpoint.

Select an approved ready policy for every slot through the optional policy profile before
preparing execution. Omitting P retains the base manifest for structural checks, but does not
create verified serving attribution. An unbound candidate is rejected with
`serving_policy_unbound` before reserving or submitting any trial. Fixed-P sessions without
approved deployment bindings can still check and submit structural designs; their development
execution is unavailable.

The fixed world must have materialized original tasks, no task stream and finite positive
`[agent].timeout_sec` in each original task's configuration. Every agent also needs finite
`budget.requests` and `budget.completion_tokens`. Do not rewrite an external benchmark's
timeout to pass this check: a missing native bound is an explicit unsupported configuration.
The graph execution profile may require further native conditions; validation applies those
actual contracts before admission.

Example private execution-space JSON:

```json
{
  "worker": "https://approved-worker.example",
  "token_env": "APPROVED_WORKER_TOKEN",
  "max_jobs": 6,
  "max_repetitions": 3,
  "max_total_agent_seconds": 3600.0,
  "max_total_requests": 1200,
  "max_total_completion_tokens": 1000000
}
```

`max_jobs` bounds planned trials across the session, including repeated, failed and cancelled
trials. The other limits reserve the sum of declared upper bounds across all plans. A cancelled
trial does not free reservation space. The credential field names an environment variable;
never put a token value in this file. Unknown fields are rejected.

Add `"execution_space": "/operator/execution-space.json"` to the private designer settings and
launch the existing [`mesocosm designer serve`](designer-mcp.md) command. The settings-file
bytes are session-bound. Changing an approval or ceiling requires a new session. Omitting this
file preserves the evidence/spec-only profile, including optional mutable P.

**This is not a hard world wall-clock budget.** Original agent timeouts bound agent turns;
builds, setup, original verification and cleanup are outside those timeouts. The capability
response explicitly marks a hard wall-clock limit unsupported. There is no client-side timer
that pretends to stop a disconnected worker.

## Designer tools and lifecycle

| Tool | Input and effect |
|---|---|
| `mesocosm_prepare_candidate` | Supply an approved `spec`. Reuse real configure/validate and freeze an opaque development `candidate_id`, its spec hash, grading scope and per-trial reservation. |
| `mesocosm_run_candidate` | Supply that `candidate_id`, stable `request_id` and bounded `repetitions` (default 1 for a smoke). Reserve the whole plan and submit its first trial. |
| `mesocosm_run_status` | Supply a returned `run_id`; inspect by default, or explicitly `advance`, `fetch`, `cancel` or `resubmit`. |
| `mesocosm_compare_runs` | Supply owned `baseline_run_id` and `candidate_run_id`. Check frozen comparison identities and return compact descriptive measurements with failed/unrun trials retained. |

Discover the advertised tools and execution bounds through `mesocosm_capabilities`. Prepare
the unchanged baseline spec and a proposed spec separately. Neither preparation nor final
submission is a student run. A smoke is an ordinary one-trial native execution with the same
original task and verifier; it is not an authored quick task or shortened evaluator.

`run_candidate` submits once. Repeating the same request ID and identical inputs returns its
existing plan. Reusing that ID with different inputs is rejected. A new request ID reserves a
new independent plan, so do not create requests to poll an existing one.

Inspect with `run_status`. The response includes planned-trial counts, trial statuses, retained
reservations, compact measurements, grading scope and `state_version`. Send that version as
`since_version` on the next inspection to receive a small unchanged envelope. An inspection
may reconcile remote status, but does not fetch results, run evaluation, resubmit a job or
launch a later trial.

After a trial becomes terminal, explicitly request `action="fetch"`. Native retrieval verifies
the retained package inventory. Completed runs then pass strict original input, serving,
runtime, transcript and verifier lineage checks before their frozen detector is evaluated.
Unqualified results are quarantined, with full diagnostics kept private. Failed and missing
trials remain unknown measurements. The public tool never falls back to an imported transcript
when native qualification fails.

After a fetched/evaluated or failed trial, explicitly request `action="advance"` to submit at
most one next planned trial. Later trials are never selected or replaced based on the desired
behavior. An integrity rejection or failed cleanup halts admission and requires operator review.

Use `action="cancel"` to stop the owned active job and mark pending trials unrun/cancelled.
Keep inspecting until the worker's terminal state is known; a cancellation request alone is
not proof of cleanup. Use `action="resubmit"` only for uncertain admission or a failure before
native job creation. It retains the same native identity and reservation. Inspecting or
repeating the original run request does not silently retry. A server/client restart can recover
durable jobs, but does not restore an interrupted harness session or replay model calls.

Final submission still freezes one immutable design. Existing status, cancellation, fetch and
comparison remain available afterward; preparation or admission of new trials closes with the
session. A final spec is not automatically promoted because a development outcome looked good.

## Comparison and evidence

Both conditions must retain identical original task trees/verifiers, frozen detector, judge and
observation settings, installed provider code identities and per-slot/team model budgets. The
current comparison requires the same number of slots; changing N requires a separately approved
budget/measurement contract. Approved P and S/E differences remain visible in their private
frozen configurations rather than being removed by a whole-manifest equality check.

Each run's original result inventory, applied P/S/E, checkpoint/deployment attribution,
evaluation and reproducer reference are linked in private operator records. The public response
contains opaque owned handles and bounded counts, not paths, model names, native job/run IDs,
evaluator rules, arbitrary log text or test/held-out measurements. To inspect trajectories as a
designer, the operator must release an explicitly reviewed and masked development corpus using
the existing corpus workflow.

Graph workers that are not finalizers have no original verifier score. Public grading scope
reports the finalizer slots, expected verifier units, ungraded agent/task units and observed
verifier units. Different finalizer scopes are labeled **partial** and do not yield a
cross-condition presence-difference bound. An ungraded worker is never a failed task with score
zero. Missing, failed, cancelled, quarantined and unrun trials remain in planned denominators.

Same-scope presence bounds describe the range allowed by unknown trials. Collective metrics
report observed event denominators and unknown-trial counts separately; an unrun trial does not
invent a number of review events. These are descriptive comparisons of one frozen task set,
not confidence intervals, independent task samples, causal estimates or evidence of improvement.

## Verified scope

[Software contracts](../tests/test_designer_execution.py) exercise native configuration and
validation, transport recovery, reservation bounds, strict fetched original-bundle lineage,
private exports and tamper rejection. The worker/transport fixtures are labeled software
fixtures. They are not actual model/agent evaluation. Live serving and at least three real
harness sessions with original task actions/verifiers still need acceptance evidence.
The [original benchmark contract blocker](reward-hack-benchmark-contract.md) remains unresolved;
this façade does not remove it. See [ADR-0073](adr/0073-bounded-designer-development-execution.md).
