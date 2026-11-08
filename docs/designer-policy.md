# Designer policy selection and real training

Use this optional mode when the designer may change P. The broader
[P/S/E workflow design](swarmstill-pse-design.md) separates this implemented policy
surface from additional proposed execution, topology and training backends.

The existing six evidence/spec tools remain. A private `policy_space` file adds:

| Tool | Effect |
|---|---|
| `mesocosm_policy_catalog` | Page approved policy handles, compatibility groups and training recipes/domains. |
| `mesocosm_train_policy` | Reserve the session budget and queue a real local SFT/LoRA job. |
| `mesocosm_policy_status` | Inspect, cooperatively cancel, or explicitly start a session-owned queued training attempt. |
| `mesocosm_activate_policy` | Deploy its completed owned artifact, or explicitly reconcile an uncertain activation. |

## Trusted preparation

Keep settings, source datasets, local model files, workbench and credentials outside the
designer sandbox. Prepare datasets using native `prepare_dataset`; approve source provenance
and preserve its group-disjoint partitions. Its internal test partition is excluded by the
native trainer and no test examples or results are returned to the designer.

Create existing native deployments with `deploy_policy` when policy selection is needed.
Use a public handle for each approved base/checkpoint/adapter. The public handle is the
operator's deliberate description; never put a secret, path or hidden condition in it.

Example private policy-space file (replace identity placeholders with real immutable IDs
and hashes; this is not an executable student benchmark):

```json
{
  "deployments": {
    "baseline-4b": "deployment-<immutable deployment id>",
    "adapter-a": "deployment-<immutable adapter deployment id>"
  },
  "recipes": {
    "demonstration-adaptation": {
      "dataset_id": "dataset-<immutable prepared dataset id>",
      "config": {
        "base_model": "/operator/pinned-local-model",
        "base_sha256": "<64 lowercase hexadecimal characters>",
        "device": "cpu",
        "dtype": "float32",
        "max_steps": 20,
        "max_seconds": 120,
        "max_train_tokens": 50000,
        "max_eval_tokens": 10000,
        "target_modules": ["q_proj", "v_proj"]
      },
      "parameter_choices": {
        "learning_rate": [0.0001, 0.0003],
        "lora_rank": [4, 8],
        "lora_alpha": [8, 16]
      }
    }
  },
  "max_training_jobs": 2,
  "max_total_steps": 40,
  "max_total_seconds": 240,
  "max_total_train_tokens": 100000,
  "max_total_eval_tokens": 20000
}
```

This example permits CPU training without a serving venue. For a recipe that can activate,
use matching bfloat16 training and add its private venue:

```json
{
  "server": "https://approved-model-server.example",
  "token_env": "APPROVED_MODEL_TOKEN",
  "base_model": "organization/pinned-base",
  "base_revision": "<40 lowercase hexadecimal characters>",
  "base_sha256": "<same base checkpoint hash as training>",
  "dtype": "bfloat16"
}
```

Put that object in the recipe's `venue`. The local model/adapter filesystem must be shared
with vLLM. There is no implicit float32-to-bfloat16 conversion or weight download. The
operator fixes device, precision, tokenizer, target modules, data and hard compute ceilings.
Only the published finite optimization choices can be changed by the designer. Current
training is SFT with LoRA; full fine-tuning, PPO and GRPO are unsupported by this interface.
Recipes start from materialized local bases. This restricted surface does not expose native
resume or arbitrary parent-adapter continuation; those require an explicit owned checkpoint
and budget-preserving training contract.

Add `"policy_space": "/operator/policy-space.json"` to the private settings from
[the stdio setup guide](designer-mcp.md), then launch the same `designer serve` command.
Omitting it retains the fixed-P mode. Changing the policy-space bytes requires a new session.

## Designer flow

Discover capabilities and page `mesocosm_policy_catalog`. One policy handle applies to every
agent with `policy_id`, or assign every public ordinal slot with `agent_policies`:

```json
{
  "changes": [],
  "agent_policies": {
    "agent-1": "adapter-a",
    "agent-2": "baseline-4b",
    "agent-3": "baseline-4b"
  }
}
```

Without a [structure space](designer-structure.md), slots follow the operator-fixed world's agent order. Equal handles share the same
policy weights, while harness sessions and contexts remain separate. Selective updates
are explicit: train a new handle, then assign it only to the intended slots. Updating a
handle never silently changes all agents that previously used its parent policy.
Mixed assignments must share endpoint, base name/revision/hash and dtype. A uniform
selection can choose another approved base; cross-base mixtures remain unsupported.
Changing population requires the separate approved structure space.

For an actual update, call `mesocosm_train_policy` with a published `recipe_id`, optional
exact `parameters` and a stable `request_id`. Reusing that ID recovers the same attempt;
changing its recipe/parameters is rejected. A new request reserves another attempt and
its full step/token/time ceilings. Failed and interrupted attempts retain reservations.
Poll the returned public `policy_job_id` with `mesocosm_policy_status`; `action=cancel`
requests cooperative cancellation. Do not repeatedly create jobs to poll progress.

After completed training, explicitly call `mesocosm_activate_policy`. It verifies the
committed optimizer/checkpoint provenance and uses native serving registration/probe.
Only ready activation yields a selectable policy handle. If the load is uncertain,
inspect its state and explicitly request `reconcile=true`; reconciliation verifies the
same alias and does not reload it. A recipe without a venue remains training-only.

Check the proposed P plus S/E changes with `mesocosm_check_spec`, then freeze one final
with `mesocosm_submit_spec`. This validates the exact packaged deployment map. It does
not execute student tasks or expose the private evaluator. New train/activate operations
close after final submission; owned status/cancellation remain available.

## Recovery

A supervisor-launch failure does not require a new training request. The response separates
native job status from `worker_start: pending | started | failed`. `started` means supervisor
launch returned; optimizer progress still comes from the native measured result. Repeat the
exact `mesocosm_train_policy` request after a known launch failure, or call
`mesocosm_policy_status` with the returned public `policy_job_id` and `action="start"`.
Both use the same admitted native job and existing budget reservation. Explicit start also
recovers a queued job after a crash between native persistence and request acknowledgement.
Running and terminal attempts are not rerun. If native admission remains unresolved, no
replacement job is created; the operator must inspect it. Starting the supervisor does not
resume interrupted optimizer state or allocate another machine.

Activation distinguishes pre-admission failure from an uncertain load. If no native request
or owned deployment exists, status is `activation_not_started`: repair the transient cause
and retry normal `mesocosm_activate_policy` with the same public handle. The native key and
alias remain identical. `reconcile=true` never starts a new load and reports that there is
nothing admitted to reconcile in this state. Once any native admission exists, an ordinary
retry is refused with `activation_uncertain`; explicit reconciliation verifies the same
owned job and alias without reloading it. A busy serving lease or ambiguous admission stays
uncertain. Reservations are retained through every failure, and new request IDs cannot be
used to evade the session budget. [ADR-0071](adr/0071-recover-owned-policy-start-and-activation.md)
records these recovery boundaries.

## Evidence and limits

The catalog, source configuration and final resolution are session-bound. Native job IDs,
checkpoint/model paths, credentials, raw diagnostics and test measurements stay private.
The existing [corpus review and OS isolation boundary](designer-mcp.md) still applies.

Software contracts: [policy lifecycle](../tests/test_designer_policy.py),
[native binding and stdio integration](../tests/test_designer_mutable.py), and
[legacy fixed mode](../tests/test_designer_mcp.py). An actual tiny CPU update establishes
optimizer execution only. Live student LoRA serving, multi-agent outcomes and responses to
unseen interventions need separate execution evidence. The original benchmark contract
blocker documented in [the audit](reward-hack-benchmark-contract.md) remains unresolved.
