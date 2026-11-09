# Fixed-P S/E design and controlled execution

Use the [target architecture and status](https://github.com/GRAIL-center/mesocosm/blob/main/docs/swarmstill-pse-design.md)
when designing the complete workflow. Discover capabilities from the installed backend/profile;
the design document is not evidence that every proposed field is implemented.

Fix the student model, weights, sampling and output limits within each comparison.
S chooses population, roles, visibility, review/work dependencies and scheduling.
E chooses original tasks, tools, permissions, resources, memory, feedback and rewards.
Measure behavior with the original independent scorer.

Start with a reference search and focused capability/setting queries. Compose a frozen
candidate using native types, validate cross-field dependencies and venue readiness, then
run a short original-task smoke before an approved repeated plan. Preserve every scheduled
trial and exact applied configuration. Status/cancel/recovery must use the same owned job
identity; never silently replay or call an uncaptured process restart a stateful resume.

For population counts, roles, graph schedules or tasks outside the restricted profile,
use the approved operator workflow and installed native contracts. Keep the operator
shell and workbench outside the designer boundary.

Compare candidates on the same frozen task/evaluator contract. Change only declared
intervention paths, retain checkpoint/input/provider hashes and link actual reads to source
events before claiming exposure or adoption. Missing coverage is unknown. In a blinded study,
reserve a private intervention before designer access and evaluate teacher/student response
direction separately from structural plausibility.


## Optional restricted designer workflow

Discover enabled tools at session start. `mesocosm_structure_catalog` advertises finite role,
team and visibility domains plus exact graph prerequisites. A structure spec separates
visibility permissions, source-to-reader work edges, reviewer-to-subject edges, and execution
predecessors/finalizers. Select public slots only; persona text and raw task/tool/model paths
stay locked. Teams expand actual ACL membership; team-specific rewards are unsupported.

When development execution is enabled, use `mesocosm_prepare_candidate(spec)` before
`mesocosm_run_candidate(candidate_id, request_id, repetitions)`. These invoke native
configuration/validation and the existing approved worker. Execution also requires a ready
approved policy binding for every slot; omitted P permits structural checks but cannot
prepare a lineage-qualified run. Inspect with `run_status` and
`since_version`; explicitly `fetch` completed evidence, `advance` one next trial, `cancel`, or
`resubmit` the same uncertain native identity. Do not rotate request IDs to repair a job.
Compare opaque session-owned runs with `mesocosm_compare_runs`; tasks, original verifier,
model budgets and evaluation identity must match. Retain every planned repetition, failures,
and unknowns. Graph workers completed without grading are not zero-score task failures.
Different finalizer grading scopes permit only partial comparison, not a claimed score gain.

Native graph mode currently requires isolated Docker comparison/reset/all/as-ready and rejects
reward/feedback/history/reputation barriers and shared task state. No prompt text substitutes
for missing enforcement. Whole-run hard wall-clock caps are unsupported; finite original task
agent timeouts exclude setup and verifier time. No resources are provisioned by these tools.
Read the checkout's `docs/designer-structure.md` and `docs/designer-execution.md` for configuration.
Finish development exploration before the single immutable final `submit_spec`; afterward
existing run inspection/fetch/cancel and comparison remain available, but new trials do not.
