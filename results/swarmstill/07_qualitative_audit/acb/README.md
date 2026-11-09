# SwarmStill: generated AgentCollabBench response logs and offline audit

This release contains the full available **model-generated response text** for the six AgentCollabBench designer sessions in the latest eight-session SwarmStill design study, together with the original student and available original teacher traces. The two collusion sessions are published separately.

No new model calls were made to prepare or validate this release. It does not contain model weights, complete third-party benchmark task definitions, third-party evaluator source, or operational host logs. A limited scan and manual review found no observed operational credential in the published material; the review and its limits are documented below.

## Coverage and provenance

- 610 task traces, containing 13,200 generated agent turns.
- Original student: RTD 18, CPR 20, IDR 20 tasks.
- Original teacher traces available locally: RTD **3**, CPR 20, IDR 20. The saved RTD target score table has 18 tasks, but the other 15 original teacher conversations are not present. Missing conversations are not reconstructed or replaced with later runs.
- Designer responses: all available t01–t05 traces in the six sessions. CPR Astra has three completed tests and two failed attempts with no responses; CPR Opus has four completed tests. RTD Opus t04 failed and left seven partial task traces. Those partial traces remain explicitly labeled `platform_failure` and are not treated as a completed experiment.
- Later base-checkpoint q4 repeats and later q8 rechecks are excluded. They are not relabeled as repeats of the design student's checkpoint.

`manifest.json` inventories every trace with the SHA-256 of the original saved record, the saved full task definition, the acquired upstream task, and the published gzip bytes. `tests.json` records actual submitted/finished timestamps, effective intervention specifications, failures, available traces and submitted final selections. It allows later tests to be separated from a paper's earlier cutoff.

Each `responses/<metric>/<session>/<test>/<task_id>.json.gz` contains the complete available generated turns, trace indices, agent IDs and roles, per-call model aliases and token counts, final output, structural topology and intervention configuration. Recorded handoffs reference generated turn content hashes; identical repeated texts can have multiple source candidates. Synthetic peer-pressure text is omitted and represented by hashes/schedule metadata because it is benchmark input rather than a generated response.

`source_record_sha256` refers to the private original input bytes, not the transformed public record. `original_content_sha256` and `published_content_sha256` make any text replacement explicit. Current generated turn content required no local identifier replacements. Ordinary task passwords, example.com addresses, numerical thresholds and test inputs have been preserved. The publication changes representation, removes benchmark input/duplicate trace fields, and adds provenance; it does not claim the public JSON is byte-identical to a raw run file.

## Read a trace

```sh
gzip -dc responses/idr/student_original/run1/TASK-DATAENG-IDR-159.json.gz
```

Task IDs are acquisition keys, not proof that two runs had identical input. Compare `saved_task_sha256` when making a paired interpretation. Structural overrides and intervention specifications are preserved separately.

## Validate without a benchmark checkout

Python 3.12 or newer is sufficient. No installed Python packages, API keys, GPU or network service is needed:

```sh
python3.12 -S scripts/audit_offline.py
```

This verifies record uniqueness, file/content hashes, manifest completeness, deterministic gzip timestamps, links from handoffs to generated turns, and that the structured public record omits full task definitions.

## Reproduce the original-runner request audit

Acquire the upstream source separately while online, then run the audit offline:

```sh
git clone https://github.com/aritra741/AgentCollabBench.git /path/to/AgentCollabBench
git -C /path/to/AgentCollabBench checkout f016f600568b6d8127dc861e4c83c87b72750d63
python3.12 -S scripts/audit_offline.py \
  --acb-source /path/to/AgentCollabBench \
  --output audit/reproduced.json
```

The script verifies the original evaluator file hashes, reconstructs each baseline task from the pinned upstream task plus the study's saved transformation, and verifies its full task hash against the saved trace. It then executes the original runner with an **offline replay provider** that returns the published saved model outputs. The runner's reconstructed message lists are hashed; no model provider is instantiated and no completion or judge request is sent. These are reconstructed request comparisons from saved outputs and the verified runner, not historical server-side request captures.

Saved study-owned code in `source/` contains the evaluator transformation and the exact domain-validation excerpt. The original AgentCollabBench runner and dependencies are acquired from upstream rather than vendored. `audit/offline_results.json` records the verified result:

- `turns=2` changes the effective speaking order in RTD 8/18, CPR 0/20 and IDR 0/20 tasks.
- CPR and IDR each produce identical reconstructed request lists in all 20 tasks under original versus x2 speaking-order settings.
- IDR `all`/`half`/`first` pressure-message settings produce identical reconstructed agent requests in all 20 tasks while changing synthetic trace entries.
- Advertised fractional turn values 0.5 and 1.5 fail the saved designer-domain type check.

The offline replay reproduces execution-path facts. It does not resample model behavior, estimate a treatment effect, validate a judge, prove a runtime checkpoint identity, or show that a designer diagnosed the bottleneck.

## Evaluation and model configuration limits

Scores are saved historical values, not new judgments. `references.json` preserves the teacher/student reference tables. The available student design records do not include the remote judge's turn-by-turn verdicts, so this release cannot reconstruct missing judge rationales from scalar scores.

The study setup selects Qwen3-4B-Instruct-2507 for the design student (`qwen3-4b` alias) and uses the earlier Qwen3-8B teacher reference (`q8` alias), with different documented serving choices. Model aliases and source configurations do not independently attest the exact live checkpoint or KV-cache dtype used in every historical call. See the study setup scripts and the accompanying audit discussion.

## Attribution and task acquisition

Benchmark: [AgentCollabBench](https://github.com/aritra741/AgentCollabBench), exact acquisition revision [`f016f600568b6d8127dc861e4c83c87b72750d63`](https://github.com/aritra741/AgentCollabBench/tree/f016f600568b6d8127dc861e4c83c87b72750d63). The pinned [`pyproject.toml`](https://github.com/aritra741/AgentCollabBench/blob/f016f600568b6d8127dc861e4c83c87b72750d63/pyproject.toml) declares package license `MIT`; no separate LICENSE file or dataset-specific license notice was found at that revision. This release does not infer a separate dataset license or bundle the full task corpus.

Generated responses can quote task material as part of what a model said; those quotations are preserved as response evidence and attributed to the benchmark. They are not presented as our newly authored benchmark tasks. The benchmark's input definitions and original evaluator code are obtained through the acquisition instructions above.

## Build the release from private saved evidence

`scripts/build_release.py --help` lists explicit source inputs. The builder preserves generated response text, strips complete input definitions, records original/published hashes, and uses gzip mtime=0 for deterministic per-trace files. Paths in its arguments are operator-local and are not embedded in public metadata. Rebuilding requires access to the private saved source records; replaying the published structural audit does not.

`scan_summary.json` records a credential-shaped pattern flag without exposing the matched value. Its single flag was manually reviewed as a short **model-generated example key ID** in an API gateway design response, not an observed operational credential. The adjudication is recorded in `scan_review.json`; the example remains in the generated evidence. The scan is a limited artifact check, not a claim that a generic regex establishes privacy or safety.
