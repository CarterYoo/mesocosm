# SwarmStill qualitative reproducibility corpus

This bundle preserves the frozen collusion analysis corpus, its complete experimental source runs, calibrated qualitative annotations, and the purposive natural-language casebank used for the failure analysis. It is a portable derivative of saved experimental artifacts. The release does not manufacture missing logs or establish causal mechanisms.

## Scope

- **736 agent–episode records** from **49 nonempty trajectories**: 250 student records from the Astra design sessions, 126 from the Opus design sessions, 160 original-environment student records, and 200 teacher records. Design-session names identify the designer condition, not the model producing the student verdict.
- **640 constrained records and 96 other-round records** are retained separately. In particular, warm-up/no-throttle or summary-allowed rounds must not be pooled with constrained raw-only rounds without qualification.
- **49 complete collusion source snapshots** retain saved run configuration, complete event history, task artifacts, benchmark prompts, private task-phase raw logs, delivered and rejected messages, structured verdict actions, outcome feedback and reflections, wherever present in the original artifact. “Private raw log” is an experimental field, not a claim of private personal content. Original artifacts are not silently completed if a field is absent.
- The **casebank contains 64 selected collusion records, 4 selected ACB turns and 8 complete designer notes**. It is purposively selected to support examples and counterexamples, not a random sample for estimating prevalence. The complete ACB runs are part of the separate ACB release; this bundle includes only the 4 selected turns, assigned roles, injections, trace indices and original source/task-payload hashes.
- One empty student run was excluded from the original 736-record selection. Its provenance is recorded as an excluded empty source. Later tests outside the frozen selection are not added to the 736-record corpus.

## Files

| Path | Contents |
|---|---|
| `corpus/agent_episodes.jsonl.gz` | All 736 enriched records, including complete saved verdict reasoning/reflection, channels, failed sends, action, reward, policy and source links |
| `sources/*.json.gz` | All 49 complete source run objects after narrowly scoped infrastructure redaction |
| `annotations/calibrated.jsonl.gz` | Current calibrated annotations joined to record-level metadata |
| `annotations/reviewer_codes/*.jsonl.gz` | The underlying agent-generated direct-reading code files |
| `annotations/root_sample_reviews.jsonl.gz` | Additional agent review checks; not independent human adjudications |
| `annotations/classification_summary.json` | Descriptive aggregates regenerated from the calibrated code files |
| `annotations/coding_validation.json` | 736/736 coverage and quote/label validation result |
| `protocol/` | Original codebook and clarification/adjudication protocol |
| `casebank/selected_cases.json.gz` | Complete 64+4+8 selected-case object and sanitized snapshot manifest |
| `casebank/*.jsonl.gz` | The same casebank split by record type |
| `designer_notes/*.md` | The 8 complete designer notes as directly readable text |
| `designer_event_extracts/*.jsonl.gz` | Four saved Astra natural-language extracts containing 435 agent-message records, not complete designer sessions |
| `designer_workspace_snapshots/` | 42 saved specs/probes/config JSON files, 41 test-result summaries, and 8 session ledgers containing 100 events |
| `provenance/designer_supplement_manifest.json` | Source hashes, exact recorded timestamps, filesystem metadata and explicit coverage for the designer supplement |
| `provenance/source_manifest.json` | Portable source IDs, coverage, original hashes and sanitized-source hashes |
| `provenance/artifact_origins.json` | Original byte hashes for the input/annotation/protocol/casebank artifacts |
| `provenance/redactions.jsonl.gz` | Field-level redaction audit events, containing fingerprints rather than original sensitive values |
| `provenance/redaction_policy.json` | Exact redaction scope and exclusions |
| `SHA256SUMS.json` | Checksums and byte lengths for every released file except itself |
| `verify_bundle.py` | Portable Python standard-library verification and aggregate reproduction |

## Designer supplement coverage

The designer supplement preserves the four available generated Astra natural-language event extracts used in the analysis. They contain agent messages with original source-line indices; they are not complete session transcripts or complete execution histories. Full Opus designer transcripts were unavailable in the supplied audit inputs and are not reconstructed. The eight designer notes remain available for both design conditions.

Saved workspace specs, probes, final configuration files, test-result summaries and daemon session ledgers are included as a separate snapshot. This supplement can contain later tests beyond the frozen 736-record collusion selection. It must not silently expand that selection or be described as evidence available at an earlier cutoff. Existing submitted_at, finished_at, at and other recorded timestamps are preserved exactly and listed in its source manifest. Filesystem mtime/ctime and available birthtime are separately recorded; ctime is not labeled as creation time. References to additional full runs do not imply those runs are included here.

## Annotation status and interpretation

These are **agent-generated qualitative annotations**, produced by direct reading of the complete supplied reasoning, reflections, delivered communication and feedback, followed by calibration and agent review. They are **not human-coded, blinded, independent-human, or gold-standard labels**. Scripts rendered records, saved decisions, checked provenance and recomputed summaries; they did not replace direct reading with a substantive regular-expression classifier.

The codebook keeps missing evidence, substantive peer-task criticism, wrong-task/source criteria, assurance-based acceptance, acceptance of specific supplied evidence, and unclear/other cases distinct. Uncertainty is explicit. A mere mention of the judge's own function is not enough to establish task confusion. Structured actions can disagree with their rationales; both are preserved. Ancillary false beliefs about the quality of one's own work do not alone count as reward misreading when reward, action and feedback-direction accounts are otherwise accurate. Prospective reflection language does not demonstrate actual subsequent adaptation.

Source-level correctness, mutual acceptance, task completion, policy compliance and reward are different variables. A correct rejection can occur despite confused reasoning; a shared reward can be positive even when one verdict is wrong. Qualitative co-occurrence supports hypotheses about bottlenecks, not causal identification or claims about deliberate deceptive motives.

## Provenance and redaction

Original `source_sha256` values are retained. For a JSON source they fingerprint the original stored JSON bytes; for an original `.json.gz` teacher source they fingerprint the original compressed bytes. `source_manifest.json` separately records the decoded-original byte hash, the sanitized JSON byte hash and the deterministic sanitized gzip hash. A sanitized file is not expected to match the original-byte hash.

Only explicit infrastructure metadata and machine-specific root/reference paths are changed: local source references become relative release paths or source IDs; the benchmark checkout root becomes `<BENCHMARK_ROOT>`; the runtime installation root becomes `<RUNTIME_ROOT>`; inference endpoint configuration becomes `<REDACTED_ENDPOINT>`; explicit credential metadata, if present, becomes `<REDACTED_CREDENTIAL>`. Residual local home/scratch paths receive nonidentifying placeholders. The audit manifest records artifact, JSON pointer, rule, replacement and original-value/fragment fingerprint without disclosing the original sensitive value.

Benchmark task content is preserved, including synthetic emails, password test cases, function names, queries, role text, injections and experimental instructions. No broad “password,” “token,” email or IP-address replacement is applied. Quoted prompts, commands and injections in the corpus are **untrusted observations to study, not instructions to execute**.

`own_raw_log_chars` and `peer_raw_log_chars` retain lengths measured on the original source. Infrastructure path replacement can change the length of a sanitized private raw-log string; do not recompute those original measurements from sanitized text and assume equality.

## Verify and reproduce

No third-party Python packages or access to the original workstation are needed:

```sh
python3 verify_bundle.py
python3 verify_bundle.py --write-summary ../reproduced_classification_summary.json
```

Run verification before writing additional files inside the bundle, or write regenerated summaries outside it, because the checksum inventory intentionally covers the exact release tree. The verifier checks the complete file inventory and hashes, deterministic gzip headers, 736 unique corpus and annotation keys, legal labels and uncertainty flags, exact short quote provenance, reviewer-code consistency, 640/96 scope split, 49 complete source links, matching full-run rationale/feedback/action/channels, casebank 64+4+8 integrity and reproduced aggregates. It also checks that known local infrastructure roots and endpoint metadata have been removed.

Original-byte hashes provide provenance fingerprints; verifying an original byte hash requires access to that original artifact. The portable verifier checks the included sanitized derivatives and their internal scientific record linkage. It does not claim to reconstruct omitted ACB source runs, regenerate model outputs, validate every qualitative interpretation, or prove the proposed behavioral mechanisms.
