# Trusted-operator CLI

Use this reference for evidence analysis, factor discovery, candidate preparation and
operations explicitly requested by the researcher. It describes the existing public
workbench, not the isolated six-tool inverse interface.

## Discover contracts, then dispatch

From a Mesocosm checkout, `uv run mesocosm research tools --json` returns the concise tool
index without starting jobs. With an existing installed executable, omit `uv run`.
Inspect one nested contract through `describe_tool` before composing an unfamiliar request.

Each call reads one JSON argument object and emits one JSON envelope. Keep requests and
the private durable store in a researcher-owned working directory, outside designer mounts.
For example, with this request saved as `describe-factors.json`:

```json
{"name":"list_factors"}
```

```bash
uv run mesocosm research call describe_tool \
  --request describe-factors.json --root runs/swarmstill --json
```

After confirming the installed contract, a focused discovery request can be:

```json
{"prefix":"history.","limit":10}
```

```bash
uv run mesocosm research call list_factors \
  --request history-factors.json --root runs/swarmstill --json
```

Save that second JSON object as `history-factors.json` before the call. Inspect both the
process exit status and envelope `ok`. Correct a rejected argument from its contract;
do not blindly retry or bypass validation. Use the same store root for a continuing study.

## Select the next operation

| Need | Existing tools to discover |
|---|---|
| Import researcher-approved evidence and inspect exact records | `inspect_transcript`, `read_events`, `read_record` |
| Freeze a question and prepare a candidate | `define_behavior`, `list_factors`, `configure_experiment`, `validate_experiment` |
| Submit an approved candidate and recover its state | `run_experiment`, `get_job_status`, `list_records` |
| Measure qualified observations and compare candidates | `evaluate_run`, `evaluate_candidate`, `compare_candidates` |
| Export the preserved inputs and evidence | `export_reproducer` |

These are names to look up, not prefilled requests. Required arguments and state-dependent
constraints come from `describe_tool`. Use the upstream
[workbench guide](https://github.com/GRAIL-center/mesocosm/blob/main/docs/reproduction.md)
for the selected operation. Read training or policy deployment documentation only when
the user requests that work; do not make it a prerequisite for ordinary analysis.

## Bounded navigation and evidence

Filter factors by prefix or names and request `detail:true` only for relevant fields.
Filter events by agent, kind or event ID. Continue using returned cursors with the same
filters. If an item is omitted or returned as a reference, use the indicated record/pointer
or string offset to read its exact content. Do not substitute an invented summary.

The dispatcher has a 24 KiB JSON result ceiling before the CLI envelope. This is a byte
bound, not a billed-token guarantee. Do not reload every schema or raw trajectory on every
call. Do not expose operator `read_record`, private paths or `--allow-test-evaluation` to
a blinded designer.

## Preparation is not execution

Candidate validation does not prove Docker readiness, live model identity, actual task
execution, or behavioral success. Use only approved existing worker/model endpoints and
their configured credential environment names. Credentials do not belong in request files,
manifests, skill files or transcripts.

The native research worker requires at least three real agents. Preserve an original
two-agent benchmark as such; do not silently add an agent or rewrite its evaluator to make
it pass. A non-Harbor tool protocol needs a real integration preserving its semantics.
See the upstream runtime and integration guides linked from this skill's README.

Keep every accepted job's identity for reconciliation after interruption. Retain failed,
incomplete and missing observations. A software fixture can verify plumbing but cannot
establish real-agent acceptance or a measured teacher/student response.
