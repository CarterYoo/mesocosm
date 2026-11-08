# Restricted designer interface

Without a policy-space configuration, the local designer MCP exposes exactly six tools:
`mesocosm_capabilities`, `mesocosm_corpus_search`, `mesocosm_corpus_stats`,
`mesocosm_read_events`, `mesocosm_check_spec`, `mesocosm_submit_spec`.
Install its optional transport dependency with `uv sync --extra mcp`.

To let a designer change P as well as S/E, add the optional private `policy_space`
configuration described in [policy selection and real training](designer-policy.md).
That mode adds four bounded tools for policy discovery, actual SFT/LoRA, status and
verified activation. The fixed-P settings below remain valid.

The following preparation belongs in the trusted operator context. None of these
configuration files, private stores, source bundles or commands belongs in a designer
sandbox. Keep the workbench and server outside that sandbox.

## Prepare the corpus

Inventory a teacher run bundle without publishing anything:

```bash
uv run mesocosm designer mask-corpus /operator/teacher-run \
  --store /operator/masked-corpus --source-format bundle --review --json
```

Inspect the original event/native text. Create a JSON policy with the inventory's
`source_sha256`, a nonempty `reviewer`, and an `events` mapping from each explicitly
approved event pointer to its approved string-field JSON pointers. For example:

```json
{
  "source_sha256": "<hash from private inventory>",
  "reviewer": "operator-reviewer",
  "events": {"/events/3": ["/data/text"]},
  "redact_literals": ["<specific identifying value>"],
  "forbidden_literals": ["<value that must not survive release>"]
}
```

Review must remove condition explanations, system/developer instructions, reward
explanations, model/checkpoint and population identities, evaluator details, operator
paths and held-out settings embedded in otherwise ordinary messages. The builder
excludes private fields/roles and reward/context events, but automatic field removal
and literal replacement cannot establish semantic decontamination. It rejects an
unreviewed, changed or ambiguously selected source. Keep the policy and review inventory
private. Import the approved projection:

```bash
uv run mesocosm designer mask-corpus /operator/teacher-run \
  --store /operator/masked-corpus --source-format bundle \
  --policy /operator/review-policy.json --json
```

`bundle` reads normalized records and verified native turn files through the existing
evidence importer; `normalized` accepts a JSON events list/object and `jsonl` accepts
a journal. Public event/corpus IDs are stable within the immutable projection. Private
source snapshots and citation mappings, review policy, source and projection hashes remain under
the private store's `operator` directory. Only its `public` directory is a corpus input
for the designer server.

Known `split: test` or `partition: test` declarations reject the entire source before
projection or writes, even when a hash-correct operator review selects only other events.
The check follows recognized metadata containers and event declarations, including the
original bundle summary, lineage and native-message metadata. It does not search dialogue
text or establish the safety of unknown metadata or undeclared partitions.

## Freeze the design space and start stdio

Write an operator allowlist with complete finite values. Its values are deliberate
public study design choices, not a dump of current hidden world values:

```json
{
  "factors": {
    "services.board": [false, true],
    "services.chat.enabled": [false, true],
    "services.chat.channels": [[], ["general"]],
    "rewards.allocation": ["individual", "team_mean", "mixed"],
    "rewards.team_weight": [0.25, 0.5, 0.75],
    "task_services": ["isolated", "shared"]
  }
}
```

The supported name boundary covers board/chat/channels, artifacts, peer review,
reputation, reward allocation/team weight, history, context feedback, task assignment,
git/cache, agent isolation and task services. The operator may narrow this set. Model,
agent identity/model, persona, skills, MCP, tasks, wiki documents, internet, judge,
observation, budgets, seeded cache text and arbitrary text remain locked. Names or
paths within an allowed value must equal a complete approved choice. Native dependency
rules still apply; an individually allowed value can fail against the fixed base world.

Create private settings pointing to an existing workbench behavior definition and
operator-chosen base world:

```json
{
  "store": "/operator/designer-session",
  "workbench_root": "/operator/research-workbench",
  "base_world": "/operator/fixed-world.yaml",
  "behavior_id": "behavior-<private object hash>",
  "allowlist": "/operator/design-space.json",
  "corpus_root": "/operator/masked-corpus/public",
  "corpus_ids": ["corpus-<public corpus id>"],
  "session_id": "independent-designer-01"
}
```

```bash
uv run mesocosm designer serve --config /operator/designer-settings.json
```

Stdout is exclusively MCP JSON-RPC. The server offers no HTTP transport. The six tools
never return the private settings, full native errors, store paths, deployment/model
references, evaluator specs, test split or held-out settings. Calls and full native
validation diagnostics are retained privately. A spec has only this shape:

```json
{"changes": [{"factor": "services.board", "value": false}]}
```

Capabilities pages return the allowlist and the public size-class/fixed-P description.
Where applicable, they also return general factor dependencies, such as nondefault chat
limits requiring enabled chat. A native-validation rejection names the changed field,
its exact allowed values and any related factors adjustable under the allowlist. Recheck
an approved combination or remove the change. These hints describe public software
contracts; they do not reveal current settings or identify which hidden setting failed.
Corpus pages return reviewed text with event IDs; use `next_offset` and, for long text,
`next_text_offset` to continue. Each response is at most 24,000 UTF-8 bytes. A successful
check establishes package/configuration feasibility, not serving, sandbox readiness or
behavioral reproduction. Submission freezes exactly one final spec for the operator's
session; repeat the identical spec to recover its hash. A different final is rejected.
The private session binding includes the exact base-world and allowlist file hashes.
Changing either file blocks checks and submissions, including after restart with the
same session ID. Restore the original setup or choose a new session. The private final
references its successfully validated immutable candidate; that reference is never
returned to the designer.

Before a real designer run, follow the [inverse-study isolation procedure](../skills/swarmstill/references/inverse-study.md):
fresh model context, separate OS sandbox, real filesystem/network boundary probes and
only intended public tools. Empty model context alone is insufficient. The tests in
`tests/test_designer_mcp.py` and `tests/test_designer_corpus.py` verify software contracts;
they do not establish actual blinded-agent acceptance or behavioral transfer.
