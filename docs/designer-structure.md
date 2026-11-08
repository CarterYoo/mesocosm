# Designer populations and enforced S graphs

Enable this optional interface when a designer may choose approved roles, population,
teams and interaction graphs. P remains the separately approved policy catalog; E changes
remain finite operator-approved values. Existing fixed-population studies retain their
previous behavior when `structure_space` is omitted.

## Operator preparation

Create a private structure-space file, then add its path as `structure_space` in the
private [designer settings](designer-mcp.md). The base world must contain every approved
role template. Template agent IDs below are private examples, not designer inputs.

```json
{
  "roles": {"worker": "original-worker", "reviewer": "original-reviewer"},
  "teams": ["team-a", "team-b"],
  "min_agents": 3,
  "max_agents": 8,
  "allowed_services": ["board", "chat", "cache"],
  "channels": ["general", "team"],
  "repositories": ["project"]
}
```

Roles and teams each have at most 16 public handles. Resource domains contain exact
approved public names; channels and repositories must already exist in the base world.
The bounds must include its population and remain within 3–32 agents. Cloneable role
templates must use native default harness URLs. Keep role instructions, task sources,
credentials, workbench roots and all other private settings outside the designer sandbox.
Use deliberate public labels that contain no hidden study condition or instruction.

The designer sees `mesocosm_structure_catalog`, never the private template IDs or contents.
The catalog reports finite domains, supported graph semantics, required conditions and
automatic prerequisites. Structure execution requires Docker isolated comparison, reset
memory, all-agent assignment, `as_ready` starts and no reward/feedback/history/reputation
episode barriers. Shared task services, shared mounts, staggered starts and side-task
verification remain unsupported. Work edges require approved declared artifact paths and
board read grants; original-score exposure and passing-only artifact eligibility are refused.
Configure these through the operator scaffold and approved E values, without changing the
original task or evaluator.

## Designer input

The following `structure` value represents two workers reporting to one reviewer. Add it
to a normal spec alongside approved `changes` and optional P selection. The operator must
have approved and enabled artifact capture for its declared original-task outputs.

```json
{
  "population": [
    {"slot": "agent-1", "role": "worker", "team": "team-a"},
    {"slot": "agent-2", "role": "worker", "team": "team-a"},
    {"slot": "agent-3", "role": "reviewer", "team": "team-a"}
  ],
  "visibility": {
    "services": {
      "board": {"read": ["agent-3"]},
      "chat": {"write": ["agent-3"]}
    },
    "channels": {
      "team": {"read": ["team:team-a"], "write": ["agent-3"]}
    }
  },
  "work": {"agent-1": ["agent-3"], "agent-2": ["agent-3"]},
  "reviews": {"agent-3": ["agent-1", "agent-2"]},
  "execution": {
    "depends_on": {"agent-3": ["agent-1", "agent-2"]},
    "finalizers": ["agent-3"]
  }
}
```

Explicit population must list `agent-1` through `agent-N` once in order. It clones only the
chosen approved role templates. Original role instructions, tools, skills, models and
per-agent budgets are retained. Duplicate persona usernames receive deterministic slot
suffixes; no new instruction is added. Omit `population` to retain the base agents; public
slots then follow their base order. Policy assignment uses these same slots.

The three graphs have separate effects:

| Input | Native enforcement |
|---|---|
| `visibility.services`, `git`, `channels`, `direct_messages` | Actual internal service permissions and message read/write visibility. Unlisted grants deny access. |
| `work` | Source-to-reader artifact ACLs, plus predecessor requirements. Ordinary board read permission is also required. |
| `reviews` | Assigned reviewer-to-subject verdict admission and completion checks. Each assigned accept/reject must exist once. |
| `execution` | Acyclic predecessor completion gates and finalizer authority. Only finalizers reach the original verifier. |

Omitting `visibility.artifacts` compiles the exact `work` graph. Supplying it requires an
identical edge set; it cannot grant extra readers. Team selectors are accepted in visibility
member lists and expand to actual population members. Empty teams and overlapping selectors
fail with actionable errors. Teams persist as role groups; they do not introduce a separate
team reward allocator. Work, review and dependency graph endpoints use individual public
slots. Reading an artifact grants access to its bytes; it does not prove the agent used it.

The compiler enables services and per-agent isolation, installs exact ACLs, and selects
the graph peer-review protocol when reviews exist. It enables chat for that protocol.
With no reviews, peer review is disabled. These prerequisites are advertised and recorded
in private before/after receipts. A competing E change cannot override managed ACL,
protocol or isolation fields. Other incompatible E choices are refused rather than reset.
Independent root stages can run concurrently; a successor starts only after all declared
predecessors complete. A failed or incomplete stage cannot satisfy its successor.

## Validation, provenance and limits

Use `mesocosm_check_spec` before submitting a final or preparing a development candidate.
Errors identify public slots, graph dependencies and approved domains; private source paths,
model identities and input dumps are not reflected. Final native validation still checks
the original installed task source and backend. A static success does not verify GPU
serving, native containment or live agents.

Private composition records retain the original base-byte hash, exact public design,
approved E changes, effective native manifest, original role instructions, role/team member
maps and a composition hash. Task and skill paths resolve against the original source
manifest before it is relocated. The compiled source has a distinct identity and native
experiment/package receipt. Runs additionally preserve stage admission/completion links,
actual review checks, source artifacts and finalizer verifier results. Subordinate work is
ungraded; missing reviews and failed predecessors remain incomplete observations.

[ADR-0072](adr/0072-native-structure-graphs-and-designer-composition.md) states the contract.
[Composer tests](../tests/test_designer_structure.py) cover public boundaries and general
N-agent compilation. Native access/scheduler/completion tests are software evidence only;
new three-agent Docker/OpenCode runs with original verifiers and trajectory inspection
are still required for live acceptance. PPO/GRPO, cross-base mixed serving, arbitrary
workflow backends and full process-state restoration are separate capabilities.
