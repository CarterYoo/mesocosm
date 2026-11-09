# Mesocosm · SwarmStill hackathon fork

## What is Mesocosm?

Mesocosm is a research platform for running AI agents together under controlled
conditions. A world defines the agents, their tasks, the tools and information
they can access, and how they receive feedback. Researchers can vary a condition,
repeat the experiment, and inspect the recorded model requests, agent actions,
and task-verifier results. For example, they can change which agents see a peer's
report while keeping the task and its verifier fixed.

## What is SwarmStill?

SwarmStill is a hackathon study of whether a frontier coding agent can design
conditions under which a team of smaller models reproduces a measured collective
behavior. The study describes behavior as **B = F(M, P, S, E)**, where **M** is
the model, **P** is its post-training, **S** is the team structure, and **E** is
the environment. Within each comparison, the student model and its weights stay
fixed. The designer changes approved features such as roles, communication
paths, memory, or feedback, depending on what the benchmark supports.

The iterative workflow gives the designer a reference behavior profile and
approved controls. The designer proposes settings, runs the student team,
reviews its scores and traces, and revises the design before submitting a final
configuration. The benchmark's scoring rules stay fixed so the final design can
be compared with the original student environment and the reference team.

- **Emergent Collusion** tracks how often two peer reviewers both choose `ACCEPT`
  on episodes with an evidence requirement. The target is the reference team's
  mutual-approval curve across episodes.
- **AgentCollabBench** includes tracer durability (RTD), which measures how far
  an exact identifier survives as agents pass information along a communication
  graph. The target is the reference team's score profile across graph types.

Profile loss measures how far the student remains from the reference. Alignment
measures how much of the starting gap the design closes. A close match describes
the measured behavior and leaves open how the agents produced it.

## Reproduce the study

| Path | Role |
| --- | --- |
| [SwarmStill skill](skills/swarmstill/README.md) | Trusted-operator guidance for evidence review and fixed-P experiment design |
| [Iterative design](studies/swarmstill_design/README.md) | Design tools, benchmark launchers, and scoring for Emergent Collusion and AgentCollabBench |
| [Emergent Collusion](studies/swarmstill_ec/README.md) | Released trace analysis, evidence masking, and mutual-approval scoring |
| [AgentCollabBench generalization](studies/swarmstill_general/README.md) | Frozen design guide, forecast comparison, and original benchmark scoring |
| [Results](results/swarmstill/README.md) | Public derived results, figures, protocols, and analysis scripts |
| [Mesocosm smoke test](docs/native-multiagent.md) | Three OpenCode agents and the original Terminal-Bench fix-git verifier |

GPT-6 Astra and Claude Opus 5.5 were used for iterative RTD design with Qwen3-4B
and Gemma-4-E4B-it students. GPT-6.1 Sol also appears in earlier collusion design
and prediction studies. The study suite does not provide a full factorial
comparison of those designers and students. The archived iterative RTD protocol
does not include a verified final held-out score.

## Measurement

Profile loss is the mean absolute gap between student and reference behavior in
percentage points. The collusion profile uses mutual ACCEPT on constrained
episodes, with warm-up excluded. The RTD profile groups the official
AgentCollabBench tracer-durability score by topology. Report original task
accuracy separately when the verifier provides it.

Alignment is 100 times the fraction of baseline profile loss closed by the
design. Use the original student environment as the baseline for RTD. The
archived collusion alignment uses a never-colluding baseline. Sequence overlap
compares distributions of complete binary trajectories and reports the all-zero
contribution separately. See the linked study code for exact definitions.

## Local checks

```bash
uv sync --locked --extra dev --extra gateway --extra mcp
uv run mesocosm --help
uv run mesocosm research tools --json
uv run pytest tests/test_swarmstill_ec.py tests/test_examples.py
```

Run these checks on Linux. They do not require a GPU. Running the original experiments requires
the benchmark sources, configured endpoints and credentials, and the Linux/GPU
infrastructure described in the study guides. The native smoke test also requires
Linux Docker. Keep original task instructions, scorers, and held-out inputs fixed.

The shared Mesocosm runtime and its regression fixtures support the retained
operator and smoke-test paths. Study code and results come from the upstream
`swarmstill/generalization` and `results` branches. See [LICENSE](LICENSE).
