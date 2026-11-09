# SwarmStill · Mesocosm hackathon fork

Mesocosm runs multi-agent experiments under controlled conditions. SwarmStill used
it to explore whether frontier designers could reproduce a measured collective
behavior in a team of small models by changing the agents' structure (S) and
environment (E). The student model (M) and its weights (P) were fixed within each
comparison. The fork supports fixed-weight runs and in-context feedback.

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
