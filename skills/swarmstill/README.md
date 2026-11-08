# SwarmStill skill

A repository-distributed skill for using Mesocosm to study multi-agent behavior:
review evidence, propose controlled experiments, and distinguish structural hypotheses
from measured behavioral reproduction. The package contains a short `SKILL.md`, two
on-demand references and agent UI metadata. Execution stays in Mesocosm.

There is no hosted service to deploy or subscribe to. Actual model calls and experimental
compute still require the user's configured resources and may have their own costs.

## Install

Clone Mesocosm, or update an existing checkout:

```bash
git clone https://github.com/GRAIL-center/mesocosm.git
cd mesocosm
```

For a first Codex installation, copy the **whole skill directory**, including references:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/swarmstill "${CODEX_HOME:-$HOME/.codex}/skills/swarmstill"
```

If `swarmstill` is already installed, update that existing copy deliberately; do not repeat
the copy command into an existing directory. Start a new agent session after installation.
For another compatible agent, use its skill installation directory instead. The copied
skill does not depend on its position inside the source checkout.

The separate Mesocosm runtime needs Python 3.11–3.13 and `uv`. From the Mesocosm checkout:

```bash
uv sync --locked
uv run mesocosm --help
uv run mesocosm research tools --json
```

An existing installed `mesocosm` executable can be used directly instead of `uv run`.
Installing this skill does not install Python dependencies, register MCP servers, modify
credentials, or provision compute.

## Use

Example requests:

```text
Use $swarmstill to inspect Mesocosm's current tools and find the supported
memory and communication controls. Do not start model or worker jobs.

Use $swarmstill to analyze this reviewed transcript and propose a minimal
plausible harness. Cite evidence and identify settings the trace cannot reveal.

Use $swarmstill to plan a teacher/student comparison with one intervention
hidden from the designer. Preserve the benchmark's original evaluator.
```

The skill first discovers the installed tool contracts. It reads only the reference
needed for the request and uses bounded evidence pages. Ordinary analysis and planning
can proceed without a running world; execution requires an approved Linux worker,
model endpoint and compatible original task.

## Operator and designer are separate roles

The public `mesocosm research` CLI is a trusted-operator workbench. It can access local
paths and broader research operations. This skill uses that interface from the operator
context and does not make it safe to expose unchanged to a blinded designer.

A formal isolated inverse study additionally needs a reviewed masked corpus, a restricted
designer interface, and a fresh designer process inside an OS sandbox. An already installed
local designer MCP can provide that interface; paid remote hosting is unnecessary.
The base CLI's presence does not establish that the designer extension is installed.
See [the inverse-study procedure](references/inverse-study.md).

The repository now provides the optional designer extension: install with
`uv sync --locked --extra mcp` and follow the [operator setup guide](https://github.com/GRAIL-center/mesocosm/blob/main/docs/designer-mcp.md)
for `mesocosm designer mask-corpus` and `mesocosm designer serve`. This skill does not install
or create an isolation launcher. Without an actual OS boundary, produce an operator
analysis or hypothesis draft and mark the isolated designer test unavailable.

## Runtime documentation

- [Research workbench](https://github.com/GRAIL-center/mesocosm/blob/main/docs/reproduction.md)
- [Runtime requirements](https://github.com/GRAIL-center/mesocosm/blob/main/docs/research-runtime.md)
- [Task integrations](https://github.com/GRAIL-center/mesocosm/blob/main/docs/integrations.md)
- [Capability evidence](https://github.com/GRAIL-center/mesocosm/blob/main/docs/capability-matrix.md)
- [Fixed-policy populations and collective detector](https://github.com/GRAIL-center/mesocosm/blob/main/docs/fixed-p-study.md)
- [Designer policy selection and real SFT/LoRA](https://github.com/GRAIL-center/mesocosm/blob/main/docs/designer-policy.md)

Source and runtime licensing follow the repository's
[license](https://github.com/GRAIL-center/mesocosm/blob/main/LICENSE).
