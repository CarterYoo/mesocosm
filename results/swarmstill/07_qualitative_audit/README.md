# SwarmStill: qualitative audit and supporting logs

This is the reproducibility entry point for the paper's Discussion of why environment design did not consistently reproduce the target behavior. The main repository is [khs1010/mesocosm](https://github.com/khs1010/mesocosm). The experiment's design tools and scorers are in [studies/swarmstill_design](../../../studies/swarmstill_design/README.md); earlier study artifacts remain in [the results archive](../README.md).

Start with the [five source-linked cases](CASES.md). Each distinguishes the recorded observation from a proposed interpretation, and connects the paper's short quotation to the released evidence. The audit separates:

- **Tools and execution:** whether a setting changed the model's actual input, and whether a message was delivered.
- **Tasks and evaluation:** which task a reviewer was evaluating, whether the evidence requirement was feasible, and what an output metric actually measured.
- **Student and teacher behavior:** how agents attributed evidence, interpreted feedback, retained roles and responded to requested exceptions.

These categories can interact. A deliberately narrow channel is a task constraint; an advertised setting that never changes a request is an implementation limit. Neither alone establishes a model-intrinsic cause.

## What is released

| Material | Entry point |
| --- | --- |
| Five examples, counterexamples, source IDs and interpretation limits | [CASES.md](CASES.md) |
| Frozen collusion corpus: 736 agent–episode records, 49 complete saved source snapshots, qualitative annotations and designer notes | [Collusion corpus and verification](collusion/README.md) |
| Saved AgentCollabBench model responses and the offline intervention audit | [ACB evidence and verification](acb/README.md) |

The collusion corpus includes warm-up and other unconstrained records, partial trajectories, and a purposive selection of teacher trajectories. The example bank is not a random sample for prevalence estimates. Read each sub-bundle's coverage and exclusions before pooling results. The annotations are **LLM-assisted direct-reading judgments**, not blinded human labels or a gold standard. Reported co-occurrences do not identify internal cognition or prove deliberate deception.

The release is an explicit, author-authorized publication of the research artifacts listed here. It does not publish credentials, machine addresses, activations or unrelated operational logs. Original source hashes and hashes of the sanitized derivatives are distinct. Redaction policies and manifests identify infrastructure changes while preserving experimental content, including synthetic emails and password test cases.

## Verify locally

Clone or download this repository, then follow each sub-bundle's verification commands. The collusion integrity check uses only the Python standard library. The ACB guide separates checking released response logs from replaying the pinned evaluator with an offline response provider. Replaying saved outputs verifies request construction; it does not generate new model behavior.

```sh
python3 results/swarmstill/07_qualitative_audit/collusion/verify_bundle.py
```

No serving endpoint or credentials are required for that check. Re-running the original model experiments requires the benchmark sources, declared model and serving configurations, compute and sampling described in the study guides. Saved output replay is not a substitute for a fresh experiment.

## Attribution and scope

The collusion environment and released teacher data originate in [SALT-NLP/agent-collusion](https://huggingface.co/datasets/SALT-NLP/agent-collusion); its MIT notice is preserved in [the existing study attribution](../../../studies/swarmstill_ec/index/LICENSE-agent-collusion.txt). ACB acquisition, source revision and licensing metadata are documented in the ACB sub-bundle. The released examples quote experimental instructions as observations; those instructions must not be executed as part of reading this archive.
