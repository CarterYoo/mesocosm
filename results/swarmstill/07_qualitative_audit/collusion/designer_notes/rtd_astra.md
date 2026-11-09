# Design notes

Scores ordered as (branching_tree, converging_dag, custom_graph).
Target: (100, 66.7, 50). Supplied original student: (100, 91.7, 58.3), distance 11.1.
Only permitted environment settings changed. Predictions/rationales were saved in test01.json–test05.json before each run.

| Test | Topology / turns / salience / distractors | Prediction | Measured | Distance |
|---|---|---|---|---|
| t01 | original / 1 / low / all | 95–100, 65–75, 45–55 | 83.3, 100, 50 | 16.7 |
| t02 | original / 1 / high / half | 100, ~83.3, 50–58.3 | 100, 100, 83.3 | 22.2 |
| t03 | original / 2 / high / all | ~100, 75, 50 | 100, 100, 50 | 11.1 |
| t04 | chain / 2 / high / all | 90–100, 65–80, 45–65 | 96.7, 94.4, 100 | 27.0 |
| t05 | original / 2 / low / all | 83.3–100, 83.3–100, ~50 | 91.7, 100, 66.7 | 19.4 |

- t01 tested weaker formatting for selective merge/relay decay. It hurt branching and left DAGs at ceiling; restore high salience and probe surrounding detail.
- t02 halved distractors to test label copying during shorter synthesis. Retention rose too far; restore all distractors and test extra rounds.
- t03 doubled turns to induce revision-related tag loss. It matched branching and custom groups but left DAGs at ceiling; best tested so far, tied with original distance.
- t04 added serial handoffs to reduce DAG retention. It erased custom-graph decay; restore original topology.
- Intended final probe: original / 1.5 / high / all, predicted (100,91.7,50). Runner and validator rejected 1.5 despite listing it in the domain; string encoding also rejected. No test consumed. Spec retained as rejected_turns_1_5.json.
- Actual t05 probes salience × length: extra rounds may recover t01's branching loss while weaker labels decay in DAG revisions. Favorable hypothesis (100,83.3,50), distance 5.5; no-interaction outcome (83.3,100,50), distance 16.7, also plausible. t03 remains fallback.

Final decision: t05 recovered some branching retention but overshot custom retention and left DAGs at ceiling. Select t03, the minimum-distance tested environment (11.1; alignment 0.0), matching the supplied original distance. It matches branching and custom targets exactly; DAG score remains 100 versus target 66.7. All five tests completed without missing scores.

Final environment: S.topology=original; S.turns=2; E.constraint_salience=high; E.distractor_load=all. Submission: t03.
