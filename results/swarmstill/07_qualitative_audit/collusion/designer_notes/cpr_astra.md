# Design notes

Target (tree, DAG, custom): (20.0, 15.5, 47.2). Original student: (14.3, 0.0, 44.4), distance 8.0. Five tests; change only topology, turns, carrier position. Metric counts any scored acceptance by each non-seed agent.

| Test | Topology | Turns | Carrier | Prediction | Result | Distance |
|---|---|---|---|---|---|---|
| t01 | root_broadcast | 1 | original | (15,12,45) | (20.7,40.5,22.2) | 16.9 |
| t02 | chain | 1 | original | (25,25,40) | (30.0,54.8,55.6) | 19.2 |

Test 1 rationale: target the DAG baseline gap by adding root edges, expecting limited tree/custom disruption. Actual result: tree close, DAG much too high, custom too low. Decision: test chain routing to reduce direct broadcasts and remove custom corrective feedback paths.

Runtime: t01 completed after about 63 minutes. Runner permits only one active test; an earlier t02 attempt was rejected and consumed no budget. Continue waiting through longer runtimes. Traces show multiple rounds (e.g. 28 turns for SWE-063 at multiplier 1); earlier one-pass reach calculations are not bounds on the actual metric. Prompted claims and generated errors can also affect scores beyond carrier reach.

Test 2 pre-run prediction: chain, turns 1, original carrier -> approximately (25,25,40), distance below 16.9. Removing DAG direct broadcasts may reduce their pollution; removing corrective feedback in custom graphs may increase persistence. Keep turns/carrier fixed to isolate routing. Updated test2.json after reading t01.

Test 2 result: all groups overshoot, especially DAGs. Chain did not reduce DAG pollution as predicted. Decision: restore original topology, whose baseline was closer, and test conversation length instead of further increasing exposure.

Test 3 pre-run: original topology, turns 1.5, original carrier. Predict (20,15,50). Original DAG independent sources may curb propagation while extra rounds allow modest later adoption; tree/custom original baselines were slightly low. Expect distance below 8.0.

Test 3 validation issue: runner rejected advertised S.turns=1.5 (both number and string); no test started. S.turns=2 passes check. Revised pre-run configuration: original, 2, original. Revised prediction (25,20,55), distance about 6; stronger length change may overshoot somewhat. Still tests restoring independent DAG sources plus additional adoption opportunities.

While t03 runs, checked S.turns=0.5: also rejected despite advertised domain. Both fractional settings are unavailable via validation; integer 1 and 2 are accepted. Validation used no test budget.
