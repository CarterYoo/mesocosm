# Design notes

## Situation (before any test)
- Target (Qwen3-8B, original env): branching 100 / converging 66.7 / custom 50.0.
- Student (Qwen3-4B, original env): 100 / 91.7 / 58.3 -> distance 11.1. Student is too durable.
- Need: keep branching at 100, drop converging by ~25 (= 1.5 of 6 tasks), custom by ~8 (= 0.5 of 6 tasks).
- 6 tasks per group, 1 task flip = 16.7 points in a group, so results are coarse and noisy.
- Custom: 5/6 tasks are multi-constraint (joint exact) with repeated speaking rounds. Converging: depth 1
  (090,133,169,SWE-061) or depth 2 (076,077); A2 etc. are separate roots without the tag.
- Lever directions (my read): root_broadcast/full add direct root edges -> more durable (wrong way);
  distractor half/none -> less noise -> more durable (wrong way); low salience -> less copying (right way,
  but global, may hit branching); chain -> tag must pass through untagged sibling roots (right way for
  converging, but deepens trees too); turns 0.5 -> may truncate one-pass orders (risky, unknown impl).

## T1 (t01): salience=low
- Expect: branching 92-100, converging ~75, custom 42-50, distance ~4-6. Why: plain formatting -> less verbatim copying.
- Result t01: 100 / 100 / 66.7, distance 16.7 (worse than original 11.1). Prediction wrong: low salience did not
  reduce copying; converging 090/076/077 all 1.0, custom 061 = 1.0 (target 0).
- Learned from trajectories (results/t01): score = union over ALL turns per layer (088: pass 1 had no ID, pass 2 did
  -> 1.0; 107 later turns dropped but score 1.0). Layer-level (any agent at the layer), not per-agent.
  Every root gets the injection: converging A2/A4 (untagged in task text) output the ID from turn 1, so aggregators see
  2-4 tagged inputs. Many turns (22-40 steps) give many chances -> student near-saturated.
- Service limitation: S.turns rejects every float (0.5, 1.5, even 1.0) as "outside its domain"; only 1 and 2 accepted
  via `check`. turns=2 is the wrong direction. So turns is unusable to cut chances.
- Topology reading: full/root_broadcast put every agent 1 hop from the root -> depth shrinks, more sources -> more
  durable (wrong way). chain = single tagged root, every agent its own layer, depth n-1 -> only lever left that
  lowers durability.

## T2 (t02): topology=chain (salience back to high, distractors all)
- Expect: branching 85-100 (6-agent two-tracer trees 137/156/171 become 5-hop chains, may lose the last hop),
  converging 75-92 (aggregator now has one tagged input; 077 two tracers over 3 hops may drop to 0.67),
  custom 42-58 (073/107 become depth-2 chains, risk; 080/059 stay 0). Distance ~6-10.
- Result t02: 85.8 / 100 / 66.7, distance 21.4. Prediction wrong for converging: chain did not cut it at all
  (22-40 steps; every relay re-includes the ID in some pass, union over turns saturates). Branching paid for the
  extra hops (137 -> 0.4, 171 -> 0.75). Custom flips are coin-flips at the first relay of two-tracer tasks
  (059 went 0 -> 1, 107 went 1 -> 0 vs t01). Union-over-turns confirmed (171: only pass 3 reached A4).
- Conclusion so far: no tested lever lowers converging; it sits at 92-100 for the student. turns<1 is the one lever
  that would cut chances and the service rejects it. Every untested lever (turns 2, full, root_broadcast) adds
  exposure. Distractor load is the only lever whose direction I cannot call from the data.

## T3 (t03): distractor_load=none (topology original, salience high)
- Expect: branching 100, converging 100, custom 67-83 (shorter injected block -> IDs more prominent, first-relay
  drops in 080/059 less likely); distance ~14-17. Run anyway because my salience prior was wrong and distractors
  change what the root relays as its "context block"; if converging drops, follow up on it.
- Result t03: 100 / 100 / 75.0, distance 19.4. Prediction right (wrong direction): without distractors per-turn
  inclusion rose (137/156 mostly 11 every pass), custom rose (080 0 -> 1, 059 -> 1). Distractors 'all' (default)
  is the least durable setting.
- Map of the space: every available lever is flat or raises durability vs the original env; converging is pinned at
  ~92-100 (union over 6-40 steps, all roots tagged). Custom is a coin flip per two-tracer task at the first relay
  (061: 1,1,0; 059: 0,1,1; 107: 1,0,0.5 across t01-t03). Best reachable design = original env
  (high salience, all distractors, original topology, turns 1); expected ~100 / 92-100 / ~58-67, distance ~11-14.

## T4 (t04): original environment (topology original, no other change)
- Expect: branching 100, converging 92-100, custom 50-67; distance 8-14. Needed as a tested candidate to submit.
- t04 failed on the test machine after 3 min (refunded, not counted). Re-running the same design as t05.
- Result t05 (original env): 100 / 100 / 50.0, distance 11.1. Within prediction. Custom group mean hit 50 but
  per task it is luck (080 1 vs target 0, 107 0 vs target 1); converging stays at 100 (target 66.7).

## Decision
- Submit t05 (original environment). Lowest measured distance (11.1 vs 16.7 / 21.4 / 19.4), and the only setting
  that no data argue against: low salience and no distractors raise durability, chain costs branching without
  touching converging. The 1 remaining test is left unused: no remaining valid lever (turns 2, full,
  root_broadcast, half distractors) has a mechanism that lowers converging, so it could not beat t05 except by noise.
- Residual gap: converging 100 vs 66.7. Cause: every root is injected and the score is the union over 6-40 steps,
  so the 4B aggregator re-states the ID in at least one pass. The lever that would attack it (turns < 1) is rejected
  by the service ("S.turns=0.5 is outside its domain"; any float is rejected).
