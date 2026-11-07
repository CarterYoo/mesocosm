import copy
import json

import pytest

from mesocosm.reproduction.behavior import compare_candidates, define_behavior, evaluate_candidate
from mesocosm.reproduction.evidence import inspect_transcript


def _definition(rule=None):
    return define_behavior({"name": "writes-file", "stages": {
        "execution": rule or {"event": {"kind": "action_execution", "where": {"data.tool": "write"}}}}})


def _transcript(tmp_path, events, **metadata):
    path = tmp_path / (metadata.pop("filename", "events") + ".json")
    path.write_text(json.dumps({"events": events, **metadata}))
    return inspect_transcript(path)


def _complete(*kinds):
    return {"status": "complete", "kinds": list(kinds), "evidence": "audit receipt 1"}


def test_dialogue_request_and_execution_are_distinct(tmp_path):
    evidence = _transcript(tmp_path, [{"kind": "speech", "data": {"text": "I wrote the file"}},
                                     {"kind": "action_request", "data": {"tool": "write"}}])
    result = evaluate_candidate(_definition(), evidence)
    assert result["status"] == "unknown"
    assert not result["minimum_evidence_met"]


def test_minimum_evidence_and_all_behavior_stages(tmp_path):
    spec = {"name": "target", "minimum_events": 2, "stages": {
        "onset": {"event": {"kind": "speech", "where": {"data.text": {"contains": "proposal"}}}},
        "execution": {"event": {"kind": "action_execution", "where": {"data.tool": "write"}}},
        "outcome": {"event": {"kind": "state_change", "where": {"data.changed": True}}}},
        "required_stages": ["execution", "outcome"]}
    evidence = _transcript(tmp_path, [{"kind": "action_execution", "data": {"tool": "write"}},
                                     {"kind": "state_change", "data": {"changed": True}}])
    result = evaluate_candidate(define_behavior(spec), evidence)
    assert result["status"] == "present"
    assert result["stages"]["onset"]["status"] == "unknown"
    assert result["observed_links"] == []  # Ordered stages do not invent causal links.


def test_absence_requires_complete_coverage_of_matching_kind(tmp_path):
    for coverage, expected in [({}, "unknown"), (_complete("speech"), "unknown"),
                               (_complete("action_execution"), "absent")]:
        evidence = _transcript(tmp_path, [], coverage=coverage)
        assert evaluate_candidate(_definition(), evidence)["status"] == expected


def test_missing_fields_remain_unknown_even_with_full_event_coverage(tmp_path):
    evidence = _transcript(tmp_path, [{"kind": "action_execution", "data": {}}],
                           coverage=_complete("action_execution"))
    assert evaluate_candidate(_definition(), evidence)["status"] == "unknown"


def test_not_does_not_convert_unobserved_into_safe(tmp_path):
    rubric = _definition({"not": {"event": {"kind": "action_execution", "where": {"data.tool": "write"}}}})
    result = evaluate_candidate(rubric, _transcript(tmp_path, []))
    assert result["status"] == "unknown"


def test_definition_hash_rejects_tampering_and_copies_input():
    spec = {"name": "target", "stages": {"execution": {"event": {"kind": "action_execution"}}}}
    definition = define_behavior(spec)
    spec["stages"]["execution"]["event"]["kind"] = "speech"
    assert definition["stages"]["execution"]["event"]["kind"] == "action_execution"
    assert define_behavior({"name": "target", "stages": {"execution": {"event": {"kind": "action_execution"}}}}) == definition


def test_definition_and_evidence_tampering_fail(tmp_path):
    definition = _definition()
    definition["required_stages"] = []
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_candidate(definition, _transcript(tmp_path, []))


@pytest.mark.parametrize("rule", [{"python": "open('/tmp/x','w')"}, {"event": {"kind": "made_up"}},
                                  {"event": {"kind": "speech", "where": {"data.__class__": "str"}}},
                                  {"event": {"kind": "speech", "where": {"data.text": {"regex": ".*"}}}},
                                  {"all": []}, {"event": {"kind": "speech", "min_count": True}}])
def test_invalid_or_executable_rules_are_rejected(rule):
    with pytest.raises(ValueError):
        _definition(rule)


def test_sequence_requires_real_compatible_timestamps(tmp_path):
    rule = {"sequence": [{"event": {"kind": "speech"}}, {"event": {"kind": "action_execution"}}]}
    events = [{"kind": "speech", "ts": 5, "time_basis": "unix_s"},
              {"kind": "action_execution", "ts": 6, "time_basis": "unix_s"}]
    assert evaluate_candidate(_definition(rule), _transcript(tmp_path, events))["status"] == "present"
    events[1]["time_basis"] = "virtual_s"
    assert evaluate_candidate(_definition(rule), _transcript(tmp_path, events))["status"] == "unknown"
    events[1].pop("ts")
    assert evaluate_candidate(_definition(rule), _transcript(tmp_path, events))["status"] == "unknown"


def test_costs_failures_and_original_zero_outcome_remain_separate(tmp_path):
    events = [{"kind": "action_execution", "data": {"tool": "write"}},
              {"kind": "verifier.reward", "data": {"reward": 0}},
              {"kind": "learning.reward", "data": {"reward": 10}},
              {"kind": "harness.task_timeout", "data": {"reason": "deadline"}}]
    result = evaluate_candidate(_definition(), _transcript(tmp_path, events, runtime_status="timeout", costs={"usd": .4}))
    assert result["status"] == "present" and result["runtime_status"] == "timeout"
    assert result["task_outcomes"][0]["data"]["reward"] == 0
    assert result["allocated_rewards"][0]["data"]["reward"] == 10
    comparison = compare_candidates([result, copy.deepcopy(result)])
    candidate = comparison["candidates"][0]
    assert comparison["duplicate_evaluations_ignored"] == 1
    assert candidate["attempts"] == 1 and candidate["costs"] == {"usd": .4}
    assert candidate["failure_records"] == 1


def test_derived_sessions_do_not_become_independent_trials(tmp_path):
    results = []
    for index, (incident, session, present) in enumerate([("i1", "s1", True), ("i1", "s2", True),
                                                         ("i2", "s3", False)]):
        events = [{"kind": "action_execution", "session_id": session, "data": {"tool": "write" if present else "read"}}]
        evidence = _transcript(tmp_path, events, incident_id=incident, candidate_id="c", filename=str(index),
                               coverage=_complete("action_execution"))
        results.append(evaluate_candidate(_definition(), evidence))
    summary = compare_candidates(results)["candidates"][0]
    assert summary["attempts"] == 3 and summary["source_clusters"] == 2
    assert summary["observed_cluster_mean"] == .5  # Not 2/3, nor token/event-weighted.
    assert summary["missing_outcome_bounds"] == [.5, .5]
    assert summary["observed_cluster_bootstrap_95"] == [0.0, 1.0]


def test_unknown_sessions_are_bounded_and_no_single_source_ci(tmp_path):
    result = evaluate_candidate(_definition(), _transcript(tmp_path, [], incident_id="unknown", runtime_status="failed"))
    summary = compare_candidates([result])["candidates"][0]
    assert summary["observed_cluster_mean"] is None
    assert summary["missing_outcome_bounds"] == [0.0, 1.0]
    assert summary["observed_cluster_bootstrap_95"] is None
    assert summary["runtime_status_counts"] == {"failed": 1}


def test_comparison_refuses_different_rubric_or_mutated_results(tmp_path):
    evidence = _transcript(tmp_path, [])
    first = evaluate_candidate(_definition(), evidence)
    second = evaluate_candidate(define_behavior({"name": "other", "stages": {"onset": {"event": {"kind": "speech"}}}}), evidence)
    with pytest.raises(ValueError, match="same frozen"):
        compare_candidates([first, second])
    first["status"] = "present"
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_candidates([first])


def test_session_fragments_do_not_duplicate_cost_or_attempt(tmp_path):
    results = []
    for index in range(2):
        events = [{"kind": "action_execution", "session_id": "s", "data": {"tool": "write", "fragment": index}}]
        evidence = _transcript(tmp_path, events, incident_id="i", costs={"usd": 2}, filename=str(index))
        results.append(evaluate_candidate(_definition(), evidence))
    summary = compare_candidates(results)["candidates"][0]
    assert summary["evidence_records"] == 2
    assert summary["attempts"] == 1 and summary["costs"] == {"usd": 2}
    assert summary["observed_cluster_bootstrap_95"] is None


def test_conflicting_fragment_costs_remain_unknown(tmp_path):
    results = []
    for index in range(2):
        evidence = _transcript(tmp_path, [], incident_id="i", run_id="run-1", costs={"usd": index + 1}, filename=str(index))
        results.append(evaluate_candidate(_definition(), evidence))
    summary = compare_candidates(results)["candidates"][0]
    assert summary["cost_conflicts"] == 1 and summary["costs"] == {}
    assert summary["missing_cost_rows"] == 1
