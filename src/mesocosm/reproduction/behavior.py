"""Frozen observable behavior rubrics and source-clustered candidate comparisons.

This evaluator is a bounded data DSL, not a Python/LLM execution interface. A
nonmatch is unknown without declared complete coverage. Success claims in speech
cannot satisfy execution/state selectors. Stage correspondence is descriptive;
only explicit source links establish a recorded transmission path.
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter, defaultdict
from typing import Any

from mesocosm.reproduction.evidence import KINDS, digest, validate_transcript

EVALUATOR_VERSION = "mesocosm-observable-behavior-v1"
STAGES = ("onset", "transmission", "adoption", "execution", "outcome")
_FIELDS = {"kind", "source_kind", "agent_id", "session_id", "request_id", "run_id",
           "source_event_id", "display_name", "principal", "evidence_basis"}
_MISSING = object()


def _validate_rule(rule: Any, depth: int = 0) -> None:
    if depth > 12 or not isinstance(rule, dict) or len(rule) != 1:
        raise ValueError("rule must have exactly one DSL operator (maximum depth 12)")
    op, value = next(iter(rule.items()))
    if op in {"all", "any", "sequence"}:
        if not isinstance(value, list) or not 1 <= len(value) <= 64:
            raise ValueError(f"{op} requires 1..64 rules")
        for child in value:
            _validate_rule(child, depth + 1)
        if op == "sequence" and any("event" not in child or child["event"].get("min_count", 1) != 1
                                    for child in value):
            raise ValueError("sequence supports only event selectors with min_count=1")
    elif op == "not":
        _validate_rule(value, depth + 1)
    elif op == "event":
        if not isinstance(value, dict) or set(value) - {"kind", "where", "min_count"}:
            raise ValueError("event supports kind, where, min_count only")
        if value.get("kind") not in KINDS:
            raise ValueError("event kind must be an observable normalized category")
        count = value.get("min_count", 1)
        if type(count) is not int or not 1 <= count <= 1_000_000:
            raise ValueError("min_count must be a positive integer")
        where = value.get("where", {})
        if not isinstance(where, dict) or len(where) > 64:
            raise ValueError("where must contain at most 64 field predicates")
        for field, predicate in where.items():
            if (not isinstance(field, str) or not field or field.startswith("_")
                    or any(not part or part.startswith("_") for part in field.split("."))
                    or not (field in _FIELDS or field.startswith(("data.", "identity.")))):
                raise ValueError(f"unsupported evidence field: {field}")
            if isinstance(predicate, dict):
                if len(predicate) != 1 or next(iter(predicate)) not in {"eq", "ne", "contains", "gte", "lte", "exists"}:
                    raise ValueError("unsupported field predicate")
                operator, expected = next(iter(predicate.items()))
                if operator == "exists" and type(expected) is not bool:
                    raise ValueError("exists requires a boolean")
                if operator == "contains" and type(expected) not in {str, int, float, bool}:
                    raise ValueError("contains requires a scalar")
                if operator in {"gte", "lte"} and (type(expected) not in {int, float}
                                                       or not math.isfinite(expected)):
                    raise ValueError("numeric predicate requires a finite number")
            elif isinstance(predicate, (list, tuple)):
                raise ValueError("use eq for structured predicates")
    else:
        raise ValueError(f"unsupported rule operator: {op}")


def define_behavior(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical frozen definition; callers persist it by its content hash."""
    if set(spec) - {"name", "description", "version", "stages", "required_stages", "minimum_events"}:
        raise ValueError("unknown behavior definition field")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("behavior name is required")
    version = spec.get("version", 1)
    if not ((type(version) is int and version > 0) or (isinstance(version, str) and version.strip())):
        raise ValueError("version must be a positive integer or nonempty string")
    if not isinstance(spec.get("description", ""), str):
        raise ValueError("description must be text")
    stages = spec.get("stages")
    if not isinstance(stages, dict) or not stages or set(stages) - set(STAGES):
        raise ValueError("stages must define observable onset/transmission/adoption/execution/outcome rules")
    for rule in stages.values():
        _validate_rule(rule)
    required = spec.get("required_stages", list(stages))
    if (not isinstance(required, list) or not required or len(required) != len(set(required))
            or set(required) - set(stages)):
        raise ValueError("required_stages must name defined stages without duplicates")
    minimum = spec.get("minimum_events", 1)
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum_events must be positive")
    definition = copy.deepcopy({
        "schema_version": 1, "evaluator_version": EVALUATOR_VERSION,
        "name": spec["name"], "description": spec.get("description", ""),
        "version": spec.get("version", 1), "stages": stages,
        "required_stages": required, "minimum_events": minimum,
    })
    definition["definition_sha256"] = digest(definition)
    return definition


def _validate_definition(definition: dict[str, Any]) -> None:
    body = {k: v for k, v in definition.items() if k != "definition_sha256"}
    if definition.get("definition_sha256") != digest(body):
        raise ValueError("behavior definition hash mismatch")
    if definition.get("evaluator_version") != EVALUATOR_VERSION or definition.get("schema_version") != 1:
        raise ValueError("unsupported evaluator/definition version")
    spec = {k: v for k, v in body.items() if k not in {"evaluator_version", "schema_version"}}
    if define_behavior(spec) != definition:
        raise ValueError("noncanonical behavior definition")


def _field(event: dict[str, Any], path: str) -> Any:
    value: Any = event
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _matches(event: dict[str, Any], selector: dict[str, Any]) -> bool | None:
    if event["kind"] != selector["kind"]:
        return False
    unknown = False
    for path, predicate in selector.get("where", {}).items():
        actual = _field(event, path)
        op, expected = next(iter(predicate.items())) if isinstance(predicate, dict) else ("eq", predicate)
        if op == "exists":
            if (actual is not _MISSING and actual is not None) != expected:
                return False
        elif actual is _MISSING or actual is None:
            unknown = True
        elif op in {"gte", "lte"}:
            if type(actual) not in {int, float}:
                unknown = True
            elif not (actual >= expected if op == "gte" else actual <= expected):
                return False
        elif op == "contains":
            if not isinstance(actual, (str, list, dict)) or (isinstance(actual, str) and not isinstance(expected, str)):
                unknown = True
            elif expected not in actual:
                return False
        elif (op == "eq" and actual != expected) or (op == "ne" and actual == expected):
            return False
    return None if unknown else True


def _result(status: str, evidence: list[str], reason: str | None = None) -> dict[str, Any]:
    return {"status": status, "evidence_event_ids": sorted(set(evidence)), "reason": reason}


def _evaluate(rule: dict[str, Any], events: list[dict[str, Any]], coverage: dict[str, Any]) -> dict[str, Any]:
    op, value = next(iter(rule.items()))
    if op == "event":
        matches = [(event, _matches(event, value)) for event in events]
        found = [event["event_id"] for event, match in matches if match is True]
        if len(found) >= value.get("min_count", 1):
            return _result("present", found)
        complete = coverage.get("status") == "complete" and value["kind"] in coverage.get("kinds", [])
        if complete and not any(match is None for _, match in matches):
            return _result("absent", found, "complete declared coverage; insufficient matching events")
        return _result("unknown", found, "missing predicate fields or incomplete declared coverage")
    if op == "not":
        result = _evaluate(value, events, coverage)
        return {**result, "status": {"present": "absent", "absent": "present", "unknown": "unknown"}[result["status"]]}
    children = [_evaluate(child, events, coverage) for child in value]
    ids = [event_id for child in children for event_id in child["evidence_event_ids"]]
    states = [child["status"] for child in children]
    if op == "sequence":
        # Source file ordering is not a global clock. Compare only finite timestamps
        # with identical declared time bases; require strict ordering and unique events.
        eligible = [event for event in events if type(event.get("ts")) in {int, float}
                    and event.get("time_basis") not in {None, "source_unspecified"}]
        endings: list[dict[str, Any]] = []
        chain: list[str] = []
        for child in value:
            current = [event for event in eligible if _matches(event, child["event"]) is True]
            if not endings:
                endings = current
            else:
                endings = [event for event in current if any(
                    before["time_basis"] == event["time_basis"] and before["ts"] < event["ts"]
                    for before in endings)]
            if not endings:
                break
            chain.extend(event["event_id"] for event in endings)
        if endings and len(chain) >= len(value) and all(state == "present" for state in states):
            return _result("present", chain)
        # Only a proven missing component establishes absence. Clock and sequence
        # uncertainty is retained even when isolated components were observed.
        status = "absent" if "absent" in states else "unknown"
    elif op == "all":
        status = "absent" if "absent" in states else "unknown" if "unknown" in states else "present"
    else:
        status = "present" if "present" in states else "unknown" if "unknown" in states else "absent"
    return {**_result(status, ids, None if status == "present" else "component or ordering evidence incomplete"),
            "components": children}


def evaluate_candidate(definition: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    """Evaluate immutable evidence with separate behavior, task outcome and failure fields."""
    _validate_definition(definition)
    validate_transcript(transcript)
    events = transcript["events"]
    stages = {stage: _evaluate(rule, events, transcript["coverage"]) for stage, rule in definition["stages"].items()}
    required = [stages[stage]["status"] for stage in definition["required_stages"]]
    status = "absent" if "absent" in required else "unknown" if "unknown" in required else "present"
    evidence_ids = {event_id for stage in stages.values() for event_id in stage["evidence_event_ids"]}
    if status == "present" and len(evidence_ids) < definition["minimum_events"]:
        status = "unknown"
    metadata = transcript.get("metadata", {})
    summary = metadata.get("summary", {})
    result = {
        "schema_version": 1, "evaluator_version": EVALUATOR_VERSION,
        "definition_sha256": definition["definition_sha256"],
        "transcript_sha256": transcript["transcript_sha256"], "source": transcript["source"],
        "incident_id": transcript["incident_id"], "session_ids": transcript["session_ids"],
        "run_id": transcript.get("run_id"), "candidate_id": metadata.get("candidate_id"),
        "status": status, "stages": stages,
        "minimum_evidence_met": len(evidence_ids) >= definition["minimum_events"],
        "evidence_event_ids": sorted(evidence_ids),
        "runtime_status": metadata.get("runtime_status", summary.get("status", "unknown")),
        "costs": copy.deepcopy(metadata.get("costs", {})),
        "task_outcomes": [e for e in events if e["kind"] == "outcome"],
        "allocated_rewards": [e for e in events if e["kind"] == "reward"],
        "failures": [e for e in events if e["source_kind"].endswith((".failed", "_timeout", ".aborted"))],
        "stage_times": {stage: [{"ts": event["ts"], "time_basis": event["time_basis"],
                                  "event_id": event["event_id"]} for event in events
                                 if event["event_id"] in item["evidence_event_ids"]]
                        for stage, item in stages.items()},
        "observed_links": [link for event in events if event["event_id"] in evidence_ids for link in event["links"]],
        "coverage": transcript["coverage"],
        "limitations": ["source records are evidence claims, not proof of collection completeness",
                        "stage correspondence and temporal order do not establish causal transmission"],
    }
    result["evaluation_sha256"] = digest(result)
    return result


def compare_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare candidates without counting derived traces/tokens as independent trials.

    Correlated sessions within an incident/source contribute one cluster mean.
    Unknowns remain in missing-outcome bounds; failed attempts remain in the table.
    Percentile bootstrap intervals are descriptive, conditional on observed clusters.
    """
    if not results:
        raise ValueError("at least one evaluated candidate is required")
    definitions = {(r.get("definition_sha256"), r.get("evaluator_version")) for r in results}
    if len(definitions) != 1:
        raise ValueError("comparison requires the same frozen definition and evaluator")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[Any, ...], str] = {}
    duplicates = 0
    for result in results:
        body = {k: v for k, v in result.items() if k != "evaluation_sha256"}
        if result.get("evaluation_sha256") != digest(body):
            raise ValueError("evaluation hash mismatch")
        candidate = str(result.get("candidate_id") or "default")
        identity = (candidate, result["transcript_sha256"], result["definition_sha256"])
        if identity in seen:
            duplicates += 1
            continue
        seen[identity] = result["evaluation_sha256"]
        grouped[candidate].append(result)
    candidates = []
    for candidate, rows in sorted(grouped.items()):
        attempts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            incident = str(row.get("incident_id") or row["source"]["id"])
            execution = row.get("run_id") or digest(row.get("session_ids") or ["unknown-session"])
            attempts[(incident, str(execution))].append(row)
        consolidated = []
        cost_conflicts = 0
        for (incident, execution), fragments in attempts.items():
            fragment_states = {fragment["status"] for fragment in fragments}
            runtime_states = {fragment["runtime_status"] for fragment in fragments}
            known_costs = {digest(fragment["costs"]): fragment["costs"] for fragment in fragments if fragment["costs"]}
            if len(known_costs) > 1:
                cost_conflicts += 1
            consolidated.append({
                "incident_id": incident, "session_ids": [execution], "run_id": execution,
                "status": next(iter(fragment_states)) if len(fragment_states) == 1 else "unknown",
                "runtime_status": next(iter(runtime_states)) if len(runtime_states) == 1 else "unknown",
                "costs": next(iter(known_costs.values())) if len(known_costs) == 1 else {},
                "failures": list({event["event_id"]: event for fragment in fragments
                                  for event in fragment["failures"]}.values()),
            })
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in consolidated:
            clusters[row["incident_id"]].append(row)
        values = []
        lower, upper = [], []
        for cluster_rows in clusters.values():
            # Reimports/derived fragments from one session never increase sample count.
            sessions: dict[str, list[str]] = defaultdict(list)
            for row in cluster_rows:
                session_key = digest(row.get("session_ids") or [row.get("run_id") or "unknown-session"])
                sessions[session_key].append(row["status"])
            states = [values[0] if len(set(values)) == 1 else "unknown" for values in sessions.values()]
            known = [state == "present" for state in states if state != "unknown"]
            if known:
                values.append(sum(known) / len(known))
            lower.append(states.count("present") / len(states))
            upper.append((states.count("present") + states.count("unknown")) / len(states))
        interval = None
        if len(values) >= 2:
            rng = random.Random(0)
            samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000))
            interval = [samples[49], samples[1949]]
        costs: dict[str, float] = defaultdict(float)
        missing_cost_rows = 0
        for row in consolidated:
            if not row["costs"]:
                missing_cost_rows += 1
            for unit, amount in row["costs"].items():
                if type(amount) not in {int, float} or not math.isfinite(amount) or amount < 0:
                    raise ValueError("costs must contain nonnegative finite numeric units")
                costs[unit] += amount
        candidates.append({
            "candidate_id": candidate, "attempts": len(consolidated), "evidence_records": len(rows),
            "source_clusters": len(clusters), "cost_conflicts": cost_conflicts,
            "status_counts": dict(Counter(row["status"] for row in consolidated)),
            "runtime_status_counts": dict(Counter(row["runtime_status"] for row in consolidated)),
            "observed_cluster_mean": sum(values) / len(values) if values else None,
            "missing_outcome_bounds": [sum(lower) / len(lower), sum(upper) / len(upper)],
            "observed_cluster_bootstrap_95": interval,
            "known_source_clusters": len(values), "costs": dict(costs), "missing_cost_rows": missing_cost_rows,
            "failure_records": sum(len(row["failures"]) for row in consolidated),
        })
    return {"schema_version": 1, "definition_sha256": results[0]["definition_sha256"],
            "evaluator_version": EVALUATOR_VERSION, "candidates": candidates,
            "duplicate_evaluations_ignored": duplicates,
            "uncertainty": {"unit": "incident/source cluster; sessions nested within source",
                            "method": "2000 deterministic percentile bootstrap resamples of observed cluster means",
                            "caveats": ["no interval with fewer than two observed source clusters",
                                        "small-cluster intervals are unstable; unknown outcomes have separate bounds",
                                        "derived sources require a common incident_id to avoid false independence",
                                        "observational comparison does not establish intervention causality"]}}
