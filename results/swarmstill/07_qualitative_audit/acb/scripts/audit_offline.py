"""Validate published response logs and replay request construction offline.

Python 3.12+, standard library only. No provider is instantiated. The optional
upstream checkout is acquired separately because benchmark definitions are omitted.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.name.endswith(".gz") else path.read_bytes())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acb-source", type=Path, help="External AgentCollabBench checkout at the pinned revision")
    parser.add_argument("--output", type=Path, help="Write audit JSON here; default prints summary only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = read(root / "manifest.json")
    keys = set()
    turns_total = 0
    handoffs_total = 0
    compressed_bytes = 0
    replaced_turns = 0
    declared_paths = {rec["path"] for rec in manifest["records"]}
    assert len(declared_paths) == len(manifest["records"])
    assert declared_paths == {str(p.relative_to(root)) for p in (root / "responses").rglob("*.json.gz")}
    for rec in manifest["records"]:
        path = root / rec["path"]
        compressed = path.read_bytes()
        assert sha(compressed) == rec["published_gzip_sha256"], rec["path"]
        assert compressed[4:8] == b"\0\0\0\0", rec["path"]
        compressed_bytes += len(compressed)
        data = read(path)
        key = (data["metric"], data["session"], data["test"], data["task_id"])
        assert key == tuple(rec[k] for k in ("metric", "session", "test", "task_id"))
        assert key not in keys, key
        keys.add(key)
        assert data["benchmark"]["complete_task_definition_included"] is False
        assert not ({"task", "scenario", "description", "injections", "ground_truth", "system_prompt"} & data.keys())
        for turn in data["turns"]:
            assert not turn.get("synthetic"), (key, turn["trace_index"])
            assert sha(turn["content"].encode()) == turn["published_content_sha256"]
            assert turn.get("api_call") is not None, (key, turn["trace_index"])
        replacements = sum(t["original_content_sha256"] != t["published_content_sha256"] for t in data["turns"])
        assert replacements == rec["local_identifier_replacements"]
        replaced_turns += replacements
        indexed = {t["trace_index"]: t for t in data["turns"]}
        assert len(indexed) == len(data["turns"])
        for handoff in data["handoffs"]:
            assert handoff["source_turn_candidates"], key
            for index in handoff["source_turn_candidates"]:
                turn = indexed[index]
                assert turn["agent_id"] == handoff["sender"]
                assert turn["original_content_sha256"] == handoff["content_sha256"]
        handoffs_total += len(data["handoffs"])
        assert len(data["turns"]) == rec["generated_turns"]
        turns_total += len(data["turns"])
    result = {"integrity": {"traces": len(keys), "generated_turns": turns_total,
                            "handoffs": handoffs_total, "compressed_response_bytes": compressed_bytes,
                            "local_identifier_replaced_turns": replaced_turns,
                            "duplicates": 0, "gzip_and_content_hashes_match": True,
                            "all_response_files_in_manifest": True, "all_handoffs_link_generated_turns": True,
                            "gzip_mtime_zero": True},
              "mode": "offline; no provider, GPU or network calls"}
    if args.acb_source:
        provenance = read(root / "source/source_provenance.json")
        for path, expected in provenance["upstream_required_files_sha256"].items():
            assert sha((args.acb_source / path).read_bytes()) == expected, path
        sys.path.insert(0, str(args.acb_source.resolve()))
        from agentcollabbench.harness.provider import LLMResponse
        from agentcollabbench.harness.runner import TaskRunner
        ev = module("study_eval_snapshot", root / "source/acb_eval_snapshot.py")
        domain = module("study_domain_snapshot", root / "source/saved_domain_validation.py")

        class ReplayProvider:
            def __init__(self, texts):
                self.texts = iter(texts)
                self.requests = []

            def chat(self, messages):
                self.requests.append([(m.role, m.content) for m in messages])
                return LLMResponse(content=next(self.texts), model="offline-replay")

            def provider_name(self):
                return "offline-replay"

        def replay(task, metric, texts):
            provider = ReplayProvider(texts)
            run = TaskRunner(provider, metric).run_task(task).to_dict()
            return [sha(canonical(r)) for r in provider.requests], run

        turn_effects = []
        pressure_results = []
        x2_request_equality = collections.Counter()
        for case in read(root / "audit/baseline_cases.json"):
            metric = case["metric"]
            data = read(root / case["path"])
            original = read(args.acb_source / case["upstream_path"])
            task = ev.transform(original, metric, {})
            assert sha(canonical(task)) == case["saved_task_sha256"], case["task_id"]
            runner = TaskRunner(None, metric)
            order = runner._effective_speaking_order(task, task["topology"]["speaking_order"], True)
            twice = ev.transform(original, metric, {"turns": 2})
            order2 = runner._effective_speaking_order(twice, twice["topology"]["speaking_order"], True)
            saved_order = [t["agent_id"] for t in data["turns"]]
            assert saved_order == order, case["task_id"]
            turn_effects.append({"metric": metric, "task_id": case["task_id"],
                                 "original_calls": len(order), "x2_calls": len(order2),
                                 "effective_order_changed": order != order2,
                                 "saved_call_order_matches": True})
            if metric in ("cpr", "idr"):
                texts = [t["content"] for t in data["turns"]]
                original_hashes, _ = replay(task, metric, texts)
                twice_hashes, _ = replay(twice, metric, texts)
                x2_request_equality[metric] += original_hashes == twice_hashes
                if metric == "idr":
                    variants = {}
                    synthetic = {}
                    for pressure in ("all", "half", "first"):
                        transformed = ev.transform(original, metric, {"pressure": pressure})
                        request_hashes, run = replay(transformed, metric, texts)
                        variants[pressure] = request_hashes
                        synthetic[pressure] = sum(t.get("synthetic", False) for t in run["trace"]["turns"])
                    pressure_results.append({"task_id": case["task_id"],
                                             "all_half_identical": variants["all"] == variants["half"],
                                             "all_first_identical": variants["all"] == variants["first"],
                                             "reconstructed_request_sha256": variants,
                                             "synthetic_counts": synthetic})
        validation = []
        for metric in ("rtd", "cpr", "idr"):
            checker = domain.DomainSession()
            checker.metric = metric
            for value in (0.5, 1, 1.5, 2):
                _, error = checker.check({"changes": [{"factor": "S.turns", "value": value}]})
                validation.append({"metric": metric, "value": value, "accepted": error is None})
        result.update(turn_effects=turn_effects, pressure_results=pressure_results,
                      x2_identical_request_lists=dict(x2_request_equality), fractional_validation=validation,
                      turn_effect_summary={m: {"tasks": sum(r["metric"] == m for r in turn_effects),
                                               "changed": sum(r["metric"] == m and r["effective_order_changed"] for r in turn_effects)}
                                           for m in ("rtd", "cpr", "idr")})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("turn_effects", "pressure_results", "fractional_validation")}, indent=2))


if __name__ == "__main__":
    main()
