"""Export generated ACB responses, not benchmark task definitions or evaluator code.

All inputs are explicit arguments. No providers, GPUs, remote services or network calls.
"""
from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import json
import re
import shutil
import textwrap
from pathlib import Path

COMMIT = "f016f600568b6d8127dc861e4c83c87b72750d63"
REPOSITORY = "https://github.com/aritra741/AgentCollabBench"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def safe_text(text, replacements):
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2).encode() + b"\n"
    if path.name.endswith(".gz"):
        data = gzip.compress(data, compresslevel=9, mtime=0)
    path.write_bytes(data)
    return sha(data)


def structure(task, metric):
    topology = task["topology"]
    inj = task["injections"][metric]
    permitted = ("expected_turns", "salience", "tracer_id", "injection_layer", "constraint_type",
                 "seed_agent", "patient_zero", "extra_agents", "target_agent")
    return {
        "expected_turns": task.get("expected_turns"),
        "topology": {"type": topology["type"], "edges": topology["edges"],
                     "speaking_order": topology["speaking_order"],
                     "agents": [{k: a.get(k) for k in ("agent_id", "role", "receives_from")}
                                for a in topology["agents"]]},
        "injection_structure": {k: inj[k] for k in permitted if k in inj},
        "distractor_count": len(inj.get("distractors", [])),
        "peer_pressure_schedule": [{"speaker": e.get("speaker"), "turn_index": e.get("turn_index"),
                                    "withheld_content_sha256": sha(e.get("message", "").encode())}
                                   for e in inj.get("peer_pressure_turns", [])],
        "withheld_field_hashes": {k: sha(canonical(v)) for k, v in inj.items() if k not in permitted},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-acb", type=Path, required=True, help="Original acbd directory")
    ap.add_argument("--teacher-runs", type=Path, required=True, help="Original teacher acb_runs directory")
    ap.add_argument("--upstream", type=Path, required=True, help="AgentCollabBench at the pinned revision")
    ap.add_argument("--scores", type=Path, required=True, help="Original generalization scores_baseline.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--redact-exact", action="append", default=[], help="Exact private local identifier to replace")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    replacements = [(s, "<LOCAL_PATH>") for s in sorted(args.redact_exact, key=len, reverse=True)]
    upstream_paths = {p.stem: p for p in (args.upstream / "tasks").rglob("TASK-*.json")}
    teacher_scores = {(r["metric"], r["task_id"]): r.get("score")
                      for r in map(json.loads, args.scores.read_text().splitlines())
                      if r.get("model") == "q8" and r.get("condition") == "baseline"}
    references = {m: read(args.study_acb / f"ref_{m}.json") for m in ("rtd", "cpr", "idr")}
    tests = []
    score_maps = {}
    jobs = []
    for metric in ("rtd", "cpr", "idr"):
        score_maps[(metric, "student_original", "run1")] = references[metric]["student"]
        for p in sorted((args.study_acb / f"runs/ref/{metric}_orig_r1").glob("TASK-*.json")):
            jobs.append((metric, "student_original", "run1", {}, "done", p))
        for designer in ("astra", "opus"):
            session = f"{metric}_{designer}"
            events = list(map(json.loads, (args.study_acb / f"ledger/{session}.jsonl").read_text().splitlines()))
            final = next((e["test_id"] for e in events if e.get("event") == "submit"), None)
            for e in events:
                if e.get("event") != "run":
                    continue
                tid = e["test_id"]
                if tid not in {f"t{i:02d}" for i in range(1, 6)}:
                    continue
                done = next((x for x in events if x.get("event") == "done" and x.get("test_id") == tid), None)
                failed = any(x.get("event") == "platform_failure" and x.get("test_id") == tid for x in events)
                status = "done" if done else "platform_failure" if failed else "incomplete_snapshot"
                paths = sorted((args.study_acb / f"runs/{session}/{tid}").glob("TASK-*.json"))
                tests.append({"metric": metric, "session": session, "test": tid, "status": status,
                              "effective_spec": e["effective_spec"], "submitted_at": e.get("submitted_at"),
                              "finished_at": done.get("finished_at") if done else None,
                              "selected_final": tid == final, "available_trace_files": len(paths)})
                score_maps[(metric, session, tid)] = done.get("per_task_scores", {}) if done else {}
                for p in paths:
                    jobs.append((metric, session, tid, e["effective_spec"], status, p))
        for p in sorted((args.teacher_runs / f"{metric}/baseline/q8").glob("TASK-*.json")):
            jobs.append((metric, "teacher_original", "run1", {}, "done", p))
    manifest, baseline_cases, rejected_sources = [], [], []
    scans = []
    source_counts = collections.Counter()
    generated_chars = 0
    for metric, session, test, spec, status, path in jobs:
        raw = path.read_bytes()
        rec = json.loads(raw)
        if "run" not in rec:
            rejected_sources.append({"metric": metric, "session": session, "test": test,
                                     "task_id": rec.get("task_id", path.stem),
                                     "source_sha256": sha(raw), "reason": "no_saved_run"})
            continue
        run = rec["run"]
        task = rec.get("task") or run["scenario"]
        tid = task["task_id"]
        upstream_path = upstream_paths[tid]
        original_task = read(upstream_path)
        turns, stimuli = [], []
        api = iter(run["trace"].get("api_calls", []))
        for index, turn in enumerate(run["trace"]["turns"]):
            if turn.get("synthetic"):
                stimuli.append({"trace_index": index, "agent_id": turn["agent_id"],
                                "role": turn.get("role"), "synthetic": True,
                                "withheld_content_sha256": sha(turn["content"].encode())})
                continue
            content = safe_text(turn["content"], replacements)
            generated_chars += len(content)
            row = {**turn, "trace_index": index, "content": content,
                   "original_content_sha256": sha(turn["content"].encode()),
                   "published_content_sha256": sha(content.encode()), "api_call": next(api, None)}
            turns.append(row)
        relative = Path(f"responses/{metric}/{session}/{test}/{tid}.json.gz")
        score = rec.get("score")
        if score is None:
            score = (teacher_scores.get((metric, tid)) if session == "teacher_original"
                     else score_maps.get((metric, session, test), {}).get(tid))
        data = {
            "schema_version": 1, "metric": metric, "session": session, "test": test, "task_id": tid,
            "test_status": status, "effective_spec": spec, "source_record_sha256": sha(raw),
            "saved_task_sha256": sha(canonical(task)), "upstream_task_sha256": sha(canonical(original_task)),
            "benchmark": {"repository": REPOSITORY, "commit": COMMIT,
                          "path": str(upstream_path.relative_to(args.upstream)),
                          "complete_task_definition_included": False},
            "task_structure": structure(task, metric), "score": score,
            "score_is_saved_not_rejudged": True, "turns": turns, "synthetic_stimuli": stimuli,
            "handoffs": [{"sender": h["sender"], "receiver": h["receiver"],
                           "timestamp": h.get("timestamp"), "content_sha256": sha(h["content"].encode()),
                           "source_turn_candidates": [t["trace_index"] for t in turns
                                                      if t["agent_id"] == h["sender"]
                                                      and t["original_content_sha256"] == sha(h["content"].encode())]}
                          for h in run["trace"].get("handoffs", [])],
            "final_result": safe_text(run["output"].get("result", ""), replacements),
            "observed_timestamps": run["trace"].get("timestamps", {}),
        }
        digest = write(out / relative, data)
        manifest.append({k: data[k] for k in ("metric", "session", "test", "task_id", "test_status",
                                             "source_record_sha256", "saved_task_sha256", "upstream_task_sha256")}
                       | {"path": str(relative), "published_gzip_sha256": digest,
                          "generated_turns": len(turns), "synthetic_stimuli_withheld": len(stimuli),
                          "local_identifier_replacements": sum(t["original_content_sha256"] != t["published_content_sha256"] for t in turns)})
        source_counts[f"{metric}/{session}/{test}"] += 1
        if session == "student_original":
            baseline_cases.append({"metric": metric, "task_id": tid, "path": str(relative),
                                   "saved_task_sha256": data["saved_task_sha256"],
                                   "upstream_path": data["benchmark"]["path"]})
        published_text = "\n".join(t["content"] for t in turns)
        for kind, pat in (("credential_shaped", r"\b(?:github_pat_[A-Za-z0-9_]{15,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})"),
                          ("private_local_path", r"/Users/|/private/tmp/")):
            hits = list(re.finditer(pat, published_text))
            if hits:
                scans.append({"path": str(relative), "pattern": kind, "matches": len(hits),
                              "match_hashes": sorted({sha(h.group().encode()) for h in hits})})
    write(out / "manifest.json", {"schema_version": 1, "scope": "latest six ACB design sessions plus original student and teacher traces",
                                   "counts": dict(sorted(source_counts.items())), "records": manifest,
                                   "unavailable_or_error_records": rejected_sources,
                                   "original_teacher_trace_coverage": {m: sum(r["metric"] == m and r["session"] == "teacher_original" for r in manifest)
                                                                        for m in ("rtd", "cpr", "idr")},
                                   "excluded": ["later q4 base-checkpoint repeats", "later q8 recheck runs", "full benchmark definitions", "third-party evaluator source", "operational host/key logs"]})
    write(out / "tests.json", tests)
    write(out / "audit/baseline_cases.json", baseline_cases)
    write(out / "scan_summary.json", {"generated_characters": generated_chars, "review_flags": scans})
    write(out / "references.json", references)
    (out / "source").mkdir(exist_ok=True)
    source_eval = args.study_acb / "acb_eval.py"
    shutil.copyfile(source_eval, out / "source/acb_eval_snapshot.py")
    daemon_path = args.study_acb / "daemon.py"
    daemon = daemon_path.read_text()
    tree = ast.parse(daemon)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ("FACTORS", "BY_METRIC") for t in n.targets)]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Session")
    check = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    excerpt = "# Exact study-owned validation excerpt; no daemon or remote operations.\n"
    excerpt += "\n\n".join(ast.get_source_segment(daemon, n) for n in assignments)
    excerpt += "\n\nclass DomainSession:\n" + textwrap.indent(ast.get_source_segment(daemon, check), "    ") + "\n"
    (out / "source/saved_domain_validation.py").write_text(excerpt)
    write(out / "source/source_provenance.json", {
        "study_evaluator_sha256": sha(source_eval.read_bytes()),
        "study_daemon_sha256": sha(daemon_path.read_bytes()),
        "upstream_commit": COMMIT,
        "upstream_required_files_sha256": {str(p.relative_to(args.upstream)): sha(p.read_bytes())
                                            for p in sorted((args.upstream / "agentcollabbench").rglob("*.py"))},
        "upstream_pyproject_declares": "MIT", "upstream_license_file_present": False,
        "upstream_source_is_external_dependency_not_bundled": True,
    })
    print(json.dumps({"traces": len(manifest), "generated_turns": sum(r["generated_turns"] for r in manifest),
                      "generated_characters": generated_chars, "scan_flags": len(scans)}))


if __name__ == "__main__":
    main()
