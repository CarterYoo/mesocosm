# Exact study-owned validation excerpt; no daemon or remote operations.
FACTORS = {  # designer-facing name -> (axis, acb_eval spec key, domain, default, meaning)
    "S.topology": ("topology", ["original", "chain", "full", "root_broadcast"], "original",
                   "who receives from whom: the task's own graph, a chain in speaking order, everyone hears "
                   "everyone, or the original graph plus root -> every agent"),
    "S.turns": ("turns", [0.5, 1, 1.5, 2], 1, "conversation length: speaking-order length multiplier"),
    "E.constraint_salience": ("salience", ["high", "low"], "high",
                              "how the tagged constraint is formatted (high: **CRITICAL CONSTRAINT [ID]: ...**)"),
    "E.distractor_load": ("distractors", ["all", "half", "none"], "all",
                          "how many of the task's distractor lines accompany the constraint"),
    "E.false_fact_carrier_position": ("seed_position", ["original", "first", "middle", "last"], "original",
                                      "which agent in the speaking order is given the false fact"),
    "E.peer_pressure_messages": ("pressure", ["all", "half", "first"], "all",
                                 "how many of the task's own peer-pressure messages are delivered"),
}

BY_METRIC = {"rtd": ["S.topology", "S.turns", "E.constraint_salience", "E.distractor_load"],
             "cpr": ["S.topology", "S.turns", "E.false_fact_carrier_position"],
             "idr": ["S.topology", "S.turns", "E.peer_pressure_messages"]}

class DomainSession:
    def check(self, spec):
            if not isinstance(spec, dict) or not isinstance(spec.get("changes"), list):
                return None, "the spec must be {\"changes\": [{\"factor\": ..., \"value\": ...}, ...]}"
            allowed, eval_spec, seen = BY_METRIC[self.metric], {}, set()
            for ch in spec["changes"]:
                if not isinstance(ch, dict) or set(ch) != {"factor", "value"}:
                    return None, "each change must be exactly {\"factor\": ..., \"value\": ...}"
                f, v = ch["factor"], ch["value"]
                if f not in allowed:
                    return None, f"unknown factor {f!r}; allowed: {allowed}"
                if f in seen:
                    return None, f"duplicate factor {f}"
                key, dom, default, _ = FACTORS[f]
                if v not in dom or type(v) is not type(default):
                    return None, f"{f}={v!r} is outside its domain {dom}"
                seen.add(f)
                eval_spec[key] = v
            return eval_spec, None
