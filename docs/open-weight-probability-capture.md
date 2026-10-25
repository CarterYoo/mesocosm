# Open-weight probability capture and native tool permissions

Set `observe.token_logprobs: true` and `observe.top_logprobs: 20` to request chosen-token
probabilities, up to 20 alternatives, and token IDs from every assigned vLLM model.
This opt-in does not change temperature or top-p. RL-bound requests preserve their
pre-existing sampling contract, including `top_logprobs: 0`.

The gateway retains bounded byte-exact forwarded request and response bodies, body
hashes, transport completion, and HTTP delivery-chunk timestamps. Those timestamps
are gateway receive times, not token-generation times. Chosen logprob scalars remain
in the normalized response; alternative probabilities, token strings and token bytes
are retained in the raw body even when normalization does not expose them separately.
Actual serving runs must verify token/logprob alignment and alternative counts; CPU
transport fixtures are not model observations. Abrupt gateway process death can lose
an in-flight capture before finalization.

The default cap is **8 MiB for each request/encoded response/decoded response body**
and **65,536 HTTP delivery chunks**. Compose records the exact
`MESOCOSM_CAPTURE_MAX_BYTES` and `MESOCOSM_CAPTURE_MAX_CHUNKS` values. A capture-limit
failure remains incomplete evidence; it is not accepted as a complete truncated
response. Base64 and normalized records add storage overhead, so these per-request
limits do not enforce an aggregate experiment storage budget. Operators must measure
aggregate trace bytes separately.

OpenCode 1.19.3 converts deprecated boolean `tools` settings into permission rules,
then merges explicit permissions afterward. A later wildcard allow can override the
earlier denies. Mesocosm therefore emits a permission map with the default wildcard
first and explicit tool decisions afterward, without emitting the deprecated `tools`
field. This avoids its merge changing permission-key order. `write`, `edit`, and `patch` use the native `edit` permission group.
Default behavior without tool overrides remains unchanged. Gateway tool advertisements
and actual permission enforcement must be checked in a fresh native run.

This behavior was checked against the immutable upstream source
[`e3c4eb242585dcd77be1a867391b7c25fd47ce52`](https://github.com/anomalyco/opencode/tree/e3c4eb242585dcd77be1a867391b7c25fd47ce52):
[`config/config.ts`](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/opencode/src/config/config.ts),
[`config/permission.ts`](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/core/src/v1/config/permission.ts),
and [`permission/index.ts`](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/packages/opencode/src/permission/index.ts).
