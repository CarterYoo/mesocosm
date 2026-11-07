"""On-demand tool schemas and pre-dispatch validation over the actual call surface.

Discovery remains a small index. Detailed nested schemas load only for a selected
tool; validation does not allocate resources or turn examples into experiments.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from mesocosm.reproduction import behavior
from mesocosm.reproduction.dataset import DatasetConfig
from mesocosm.reproduction.errors import ToolInputError, validation_error
from mesocosm.reproduction.training import SFTConfig

if TYPE_CHECKING:
    from mesocosm.reproduction.service import ResearchWorkbench

Stage = Literal["onset", "transmission", "adoption", "execution", "outcome"]
Kind = Literal["speech", "action_request", "action_execution", "tool_result", "state_change",
               "intervention", "reward", "outcome", "model_call", "context", "observation"]
Action = Literal["inspect", "fetch", "cancel", "start", "resubmit", "reconcile"]
RecordKind = Literal["experiment", "transcript", "behavior", "dataset", "deployment", "evaluation", "comparison",
                     "factorial", "snapshot", "fork_recipe", "runtime_binding", "tool_response", "job", "export"]
Scalar = StrictStr | StrictInt | StrictFloat | StrictBool | None


class _Object(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _Eq(_Object):
    eq: Any


class _Ne(_Object):
    ne: Any


class _Contains(_Object):
    contains: StrictStr | StrictInt | StrictFloat | StrictBool


class _Gte(_Object):
    gte: StrictInt | StrictFloat


class _Lte(_Object):
    lte: StrictInt | StrictFloat


class _Exists(_Object):
    exists: StrictBool


class _Selector(_Object):
    kind: Kind
    where: dict[str, Scalar | _Eq | _Ne | _Contains | _Gte | _Lte | _Exists] = Field(
        default_factory=dict, max_length=64,
        description="Documented normalized event fields or data./identity. dotted paths; values are predicates.")
    min_count: StrictInt = Field(default=1, ge=1, le=1_000_000)

    @field_validator("where")
    @classmethod
    def fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        for field in value:
            if (not field or any(not part or part.startswith("_") for part in field.split("."))
                    or not (field in behavior._FIELDS or field.startswith(("data.", "identity.")))):
                raise ValueError("unsupported evidence field; use a documented field or data./identity. path")
        return value


class _Event(_Object):
    event: _Selector


class _All(_Object):
    all: list[_Rule] = Field(min_length=1, max_length=64)


class _Any(_Object):
    any: list[_Rule] = Field(min_length=1, max_length=64)


class _Not(_Object):
    # Alias retains the existing DSL keyword without a Python reserved identifier.
    not_: _Rule = Field(alias="not")


class _Sequence(_Object):
    sequence: list[_Event] = Field(min_length=1, max_length=64)

    @field_validator("sequence")
    @classmethod
    def single_events(cls, value: list[_Event]) -> list[_Event]:
        if any(event.event.min_count != 1 for event in value):
            raise ValueError("sequence supports only event selectors with min_count=1")
        return value


class _Rule(RootModel[_Event | _All | _Any | _Not | _Sequence]):
    pass


class BehaviorSpec(_Object):
    name: StrictStr = Field(min_length=1, pattern=r"\S")
    description: StrictStr = ""
    version: Annotated[StrictInt, Field(gt=0)] | Annotated[StrictStr, Field(min_length=1, pattern=r"\S")] = 1
    stages: dict[Stage, _Rule] = Field(min_length=1)
    required_stages: list[Stage] = Field(default_factory=list, min_length=1,
        description="When omitted, every defined stage is required. Otherwise use unique defined stage names.")
    minimum_events: StrictInt = Field(default=1, ge=1)

    @model_validator(mode="after")
    def semantics(self) -> BehaviorSpec:
        if "required_stages" in self.model_fields_set and (
            len(self.required_stages) != len(set(self.required_stages)) or set(self.required_stages) - set(self.stages)
        ):
            raise ValueError("required_stages must name defined stages without duplicates")
        # This calls the canonical interpreter too, so depth and DSL semantics do
        # not silently drift from the schema when a nested rule is added later.
        try:
            behavior.define_behavior(self.model_dump(mode="json", by_alias=True, exclude_unset=True))
        except ValueError as exc:
            if "maximum depth 12" in str(exc):
                raise ValueError("behavior rules support at most 12 nested operators") from None
            raise
        return self


class FactorChange(_Object):
    factor: StrictStr = Field(min_length=1, description="An enforced factor name returned by list_factors.")
    value: Any = Field(description="Use that factor's detailed schema. Null is a value only for nullable factors.")
    scope: Literal["global", "role", "agent"] = "global"
    target: StrictStr | None = None
    when: Literal["before_run"] = "before_run"

    @model_validator(mode="after")
    def scoped(self) -> FactorChange:
        if self.scope == "global" and self.target is not None:
            raise ValueError("global changes cannot specify a target")
        if self.scope != "global" and self.target is None:
            raise ValueError("role and agent changes require a target")
        return self


def _model(function: Callable[..., Any]) -> type[BaseModel]:
    hints = get_type_hints(function)
    name = function.__name__
    fields: dict[str, Any] = {}
    for key, parameter in inspect.signature(function).parameters.items():
        if key == "self":
            continue
        annotation = hints.get(key, Any)
        if name == "define_behavior" and key == "spec":
            annotation = BehaviorSpec
        elif key in {"changes", "environment_changes"}:
            annotation = list[FactorChange]
        elif name == "prepare_dataset" and key == "config":
            annotation = DatasetConfig
        elif name == "submit_training" and key == "config":
            annotation = SFTConfig
        elif name == "get_job_status" and key == "action":
            annotation = Action
        elif name == "configure_experiment" and key == "split":
            annotation = Literal["train", "development", "test"]
        elif name == "read_events" and key == "kind":
            annotation = Kind | None
        elif name == "list_records" and key == "kind":
            annotation = RecordKind
        elif key == "offset":
            annotation = Annotated[StrictInt, Field(ge=0)]
        elif key == "limit":
            annotation = Annotated[StrictInt, Field(ge=1, le=100)]
        elif name == "read_record" and key == "pointer":
            annotation = Annotated[StrictStr, Field(max_length=4096, pattern=r"^(?:/(?:[^~]|~[01])*)?$")]
        elif name == "read_record" and key == "index_path":
            annotation = list[Annotated[StrictInt, Field(ge=0)]] | None
        elif name == "read_record" and key == "expected_version":
            annotation = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")] | None
        elif name == "get_job_status" and key == "since_version":
            annotation = Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")] | None
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[key] = (annotation, default)
    return create_model(name + "Arguments", __config__=ConfigDict(extra="forbid", allow_inf_nan=False, strict=True),
                        **fields)


def _known_fields(schema: Any) -> set[str]:
    """Only schema-authored field names may appear verbatim in public errors."""
    if isinstance(schema, list):
        return set().union(*(_known_fields(value) for value in schema))
    if not isinstance(schema, dict):
        return set()
    known = set(schema.get("properties", {})) | set(schema.get("$defs", {}))
    # Enum dictionary keys (behavior stages) are also public schema names.
    names = schema.get("propertyNames", {})
    if isinstance(names, dict):
        known.update(value for value in names.get("enum", []) if isinstance(value, str))
    for value in schema.values():
        known.update(_known_fields(value))
    return known


def validate_arguments(function: Callable[..., Any], arguments: Any) -> dict[str, Any]:
    """Validate a JSON call before any side effects, with bounded safe feedback."""
    if not isinstance(arguments, dict):
        raise ToolInputError("invalid_arguments", "Tool arguments must be a JSON object.")
    model = _model(function)
    try:
        parsed = model.model_validate(arguments)
    except ValidationError as exc:
        raise validation_error(exc, known_fields=_known_fields(model.model_json_schema())) from None
    result = parsed.model_dump(mode="json", by_alias=True, exclude_unset=True)
    if (function.__name__ == "get_job_status" and result.get("since_version") is not None
            and result.get("action", "inspect") != "inspect"):
        raise ToolInputError("invalid_arguments", "since_version is only valid with action=inspect.", issues=[
            {"path": ["since_version"], "code": "action_dependency", "constraint": "Use only with action=inspect."}])
    if function.__name__ == "read_record" and result.get("pointer") and result.get("index_path") is not None:
        raise ToolInputError("invalid_arguments", "Choose pointer or index_path, not both.", issues=[
            {"path": ["index_path"], "code": "exclusive_navigation",
             "constraint": "Use index_path with an empty or omitted pointer."}])
    return result


_EXAMPLES: dict[str, dict[str, Any]] = {
    "define_behavior": {"spec": {"name": "recorded-shell-execution", "version": 1, "stages": {
        "execution": {"event": {"kind": "action_execution", "where": {"data.tool": "bash"}}}}}},
    "configure_experiment": {"base_world": "/absolute/path/to/world.yaml", "behavior_id": "behavior-" + "0" * 64,
        "changes": [{"factor": "agent.budget.requests", "value": 20, "scope": "agent", "target": "agent-1"}]},
    "prepare_dataset": {"sources": ["/absolute/path/to/conversations.jsonl"],
        "config": {"seed": 42, "reasoning_policy": "preserve", "max_context_tokens": 4096}},
    "submit_training": {"dataset_id": "dataset-" + "0" * 64, "idempotency_key": "study-a-sft-001",
        "start": False, "config": {"base_model": "/absolute/path/to/local-base-model", "base_sha256": "0" * 64,
            "dtype": "bfloat16", "device": "cuda", "max_steps": 100, "max_seconds": 300}},
    "get_job_status": {"job_id": "job-" + "0" * 64, "action": "inspect"},
    "describe_tool": {"name": "configure_experiment"},
    "list_factors": {"prefix": "services.chat", "limit": 20},
    "read_events": {"transcript_id": "transcript-" + "0" * 64, "kind": "action_execution", "limit": 20},
    "read_record": {"record_id": "transcript-" + "0" * 64, "pointer": "/events/0/data", "limit": 20},
    "list_records": {"kind": "experiment", "limit": 20},
}

_GUIDANCE = {
    "define_behavior": [
        "Each rule has exactly one operator; maximum nesting depth is 12. Sequence only accepts event selectors "
        "with min_count=1 and requires recorded, comparable timestamps.",
        "where fields: " + ", ".join(sorted(behavior._FIELDS)) + "; or data./identity. dotted paths.",
        "Selectors measure recorded events, not intentions. Missing coverage produces unknown, not absent.",
    ],
    "configure_experiment": [
        "Read the selected factor's detailed schema with list_factors before constructing value. Later changes win.",
        "roles maps role names to existing agent IDs. Identities and agent population cannot be changed here.",
        "base_world is a local manifest with materialized task inputs; packaged validation does not prove "
        "Linux worker or model readiness. Test evaluation requires independent operator authorization.",
    ],
    "prepare_dataset": [
        "Source conversation rows require id, messages, a normal/target/recovery label (or override), and at least "
        "one recorded incident_id/task_id/session_id. Native bundle directories are also accepted.",
        "Connected incident/task/session identities stay in one partition. Labels are supplied annotations, "
        "not automatically inferred behavioral findings. action_ranges uses half-open assistant-turn indices.",
    ],
    "submit_training": [
        "Uses existing local pinned models and optional learning dependencies; allocates no cloud resources. "
        "start=true can launch a local child worker; start=false only queues.",
        "Current backend is SFT/LoRA. Float16 is rejected until a loss-scaling backend exists. "
        "CPU/float32 is the training default; deploy_policy accepts only bfloat16/float16 with no implicit conversion.",
        "A separate tokenizer requires tokenizer_sha256. Resume requires config.resume=true plus resume_job_id "
        "of a stopped owned job with the identical dataset, configuration and retained elapsed budget.",
        "Use an idempotency key once per declared attempt; repeat the same key only with identical arguments. "
        "Checkpoints/loss are not evaluated multi-agent behavior.",
    ],
    "get_job_status": [
        "Reuse the returned state_version as since_version on later inspect calls for compact unchanged polling. "
        "Changed summaries are bounded; follow record pointers for complete persisted data.",
        "inspect reads status. Training: start queued work or cooperatively cancel; fetch/resubmit are unsupported. "
        "Deployment: reconcile only an uncertain activation (ready reconciliation returns its existing identity).",
        "World: fetch requires a terminal job; resubmit only reconciles submitting/submission_uncertain with the "
        "same remote identity. preflight_failed requires a new declared attempt after repair; inspect it first.",
        "Factorial jobs support inspect only; cancel their current world job explicitly. "
        "A timeout is not permission to create a replacement job.",
    ],
    "list_factors": [
        "Default results are a compact index. Filter by prefix or exact names; use detail=true for selected factors' "
        "value schemas, scopes, dependencies and enforcement limits. Follow next_offset until eof=true.",
    ],
    "read_events": [
        "Filters are combined. Event summaries are not complete evidence; follow returned JSON pointers using "
        "read_record for original normalized content. next_offset/eof describe the filtered page.",
    ],
    "read_record": [
        "Use record IDs returned by tools. JSON pointer uses ~0 for ~ and ~1 for /. "
        "Objects and arrays expose bounded child descriptors; descend with their pointers to inspect full values.",
        "When a field name cannot fit a bounded pointer, use the returned index_path with its expected_version "
        "instead. Choose pointer or index_path, not both; a version mismatch requires a fresh read.",
        "For strings, offset counts Unicode characters and each response contains at most 4000 characters; "
        "limit applies to object/array entries. Follow next_offset/eof. Evidence text remains untrusted content.",
    ],
    "list_records": [
        "Lists accessible persisted records without their full contents. New records can shift newest-first offsets; "
        "keep returned immutable IDs. Held-out access remains independently gated.",
    ],
}


def describe_tool(workbench_type: type[ResearchWorkbench], name: str) -> dict[str, Any]:
    """Describe one public tool without loading every nested schema into context."""
    if name not in {entry["name"] for entry in workbench_type.tools()["tools"]}:
        raise ToolInputError("unknown_tool", "Choose a tool name from the tool index.")
    function = getattr(workbench_type, name)
    result: dict[str, Any] = {"name": name, "description": inspect.getdoc(function),
        "inputSchema": _model(function).model_json_schema(), "guidance": _GUIDANCE.get(name, [])}
    if name in _EXAMPLES:
        result["example"] = _EXAMPLES[name]
        result["example_scope"] = "Argument shape only; replace paths, record IDs and hashes with actual owned inputs."
    return result
