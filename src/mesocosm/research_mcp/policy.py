"""Opaque policy choices and bounded access to the real local SFT/serving lifecycle."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from mesocosm.execution.client import worker_url
from mesocosm.learning.config import PolicyIdentity
from mesocosm.reproduction.dataset import load_dataset
from mesocosm.reproduction.deployment import training_artifact
from mesocosm.reproduction.errors import ToolInputError
from mesocosm.reproduction.service import ResearchWorkbench
from mesocosm.reproduction.store import Store, canonical, digest
from mesocosm.reproduction.training import SFTConfig, training_capabilities

_HANDLE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_REQUEST = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TRAINING_PARAMETERS = frozenset({"learning_rate", "batch_size", "epochs", "lora_rank", "lora_alpha", "seed",
                                "max_grad_norm", "weight_decay", "scheduler"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PolicyVenue(_Strict):
    server: str
    token_env: str | None = None
    base_model: str
    base_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dtype: Literal["bfloat16"] = "bfloat16"


class TrainingRecipe(_Strict):
    dataset_id: str = Field(pattern=r"^dataset-[a-f0-9]{64}$")
    config: dict[str, Any]
    parameter_choices: dict[str, list[Any]] = Field(default_factory=dict)
    venue: PolicyVenue | None = None

    def configuration(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameters, dict) or set(parameters) - set(self.parameter_choices):
            _fail("invalid_policy_parameters", "Choose only advertised training parameters.", "parameters")
        for name, value in parameters.items():
            if canonical(value) not in {canonical(choice) for choice in self.parameter_choices[name]}:
                _fail("invalid_policy_parameters", "Choose an exact advertised parameter value.", "parameters")
        return SFTConfig.model_validate({**self.config, **parameters}).model_dump(mode="json")


class PolicySpace(_Strict):
    deployments: dict[str, str] = Field(default_factory=dict, max_length=64)
    recipes: dict[str, TrainingRecipe] = Field(default_factory=dict, max_length=32)
    max_training_jobs: int = Field(default=1, ge=1, le=32)
    max_total_steps: int = Field(default=100, ge=1, le=100_000)
    max_total_train_tokens: int = Field(default=1_000_000, ge=1, le=1_000_000_000)
    max_total_eval_tokens: int = Field(default=1_000_000, ge=1, le=1_000_000_000)
    max_total_seconds: float = Field(default=3600, gt=0, le=604_800)

    def allowed_servers(self, workbench: ResearchWorkbench) -> dict[str, str | None]:
        servers: dict[str, str | None] = {}
        venues = [(value.server, value.token_env) for recipe in self.recipes.values()
                  if (value := recipe.venue) is not None]
        venues.extend((value["server"], value["server_token_env"])
                      for identifier in self.deployments.values()
                      for value in [workbench.store.get(identifier, "deployment")])
        for endpoint, variable in venues:
            endpoint = worker_url(endpoint)
            if variable is not None and not re.fullmatch(r"[A-Z_][A-Z0-9_]*", variable):
                raise ValueError("invalid policy venue credential variable")
            if endpoint in servers and servers[endpoint] != variable:
                raise ValueError("one policy venue cannot have conflicting credentials")
            servers[endpoint] = variable
        return servers

    def checked(self, workbench: ResearchWorkbench) -> PolicySpace:
        for handle in [*self.deployments, *self.recipes]:
            if not _HANDLE.fullmatch(handle):
                raise ValueError("policy and recipe handles require 1 to 64 public identifier characters")
        self.allowed_servers(workbench)
        for identifier in self.deployments.values():
            _deployment(workbench, identifier)
        for recipe in self.recipes.values():
            spec = SFTConfig.model_validate(recipe.config)
            if spec.resume or spec.tokenizer is not None:
                raise ValueError("designer recipes cannot resume or replace the tokenizer")
            if not Path(spec.base_model).expanduser().is_dir():
                raise ValueError("designer recipes require an already materialized local base")
            if recipe.venue and (spec.dtype != recipe.venue.dtype or spec.base_sha256 != recipe.venue.base_sha256):
                raise ValueError("training and serving must use the same base hash and dtype")
            manifest, _ = load_dataset(workbench.store.directory("datasets", recipe.dataset_id))
            if manifest != workbench.store.get(recipe.dataset_id, "dataset"):
                raise ValueError("recipe dataset identity changed")
            if set(recipe.parameter_choices) - TRAINING_PARAMETERS:
                raise ValueError("recipe choices cannot change model, paths, compute budgets or execution settings")
            if len(canonical(recipe.parameter_choices).encode()) > 4000:
                raise ValueError("recipe parameter domains exceed their public response bound")
            for name, choices in recipe.parameter_choices.items():
                if not 1 <= len(choices) <= 16 or any(type(value) not in {str, int, float} for value in choices):
                    raise ValueError("recipe parameters require 1 to 16 finite scalar choices")
                if len({canonical(value) for value in choices}) != len(choices):
                    raise ValueError("recipe parameter choices must be distinct")
                for value in choices:
                    SFTConfig.model_validate({**recipe.config, name: value})
        return self


def _fail(code: str, message: str, field: str) -> NoReturn:
    raise ToolInputError(code, message, issues=[{"field": field, "code": code, "message": message}])


def _deployment(workbench: ResearchWorkbench, identifier: str) -> dict[str, Any]:
    value = workbench.store.get(identifier, "deployment")
    PolicyIdentity.model_validate(value["policy"])
    job = workbench.store.job(value["deployment_job_id"])
    if job.get("kind") != "deployment" or job.get("status") != "ready" or job.get("deployment_id") != identifier:
        raise ValueError("policy deployment is not ready")
    return value


def _compatibility(value: dict[str, Any]) -> tuple[Any, ...]:
    return (worker_url(value["server"]), *(value["policy"][key] for key in
            ("base_model", "base_revision", "base_sha256", "dtype")))


def _group(value: dict[str, Any]) -> str:
    return "group-" + digest(_compatibility(value))


class PolicyManager:
    def __init__(self, space: PolicySpace, store: Store, workbench: ResearchWorkbench,
                 session_id: str, context_sha256: str):
        self.space, self.store, self.workbench = space, store, workbench
        self.session_id, self.context_sha256 = session_id, context_sha256

    def _owned(self, identifier: str) -> dict[str, Any]:
        if not isinstance(identifier, str) or not re.fullmatch(r"policy_job-[a-f0-9]{64}", identifier):
            _fail("unknown_policy_job", "Choose a policy job returned in this session.", "policy_job_id")
        try:
            value = self.store.job(identifier)
        except (KeyError, ValueError):
            _fail("unknown_policy_job", "Choose a policy job returned in this session.", "policy_job_id")
        if (value.get("kind") != "designer_policy" or value.get("session_id") != self.session_id
                or value.get("context_sha256") != self.context_sha256):
            _fail("unknown_policy_job", "Choose a policy job returned in this session.", "policy_job_id")
        return value

    def _resolve(self, identifier: str, field: str) -> tuple[str, dict[str, Any]]:
        if identifier in self.space.deployments:
            native = self.space.deployments[identifier]
        elif isinstance(identifier, str) and re.fullmatch(r"policy-[a-f0-9]{64}", identifier):
            try:
                owned = self.store.get(identifier, "policy")
            except (KeyError, ValueError):
                _fail("unknown_policy", "Choose a ready policy advertised in this session.", field)
            if owned.get("session_id") != self.session_id or owned.get("context_sha256") != self.context_sha256:
                _fail("unknown_policy", "Choose a ready policy advertised in this session.", field)
            native = owned["deployment_id"]
        else:
            _fail("unknown_policy", "Choose a ready policy advertised in this session.", field)
        try:
            return native, _deployment(self.workbench, native)
        except (KeyError, ValueError):
            _fail("policy_not_ready", "The selected policy is no longer ready; ask the operator to inspect it.", field)
        raise AssertionError("unreachable")

    def catalog(self, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 10:
            _fail("invalid_page", "Use offset >= 0 and limit between 1 and 10.", "offset")
        identifiers = set(self.space.deployments)
        with self.store.connection() as db:
            for identifier, raw in db.execute("SELECT id,body FROM objects WHERE kind='policy'"):
                value = json.loads(raw)
                if value.get("session_id") == self.session_id and value.get("context_sha256") == self.context_sha256:
                    identifiers.add(identifier)
        entries = [("policy", identifier) for identifier in sorted(identifiers)]
        entries += [("recipe", identifier) for identifier in sorted(self.space.recipes)]
        policies: list[dict[str, Any]] = []
        recipes: list[dict[str, Any]] = []
        count = 0
        for kind, identifier in entries[offset:offset + limit]:
            entry: dict[str, Any]
            if kind == "policy":
                _, deployed = self._resolve(identifier, "policy_id")
                entry = {"policy_id": identifier, "compatibility_group": _group(deployed)}
                page = policies
            else:
                recipe = self.space.recipes[identifier]
                config = SFTConfig.model_validate(recipe.config)
                entry = {"recipe_id": identifier, "parameter_choices": recipe.parameter_choices,
                         "serving_configured": recipe.venue is not None,
                         "training_budget": {name: getattr(config, name) for name in
                                             ("max_steps", "max_train_tokens", "max_eval_tokens", "max_seconds")}}
                page = recipes
            page.append(entry)
            if len(canonical({"policies": policies, "recipes": recipes}).encode()) > 18_000:
                page.pop()
                break
            count += 1
        end = offset + count
        limits = {"jobs": self.space.max_training_jobs, "steps": self.space.max_total_steps,
                  "train_tokens": self.space.max_total_train_tokens, "eval_tokens": self.space.max_total_eval_tokens,
                  "seconds": self.space.max_total_seconds}
        with self.store.connection() as db:
            prior = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
        prior = [value for value in prior if value.get("kind") == "designer_policy"
                 and value.get("session_id") == self.session_id]
        reserved = {name: sum(value["reservation"][name] for value in prior) for name in limits if name != "jobs"}
        reserved["jobs"] = len(prior)
        remaining = {name: max(0, amount - reserved[name]) for name, amount in limits.items()}
        backend = training_capabilities()
        return {"policies": policies, "recipes": recipes, "total": len(entries),
                "next_offset": end if end < len(entries) else None,
                "training_budget": {"limits": limits, "reserved": reserved, "remaining": remaining},
                "training_backend": {name: backend[name] for name in
                                     ("backend", "available", "missing_packages", "methods")},
                "policy_updates": "real_local_sft_lora", "automatic_cloud_allocation": False}

    def bindings(self, policy_id: str | None, agent_policies: dict[str, str] | None,
                 agent_ids: list[str]) -> dict[str, Any]:
        if policy_id is None and agent_policies is None:
            return {}
        if policy_id is not None and agent_policies is not None:
            _fail("conflicting_policy_selection", "Choose shared policy_id or agent_policies, once.", "policy_id")
        if policy_id is not None:
            choices = {agent: policy_id for agent in agent_ids}
        else:
            assert agent_policies is not None
            slots = {f"agent-{index}": agent for index, agent in enumerate(agent_ids, 1)}
            if set(agent_policies) != set(slots):
                _fail("incomplete_population", "Assign one ready policy to every advertised agent slot.",
                      "agent_policies")
            choices = {slots[slot]: identifier for slot, identifier in agent_policies.items()}
        resolved = {agent: self._resolve(identifier, "policy_id" if policy_id else "agent_policies")
                    for agent, identifier in choices.items()}
        if len({_compatibility(value) for _, value in resolved.values()}) != 1:
            _fail("incompatible_population", "Mixed policies must have the same advertised compatibility group.",
                  "agent_policies")
        return {"agent_deployments": {agent: identifier for agent, (identifier, _) in resolved.items()}}

    def train_policy(self, recipe_id: str, parameters: dict[str, Any], request_id: str) -> dict[str, Any]:
        if not isinstance(recipe_id, str) or recipe_id not in self.space.recipes:
            _fail("unknown_recipe", "Choose a recipe advertised in this session.", "recipe_id")
        if not isinstance(request_id, str) or not _REQUEST.fullmatch(request_id):
            _fail("invalid_request_id", "Use 1 to 64 letters, numbers, underscores or hyphens.", "request_id")
        recipe = self.space.recipes[recipe_id]
        config = recipe.configuration(parameters)
        arguments = {"recipe_id": recipe_id, "config": config, "context_sha256": self.context_sha256}
        key = "designer-training:" + self.session_id + ":" + request_id
        fingerprint = digest(arguments)
        identifier = "policy_job-" + digest({"session_id": self.session_id, "request_id": request_id,
                                               "context_sha256": self.context_sha256})
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest FROM requests WHERE key=?", (key,)).fetchone()
            if row:
                if row[0] != fingerprint:
                    _fail("request_id_reused", "This request_id already names different training choices.",
                          "request_id")
                saved = db.execute("SELECT body FROM jobs WHERE id=?", (identifier,)).fetchone()
                return self._public(json.loads(saved[0]))
            prior = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
            prior = [value for value in prior if value.get("kind") == "designer_policy"
                     and value.get("session_id") == self.session_id]
            reservation = {"steps": config["max_steps"], "train_tokens": config["max_train_tokens"],
                           "eval_tokens": config["max_eval_tokens"], "seconds": config["max_seconds"]}
            ceilings = {"steps": self.space.max_total_steps, "train_tokens": self.space.max_total_train_tokens,
                        "eval_tokens": self.space.max_total_eval_tokens, "seconds": self.space.max_total_seconds}
            if len(prior) >= self.space.max_training_jobs or any(
                sum(value["reservation"][name] for value in prior) + amount > ceilings[name]
                for name, amount in reservation.items()
            ):
                _fail("training_budget_exhausted", "This session's reserved training budget is exhausted.", "recipe_id")
            value: dict[str, Any] = {"kind": "designer_policy", "session_id": self.session_id,
                     "context_sha256": self.context_sha256, "recipe_id": recipe_id, "config": config,
                     "status": "submission_uncertain", "reservation": reservation,
                     "native_key": "designer-sft:" + digest({"session": self.session_id, "request": request_id,
                                                              "context": self.context_sha256}), "job_id": identifier}
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,NULL)",
                       (key, fingerprint, "train_policy", "completed", canonical({"policy_job_id": identifier})))
            db.execute("INSERT INTO jobs VALUES (?,?)", (identifier, canonical(value)))
        try:
            # Reuse the native queue, pinned dataset and real optimizer implementation.
            queued = self.workbench.submit_training(recipe.dataset_id, config, value["native_key"], start=True)
            value.update(training_job_id=queued["job_id"], status=queued["status"])
        except Exception as exc:
            value["error_type"] = type(exc).__name__
        self.store.save_job(identifier, value)
        return self._public(value)

    def _recover(self, value: dict[str, Any], *, activation: bool = False) -> dict[str, Any]:
        target = "activation_job_id" if activation else "training_job_id"
        key = value.get("activation_key" if activation else "native_key")
        if target not in value and key:
            with self.workbench.store.connection() as db:
                row = db.execute("SELECT result FROM requests WHERE key=? AND status='completed'", (key,)).fetchone()
            if row and row[0]:
                result = json.loads(row[0])
                if isinstance(result, dict) and result.get("job_id"):
                    value[target] = result["job_id"]
                    self.store.save_job(value["job_id"], value)
            if activation and target not in value:
                # Native deployment admission atomically persists its job before
                # loading. Recover that unique owned alias without another load.
                with self.workbench.store.connection() as db:
                    jobs = [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs")]
                owned = [job for job in jobs if job.get("kind") == "deployment"
                         and (job.get("policy") or {}).get("model") == self._alias(value["job_id"])
                         and (job.get("training_artifact") or {}).get("training_job_id")
                         == value.get("training_job_id")]
                if len(owned) == 1:
                    value[target] = owned[0]["job_id"]
                    self.store.save_job(value["job_id"], value)
        return value

    @staticmethod
    def _alias(identifier: str) -> str:
        return "policy-" + digest({"job": identifier})[:24]

    def _public(self, value: dict[str, Any]) -> dict[str, Any]:
        result = {"policy_job_id": value["job_id"], "status": value["status"],
                  "automatic_cloud_allocation": False, "behavioral_improvement": "unverified"}
        if value.get("policy_id"):
            result["policy_id"] = value["policy_id"]
        measured = value.get("measured", {})
        if measured:
            result["measured"] = measured
        return result

    async def policy_status(self, policy_job_id: str, action: str = "inspect") -> dict[str, Any]:
        if action not in {"inspect", "cancel"}:
            _fail("invalid_policy_action", "Choose inspect or cancel.", "action")
        value = self._recover(self._owned(policy_job_id))
        if action == "cancel" and value.get("activation_key"):
            _fail("activation_already_started", "Cancellation stops training; this policy has entered activation.",
                  "action")
        if value.get("training_job_id"):
            await self.workbench.get_job_status(value["training_job_id"], action=action)
            native = self.workbench.store.job(value["training_job_id"])
            if not value.get("activation_key"):
                value["status"] = native["status"]
            measured = native.get("result") or {}
            value["measured"] = {name: measured[name] for name in
                                 ("optimizer_steps", "train_tokens", "eval_tokens_completed", "elapsed_seconds")
                                 if type(measured.get(name)) in {int, float} and math.isfinite(measured[name])}
            self.store.save_job(policy_job_id, value)
        return self._public(value)

    async def activate_policy(self, policy_job_id: str, reconcile: bool = False) -> dict[str, Any]:
        if type(reconcile) is not bool:
            _fail("invalid_policy_action", "reconcile must be a boolean.", "reconcile")
        with self.store.lease("designer-activation:" + policy_job_id):
            value = self._recover(self._owned(policy_job_id))
            recipe = self.space.recipes[value["recipe_id"]]
            venue = recipe.venue
            if venue is None:
                _fail("serving_not_configured", "This recipe has no approved serving venue.", "policy_job_id")
            if not value.get("training_job_id"):
                _fail("training_not_completed", "Activation requires a completed training job.", "policy_job_id")
            try:
                artifact = training_artifact(self.workbench.store, value["training_job_id"])
            except (OSError, KeyError, ValueError):
                _fail("training_not_completed", "Activation requires intact, completed native SFT output.",
                      "policy_job_id")
            if artifact["config"] != {key: item for key, item in value["config"].items() if key != "resume"}:
                _fail("training_identity_changed", "The trained artifact differs from this policy request.",
                      "policy_job_id")
            value = self._recover(value, activation=True)
            if value.get("policy_id"):
                self._resolve(value["policy_id"], "policy_job_id")
                return self._public(value)
            if value.get("activation_key"):
                if not reconcile or not value.get("activation_job_id"):
                    _fail("activation_uncertain", "Use explicit reconciliation; a policy is never loaded twice.",
                          "policy_job_id")
                result = await self.workbench.get_job_status(value["activation_job_id"], action="reconcile")
            else:
                if reconcile:
                    _fail("activation_not_started", "There is no activation to reconcile.", "policy_job_id")
                value.update(status="activation_uncertain", activation_key="designer-activate:" + digest({
                    "job": policy_job_id, "context": self.context_sha256}))
                self.store.save_job(policy_job_id, value)
                try:
                    result = await self.workbench.deploy_policy(
                        server=venue.server, model=self._alias(policy_job_id),
                        base_model=venue.base_model, base_revision=venue.base_revision,
                        base_sha256=venue.base_sha256, dtype=venue.dtype,
                        idempotency_key=value["activation_key"], training_job_id=value["training_job_id"])
                except Exception as exc:
                    value["error_type"] = type(exc).__name__
                    self.store.save_job(policy_job_id, value)
                    return self._public(value)
            value.update(activation_job_id=result["job_id"], status=result["status"])
            if result["status"] == "ready":
                _deployment(self.workbench, result["deployment_id"])
                value["policy_id"] = self.store.put("policy", {"session_id": self.session_id,
                    "context_sha256": self.context_sha256, "deployment_id": result["deployment_id"],
                    "policy_job_id": policy_job_id})
            self.store.save_job(policy_job_id, value)
            return self._public(value)
