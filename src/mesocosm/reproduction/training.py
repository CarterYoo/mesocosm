"""Local, bounded SFT/LoRA with assistant-only targets and durable optimizer resume.

This updates actual policy parameters. It neither allocates compute nor downloads models,
and a completed update is not evidence of behavioral improvement or native acceptance.
"""

from __future__ import annotations

import fcntl
import importlib
import importlib.util
import json
import math
import pathlib
import random
import shutil
import time
from importlib.metadata import version
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mesocosm.learning.checkpoints import checkpoint_sha256, write_json
from mesocosm.reproduction.dataset import digest, file_sha256, load_dataset


class SFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    backend: Literal["sft_lora"] = "sft_lora"
    base_model: str
    base_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokenizer: str | None = None
    tokenizer_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    learning_rate: float = Field(default=1e-4, gt=0, le=1)
    batch_size: int = Field(default=1, ge=1)
    epochs: int = Field(default=1, ge=1)
    max_steps: int = Field(default=100, ge=1)
    max_seconds: float = Field(default=3600, gt=0)
    max_train_tokens: int = Field(default=1_000_000, ge=1)
    max_eval_tokens: int = Field(default=1_000_000, ge=1)
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"], min_length=1)
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    device: Literal["cpu", "cuda"] = "cpu"
    seed: int = Field(default=0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    scheduler: Literal["constant", "linear"] = "constant"
    resume: bool = False

    @model_validator(mode="after")
    def supported(self) -> SFTConfig:
        if self.dtype == "float16":
            raise ValueError("float16 SFT requires a loss-scaling backend; use float32 or bfloat16")
        if self.tokenizer is not None and self.tokenizer_sha256 is None:
            raise ValueError("a separate tokenizer requires tokenizer_sha256")
        return self


def training_capabilities() -> dict[str, Any]:
    missing = [name for name in ("torch", "transformers", "peft") if importlib.util.find_spec(name) is None]
    return {"backend": "sft_lora", "status": "supported", "available": not missing, "missing_packages": missing,
            "evidence": ["tests/test_reproduction_training.py"],
            "methods": {"sft_lora": "supported", "qlora": "unsupported", "full_finetuning": "unsupported",
                        "preference_learning": "unsupported", "multi_agent_joint_optimization": "unsupported"},
            "model_sources": "local_pinned_causal_lm", "loss_mask": "latest_assistant_action_only",
            "resume": ["adapter", "optimizer", "scheduler", "python_rng", "torch_rng", "cuda_rng", "data_cursor"],
            "agent_selection": "dataset_agent_ids", "policy_sharing": "one_adapter_per_training_job",
            "existing_rl": "mesocosm.learning REINFORCE uses its separate verified-rollout contract",
            "behavioral_improvement": "unverified", "native_multiagent_acceptance": "unverified"}


def training_source_pins() -> dict[str, str]:
    """Pin the implementation that interprets data and commits optimizer updates."""
    package = pathlib.Path(__file__).resolve().parent.parent
    return {name: file_sha256(package / name) for name in (
        "reproduction/training.py", "reproduction/dataset.py", "learning/checkpoints.py")}


def tokenizer_sha256(path: pathlib.Path) -> str:
    files = sorted(file for file in path.rglob("*") if file.is_file() and ".cache" not in file.relative_to(path).parts)
    return digest({file.relative_to(path).as_posix(): file_sha256(file) for file in files})


def _encode(tokenizer: Any, example: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    observations, action = example["observation"], example["action"]
    template = tokenizer.get_chat_template(tools=example.get("tools"))
    if example.get("tools") and "tools" not in template:
        raise ValueError("chat template does not serialize the recorded tool schemas")
    if any(message.get("tool_calls") or message.get("role") == "tool" for message in [*observations, action]):
        if "tool_calls" not in template or "tool" not in template:
            raise ValueError("chat template does not support recorded tool calls/results")
    if config["reasoning_policy"] == "preserve" and any(
        message.get("reasoning") or message.get("reasoning_content") for message in [*observations, action]
    ) and not any(name in template for name in ("reasoning_content", "reasoning")):
        raise ValueError("reasoning retention requested but chat template does not serialize reasoning")
    options = {"tokenize": True, "return_dict": False, "tools": example.get("tools")}
    rendered = tokenizer.apply_chat_template([*observations, action], tokenize=False, tools=example.get("tools"),
                                             add_generation_prompt=False)
    required_text = []
    for message in [*observations, action]:
        if message.get("role") == "tool" and message.get("content"):
            required_text.append(message["content"])
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            required_text.extend([function.get("name"), function.get("arguments")])
    for tool in example.get("tools") or []:
        function = tool.get("function") or {}
        required_text.extend([function.get("name"), function.get("description")])
        required_text.extend((function.get("parameters") or {}).get("properties", {}))
    if any(not isinstance(value, str) or value not in rendered for value in required_text if value):
        raise ValueError("chat template omitted or transformed recorded tool schemas/actions/results")
    if config["reasoning_policy"] == "preserve":
        for message in [*observations, action]:
            for key in ("reasoning", "reasoning_content"):
                if message.get(key) and message[key] not in rendered:
                    raise ValueError("chat template dropped recorded reasoning despite retention policy")
    prefix = tokenizer.apply_chat_template(observations, add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template([*observations, action], add_generation_prompt=False, **options)
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("chat template has no exact observation/action token boundary")
    if len(full) > config["max_context_tokens"]:
        if config["overflow"] == "drop":
            return None
        raise ValueError(f"context limit exceeded for {example['id']}; no silent truncation")
    return {"id": example["id"], "input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):],
            "action_tokens": len(full) - len(prefix)}


def _batch_loss(model: Any, rows: list[dict[str, Any]], torch: Any, device: str) -> tuple[Any, int]:
    # Separate forwards avoid changing attention semantics through batch padding. The
    # loss is an action-token mean across the configured effective batch.
    tokens = sum(row["action_tokens"] for row in rows)
    total = None
    for row in rows:
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor(row["labels"][1:], dtype=torch.long, device=device)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction="sum") / tokens
        total = loss if total is None else total + loss
    return total, tokens


def _evaluate(model: Any, rows: list[dict[str, Any]], torch: Any, device: str) -> float:
    model.eval()
    value, tokens = 0.0, 0
    with torch.no_grad():
        for row in rows:
            loss, count = _batch_loss(model, [row], torch, device)
            value += float(loss.item()) * count
            tokens += count
    model.train()
    result = value / tokens
    if not math.isfinite(result):
        raise ValueError("nonfinite development loss")
    return result


def _checkpoint(root: pathlib.Path, model: Any, optimizer: Any, scheduler: Any, state: dict[str, Any],
                torch: Any) -> pathlib.Path:
    name = f"checkpoint-{state['step']:08d}"
    target, temporary = root / name, root / f".{name}.tmp"
    if target.exists():
        raise FileExistsError("refusing to overwrite a committed optimizer step")
    if temporary.exists():
        shutil.rmtree(temporary)  # Only this job's incomplete checkpoint staging directory.
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               temporary / "state.pt")
    payload = {**state, "files": {path.relative_to(temporary).as_posix(): file_sha256(path)
                                  for path in temporary.rglob("*") if path.is_file()}}
    write_json(temporary / "checkpoint.json", payload)
    temporary.rename(target)
    return target


def _read_checkpoint(path: pathlib.Path, identity: str) -> dict[str, Any]:
    state = json.loads((path / "checkpoint.json").read_text())
    if state["identity"] != identity:
        raise ValueError("checkpoint dataset/base/settings identity mismatch")
    for name, expected in state["files"].items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path.resolve()) or file_sha256(artifact) != expected:
            raise ValueError("checkpoint artifact integrity mismatch")
    return state


def run_training(dataset: pathlib.Path, output: pathlib.Path, config: dict[str, Any], *,
                 cancel_path: pathlib.Path | None = None, elapsed_floor_seconds: float = 0.) -> dict[str, Any]:
    """Run/resume a bounded real SFT update under an exclusive process lock.

    A resumed job has the same total token/step/time limits, not a fresh allocation.
    Committed step directories are authoritative if termination preceded the receipt.
    Parent process supervision provides hard wall-clock cancellation between checkpoints.
    """
    spec = SFTConfig.model_validate(config)
    if not math.isfinite(elapsed_floor_seconds) or elapsed_floor_seconds < 0:
        raise ValueError("elapsed_floor_seconds must be a nonnegative measured duration")
    output = pathlib.Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("training job already running") from exc
        return _run(dataset, output, spec, cancel_path, elapsed_floor_seconds)


def _run(dataset: pathlib.Path, output: pathlib.Path, spec: SFTConfig,
         cancel_path: pathlib.Path | None = None, elapsed_floor_seconds: float = 0.) -> dict[str, Any]:
    started = time.monotonic()
    manifest, examples = load_dataset(dataset)
    base = pathlib.Path(spec.base_model).expanduser().resolve()
    if not base.is_dir() or checkpoint_sha256(base) != spec.base_sha256:
        raise ValueError("local base checkpoint missing or content hash mismatch")
    tokenizer_path = pathlib.Path(spec.tokenizer).expanduser().resolve() if spec.tokenizer else base
    if spec.tokenizer and tokenizer_sha256(tokenizer_path) != spec.tokenizer_sha256:
        raise ValueError("tokenizer content hash mismatch")
    frozen_config = spec.model_dump(mode="json", exclude={"resume"})
    source_pins = training_source_pins()
    identity = digest({"dataset_id": manifest["dataset_id"], "config": frozen_config, "source_pins": source_pins})
    declaration = output / "declaration.json"
    declared = {"identity": identity, "dataset_id": manifest["dataset_id"],
                "config": frozen_config, "source_pins": source_pins}
    if declaration.exists():
        if json.loads(declaration.read_text()) != declared:
            raise ValueError("training dataset/base/settings/sources changed; declare a new job")
    else:
        if list(output.glob("checkpoint-*")):
            raise ValueError("checkpoint without declaration")
        write_json(declaration, declared)
    checkpoints = sorted(output.glob("checkpoint-*"))
    previous = json.loads((output / "result.json").read_text()) if (output / "result.json").exists() else None
    if checkpoints:
        state = _read_checkpoint(checkpoints[-1], identity)
        if previous and previous.get("status") == "completed":
            if previous.get("identity") != identity or previous.get("optimizer_steps") != state["step"]:
                raise ValueError("completed receipt does not match the committed checkpoint")
            return previous
        if not spec.resume:
            raise ValueError("interrupted job has checkpoints; resume must be explicit")
    else:
        if previous and not spec.resume:
            raise ValueError("interrupted job requires explicit resume")
        state = {"identity": identity, "step": 0, "cursor": 0, "train_tokens": 0, "elapsed_seconds": 0.,
                 "losses": []}
    state["eval_tokens_reserved"] = max(state.get("eval_tokens_reserved", 0),
                                        (previous or {}).get("eval_tokens_reserved", 0))
    state["eval_tokens_completed"] = max(state.get("eval_tokens_completed", 0),
                                         (previous or {}).get("eval_tokens_completed", 0))
    state["train_tokens_reserved"] = max(state.get("train_tokens_reserved", 0),
                                         (previous or {}).get("train_tokens_reserved", 0))
    if previous is not None and "initial_development_loss" in previous and "initial_development_loss" not in state:
        state["initial_development_loss"] = previous["initial_development_loss"]
    elapsed_before = max(state["elapsed_seconds"], float((previous or {}).get("elapsed_seconds", 0)),
                         elapsed_floor_seconds)
    if (cancel_path is not None and cancel_path.exists()) or elapsed_before >= spec.max_seconds:
        result: dict[str, Any] = {
                  "status": "cancelled" if cancel_path is not None and cancel_path.exists() else "budget_exhausted",
                  "identity": identity, "dataset_id": manifest["dataset_id"], "config": frozen_config,
                  "source_pins": source_pins,
                  "optimizer_steps": state["step"], "train_tokens": state["train_tokens"],
                  "train_tokens_reserved": state["train_tokens_reserved"],
                  "eval_tokens_reserved": state["eval_tokens_reserved"],
                  "eval_tokens_completed": state["eval_tokens_completed"],
                  "elapsed_seconds": elapsed_before + time.monotonic() - started,
                  "behavioral_improvement_demonstrated": False, "test_split_evaluated": False}
        if checkpoints:
            result["checkpoint"] = str(checkpoints[-1])
        write_json(output / "result.json", result)
        return result
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    runtime = {name: version(name) for name in ("torch", "transformers", "peft")}
    if "runtime" in state and state["runtime"] != runtime:
        raise ValueError("resume requires identical training library versions")
    if spec.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                                          trust_remote_code=False)
    tokenized: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    dropped = []
    for example in examples:
        if example["split"] not in tokenized:
            continue  # Test examples are not tokenized, scored or used for model selection.
        encoded = _encode(tokenizer, example, manifest["config"])
        if encoded is None:
            dropped.append(example["id"])
        else:
            tokenized[example["split"]].append(encoded)
    if not all(tokenized.values()):
        raise ValueError("training requires nonempty independent train and development splits")
    development_tokens = sum(row["action_tokens"] for row in tokenized["development"])
    if development_tokens > spec.max_eval_tokens:
        raise ValueError("development tokens exceed the declared evaluation budget")
    order = []
    sampler = random.Random(spec.seed)
    for _ in range(spec.epochs):
        indices = list(range(len(tokenized["train"])))
        sampler.shuffle(indices)
        order.extend(indices)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base, dtype=getattr(torch, spec.dtype), local_files_only=True, trust_remote_code=False)
    if checkpoints:
        model = peft.PeftModel.from_pretrained(model, checkpoints[-1] / "adapter", is_trainable=True)
    else:
        model = peft.get_peft_model(model, peft.LoraConfig(task_type="CAUSAL_LM", r=spec.lora_rank,
                                    lora_alpha=spec.lora_alpha, lora_dropout=0.0, bias="none",
                                    target_modules=spec.target_modules))
    model.to(spec.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("no trainable adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    total_steps = min(spec.max_steps, math.ceil(len(order) / spec.batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 if spec.scheduler == "constant" else max(0., 1. - step / total_steps))
    if checkpoints:
        restored = torch.load(checkpoints[-1] / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        random.setstate(restored["python_rng"])
        torch.set_rng_state(restored["torch_rng"])
        if spec.device == "cuda":
            torch.cuda.set_rng_state_all(restored["cuda_rng"])
    state.update(runtime=runtime)
    result = {"status": "running", "identity": identity, "dataset_id": manifest["dataset_id"],
                              "source_pins": source_pins,
                              "config": frozen_config, "runtime": runtime, "dropped_context_examples": dropped,
                              "behavioral_improvement_demonstrated": False, "test_split_evaluated": False,
                              "token_budget_unit": "assistant_target_tokens",
                              "token_budget_enforcement": "nonrefundable_reservation_before_forward",
                              "train_tokens": state["train_tokens"], "optimizer_steps": state["step"],
                              "train_tokens_reserved": state["train_tokens_reserved"],
                              "evaluation_token_budget_scope": "all_development_passes_including_resume",
                              "eval_tokens_reserved": state["eval_tokens_reserved"],
                              "eval_tokens_completed": state["eval_tokens_completed"],
                              "walltime_enforcement": "cooperative_at_batch_boundaries",
                              "elapsed_floor_seconds": elapsed_floor_seconds}
    write_json(output / "result.json", result)
    try:
        if (cancel_path is not None and cancel_path.exists()) or (
            elapsed_before + time.monotonic() - started >= spec.max_seconds
        ):
            result.update(status=("cancelled" if cancel_path is not None and cancel_path.exists()
                                  else "budget_exhausted"),
                          optimizer_steps=state["step"], train_tokens=state["train_tokens"],
                          elapsed_seconds=elapsed_before + time.monotonic() - started)
            write_json(output / "result.json", result)
            return result
        if "initial_development_loss" not in state:
            if state["eval_tokens_reserved"] + development_tokens > spec.max_eval_tokens:
                result.update(status="budget_exhausted", stop_reason="evaluation_budget",
                              optimizer_steps=state["step"], train_tokens=state["train_tokens"],
                              elapsed_seconds=elapsed_before + time.monotonic() - started)
                write_json(output / "result.json", result)
                return result
            state["eval_tokens_reserved"] += development_tokens
            result["eval_tokens_reserved"] = state["eval_tokens_reserved"]
            write_json(output / "result.json", result)
            state["initial_development_loss"] = _evaluate(model, tokenized["development"], torch, spec.device)
            state["eval_tokens_completed"] += development_tokens
            result.update(initial_development_loss=state["initial_development_loss"],
                          eval_tokens_completed=state["eval_tokens_completed"])
            write_json(output / "result.json", result)
        model.train()
        stop_reason = "epochs"
        while state["cursor"] < len(order) and state["step"] < spec.max_steps:
            if cancel_path is not None and cancel_path.exists():
                stop_reason = "cancelled"
                break
            elapsed = elapsed_before + time.monotonic() - started
            if elapsed >= spec.max_seconds:
                stop_reason = "time_budget"
                break
            batch = [tokenized["train"][index] for index in order[state["cursor"]:state["cursor"] + spec.batch_size]]
            tokens = sum(row["action_tokens"] for row in batch)
            if state["train_tokens_reserved"] + tokens > spec.max_train_tokens:
                stop_reason = "token_budget"
                break
            state["train_tokens_reserved"] += tokens
            result["train_tokens_reserved"] = state["train_tokens_reserved"]
            write_json(output / "result.json", result)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch, torch, spec.device)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite SFT loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, spec.max_grad_norm, error_if_nonfinite=True)
            before = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()
            scheduler.step()
            delta = sum(float((parameter.detach() - old).float().square().sum().item())
                        for parameter, old in zip(parameters, before, strict=True)) ** .5
            if not math.isfinite(delta):
                raise ValueError("nonfinite adapter update")
            state["step"] += 1
            state["cursor"] += len(batch)
            state["train_tokens"] += tokens
            state["losses"].append({"step": state["step"], "loss": float(loss.item()),
                                    "gradient_norm": float(norm.item()), "weight_delta_l2": delta})
            state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            checkpoint = _checkpoint(output, model, optimizer, scheduler, state, torch)
            result.update(optimizer_steps=state["step"], checkpoint=str(checkpoint),
                          elapsed_seconds=state["elapsed_seconds"], train_tokens=state["train_tokens"])
            write_json(output / "result.json", result)
        if state["step"] >= spec.max_steps:
            stop_reason = "max_steps"
        if cancel_path is not None and cancel_path.exists():
            stop_reason = "cancelled"
        elapsed = elapsed_before + time.monotonic() - started
        final_dev = None
        if elapsed < spec.max_seconds and stop_reason != "cancelled" and (
            state["eval_tokens_reserved"] + development_tokens <= spec.max_eval_tokens
        ):
            state["eval_tokens_reserved"] += development_tokens
            result["eval_tokens_reserved"] = state["eval_tokens_reserved"]
            write_json(output / "result.json", result)
            final_dev = _evaluate(model, tokenized["development"], torch, spec.device)
            state["eval_tokens_completed"] += development_tokens
        result.update(status=("completed" if stop_reason in {"epochs", "max_steps"} else
                              "cancelled" if stop_reason == "cancelled" else "budget_exhausted"),
                      stop_reason=stop_reason, optimizer_steps=state["step"], train_tokens=state["train_tokens"],
                      initial_development_loss=state["initial_development_loss"], development_loss=final_dev,
                      development_evaluation_status="completed" if final_dev is not None else "not_evaluated",
                      eval_tokens_reserved=state["eval_tokens_reserved"],
                      eval_tokens_completed=state["eval_tokens_completed"],
                      losses=state["losses"], elapsed_seconds=elapsed_before + time.monotonic() - started,
                      trainable_parameters=sum(parameter.numel() for parameter in parameters))
        if state["step"]:
            final_checkpoint = output / f"checkpoint-{state['step']:08d}"
            result.update(checkpoint=str(final_checkpoint), adapter_path=str(final_checkpoint / "adapter"),
                          adapter_sha256=checkpoint_sha256(final_checkpoint / "adapter", adapter=True))
        write_json(output / "result.json", result)
        return result
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=f"{type(exc).__name__}: {exc}", elapsed_seconds=elapsed_before + time.monotonic() - started)
        write_json(output / "result.json", result)
        raise
