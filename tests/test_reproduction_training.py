"""Real tiny CPU LM/PEFT integration checks; not external-task learning evidence."""

from __future__ import annotations

import json

import pytest

from mesocosm.learning.checkpoints import checkpoint_sha256
from mesocosm.reproduction.dataset import prepare_dataset
from mesocosm.reproduction.training import SFTConfig, _encode, run_training, training_capabilities


@pytest.fixture
def local_training_assets(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, attention_dropout=0.0))
    base = tmp_path / "base"
    model.save_pretrained(base, safe_serialization=True)
    tokenizer_core = tokenizers.Tokenizer(tokenizers.models.WordLevel({
        "[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "user": 3, "assistant": 4,
        "question": 5, "answer": 6, "other": 7, "normal": 8}, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer_core,
        unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}"
                               "{% if message['role'] == 'assistant' %}{{ eos_token + ' ' }}{% endif %}{% endfor %}"
                               "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}")
    tokenizer.save_pretrained(base)
    conversations = [{"id": name, "session_id": name, "task_id": name, "label": "normal", "messages": [
        {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    } for name in ["train-1", "train-2", "development", "test"]]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(conversations))
    dataset = tmp_path / "dataset"
    prepare_dataset([source], dataset, {"split_assignments": {
        "train-1": "train", "train-2": "train", "development": "development", "test": "test"}})
    config = {"base_model": str(base), "base_sha256": checkpoint_sha256(base),
              "max_steps": 3, "epochs": 2, "lora_rank": 2, "lora_alpha": 4,
              "learning_rate": .01, "device": "cpu", "seed": 5, "max_seconds": 120}
    return dataset, config, torch, tokenizer


def test_unsupported_backend_and_precision_fail_explicitly():
    assert training_capabilities()["methods"]["qlora"] == "unsupported"
    for patch in ({"backend": "qlora"}, {"dtype": "float16"}, {"quantization": "int4"}):
        with pytest.raises(ValueError):
            SFTConfig.model_validate({"base_model": "local", "base_sha256": "a" * 64, **patch})


def test_real_sft_changes_adapter_and_preserves_assistant_only_boundary(tmp_path, local_training_assets):
    dataset, config, torch, tokenizer = local_training_assets
    encoded = _encode(tokenizer, {"id": "one", "observation": [{"role": "user", "content": "question"}],
                                  "action": {"role": "assistant", "content": "answer"}},
                      {"max_context_tokens": 100, "overflow": "reject", "reasoning_policy": "preserve"})
    assert encoded["labels"][:3] == [-100] * 3
    assert encoded["labels"][3:] == [6, 1]
    result = run_training(dataset, tmp_path / "training", config)
    assert result["status"] == "completed" and result["optimizer_steps"] == 3
    assert all(item["weight_delta_l2"] > 0 for item in result["losses"])
    assert result["development_loss"] < result["initial_development_loss"]
    assert not result["test_split_evaluated"] and not result["behavioral_improvement_demonstrated"]
    assert checkpoint_sha256(result["adapter_path"], adapter=True) == result["adapter_sha256"]
    assert run_training(dataset, tmp_path / "training", config) == result
    with pytest.raises(ValueError, match="changed"):
        run_training(dataset, tmp_path / "training", {**config, "learning_rate": .02, "resume": True})


def test_resume_restores_optimizer_scheduler_rng_and_never_duplicates_step(tmp_path, local_training_assets, monkeypatch):
    import mesocosm.reproduction.training as training

    dataset, config, torch, _ = local_training_assets
    uninterrupted = run_training(dataset, tmp_path / "whole", config)
    original = training._checkpoint

    def terminate_after_commit(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("simulated operator interruption after durable checkpoint")

    monkeypatch.setattr(training, "_checkpoint", terminate_after_commit)
    with pytest.raises(KeyboardInterrupt):
        run_training(dataset, tmp_path / "resumed", config)
    with pytest.raises(ValueError, match="resume"):
        run_training(dataset, tmp_path / "resumed", config)
    monkeypatch.setattr(training, "_checkpoint", original)
    resumed = run_training(dataset, tmp_path / "resumed", {**config, "resume": True})
    assert resumed["losses"] == uninterrupted["losses"]
    assert resumed["adapter_sha256"] == uninterrupted["adapter_sha256"]
    assert resumed["development_loss"] == uninterrupted["development_loss"]
    assert len(list((tmp_path / "resumed").glob("checkpoint-*"))) == 3


def test_checkpoint_tamper_rejected_even_for_completed_idempotent_job(tmp_path, local_training_assets):
    dataset, config, _, _ = local_training_assets
    result = run_training(dataset, tmp_path / "training", {**config, "max_steps": 1})
    path = tmp_path / "training/checkpoint-00000001/state.pt"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        run_training(dataset, tmp_path / "training", {**config, "max_steps": 1})
    assert result["optimizer_steps"] == 1


def test_budget_and_cancellation_do_not_claim_update(tmp_path, local_training_assets):
    dataset, config, _, _ = local_training_assets
    result = run_training(dataset, tmp_path / "limited", {**config, "max_train_tokens": 1})
    assert result["status"] == "budget_exhausted" and result["optimizer_steps"] == 0
    assert "adapter_path" not in result
    sentinel = tmp_path / "cancel.requested"
    sentinel.touch()
    cancelled = run_training(dataset, tmp_path / "cancelled", config, cancel_path=sentinel)
    assert cancelled["status"] == "cancelled" and cancelled["optimizer_steps"] == 0


def test_context_limit_and_reasoning_template_fail_closed(local_training_assets):
    _, _, _, tokenizer = local_training_assets
    example = {"id": "one", "observation": [{"role": "user", "content": "question"}],
               "action": {"role": "assistant", "content": "answer"}}
    with pytest.raises(ValueError, match="context limit"):
        _encode(tokenizer, example, {"max_context_tokens": 2, "overflow": "reject", "reasoning_policy": "preserve"})
    assert _encode(tokenizer, example, {"max_context_tokens": 2, "overflow": "drop",
                                       "reasoning_policy": "preserve"}) is None
    example["action"]["reasoning_content"] = "reason"
    with pytest.raises(ValueError, match="reasoning retention"):
        _encode(tokenizer, example, {"max_context_tokens": 100, "overflow": "reject", "reasoning_policy": "preserve"})


def test_tool_template_comment_cannot_falsely_claim_retention(local_training_assets):
    _, _, _, tokenizer = local_training_assets
    tokenizer.chat_template += "{# tools tool_calls #}"
    example = {"id": "one", "observation": [{"role": "user", "content": "question"}],
               "action": {"role": "assistant", "content": "answer", "tool_calls": [{"id": "call", "function": {
                   "name": "read_file", "arguments": '{"path":"original.txt"}'}}]},
               "tools": [{"type": "function", "function": {"name": "read_file", "description": "read a file"}}]}
    with pytest.raises(ValueError, match="tool schemas/actions/results"):
        _encode(tokenizer, example, {"max_context_tokens": 100, "overflow": "reject", "reasoning_policy": "preserve"})


def test_evaluation_budget_is_total_and_interruption_reservation_is_not_refunded(
    tmp_path, local_training_assets, monkeypatch,
):
    import mesocosm.reproduction.training as training

    dataset, config, _, _ = local_training_assets
    result = run_training(dataset, tmp_path / "limited_eval", {**config, "max_eval_tokens": 2})
    assert result["eval_tokens_completed"] == result["eval_tokens_reserved"] == 2
    assert result["development_loss"] is None
    assert result["development_evaluation_status"] == "not_evaluated"

    def interrupted_evaluation(*args):
        raise KeyboardInterrupt("interrupted evaluation")

    monkeypatch.setattr(training, "_evaluate", interrupted_evaluation)
    with pytest.raises(KeyboardInterrupt):
        run_training(dataset, tmp_path / "interrupted_eval", {**config, "max_eval_tokens": 2})
    resumed = run_training(dataset, tmp_path / "interrupted_eval", {**config, "max_eval_tokens": 2, "resume": True})
    assert resumed["status"] == "budget_exhausted"
    assert resumed["eval_tokens_reserved"] == 2 and resumed["eval_tokens_completed"] == 0
    assert resumed["optimizer_steps"] == 0


def test_supervisor_charged_time_floor_stops_before_import_or_model_load(tmp_path, local_training_assets, monkeypatch):
    import mesocosm.reproduction.training as training

    dataset, config, _, _ = local_training_assets
    monkeypatch.setattr(training.importlib, "import_module", lambda *args: pytest.fail("no compute after spent budget"))
    result = run_training(dataset, tmp_path / "spent", config, elapsed_floor_seconds=config["max_seconds"])
    assert result["status"] == "budget_exhausted" and result["optimizer_steps"] == 0


def test_interrupted_forward_does_not_refund_training_token_budget(tmp_path, local_training_assets, monkeypatch):
    import mesocosm.reproduction.training as training

    dataset, config, _, _ = local_training_assets
    original = training._batch_loss

    def interrupt_training_forward(model, *args):
        if model.training:
            raise KeyboardInterrupt("stopped during training forward")
        return original(model, *args)

    monkeypatch.setattr(training, "_batch_loss", interrupt_training_forward)
    with pytest.raises(KeyboardInterrupt):
        run_training(dataset, tmp_path / "training", {**config, "max_train_tokens": 2})
    monkeypatch.setattr(training, "_batch_loss", original)
    result = run_training(dataset, tmp_path / "training", {**config, "max_train_tokens": 2, "resume": True})
    assert result["status"] == "budget_exhausted" and result["optimizer_steps"] == 0
    assert result["train_tokens"] == 0 and result["train_tokens_reserved"] == 2


def test_cancellation_after_commit_retains_checkpoint_and_skips_final_evaluation(
    tmp_path, local_training_assets, monkeypatch,
):
    import mesocosm.reproduction.training as training

    dataset, config, _, _ = local_training_assets
    original = training._checkpoint
    cancel_path = tmp_path / "cancel.requested"

    def cancel_after_commit(*args):
        path = original(*args)
        cancel_path.touch()
        return path

    monkeypatch.setattr(training, "_checkpoint", cancel_after_commit)
    result = run_training(dataset, tmp_path / "training", {**config, "max_steps": 1}, cancel_path=cancel_path)
    assert result["status"] == "cancelled" and result["optimizer_steps"] == 1
    assert result["development_loss"] is None
    assert result["eval_tokens_completed"] == 2
    assert result["adapter_path"]


def test_source_changes_reject_completed_reuse_and_resume(tmp_path, local_training_assets, monkeypatch):
    import mesocosm.reproduction.training as training

    dataset, config, _, _ = local_training_assets
    result = run_training(dataset, tmp_path / "training", {**config, "max_steps": 1})
    pins = training.training_source_pins()
    assert result["source_pins"] == pins
    assert set(pins) == {"reproduction/training.py", "reproduction/dataset.py", "learning/checkpoints.py"}
    monkeypatch.setattr(training, "training_source_pins", lambda: {**pins, "reproduction/training.py": "0" * 64})
    for resume in (False, True):
        with pytest.raises(ValueError, match="changed"):
            run_training(dataset, tmp_path / "training", {**config, "max_steps": 1, "resume": resume})
