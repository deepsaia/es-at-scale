"""Tests for model loading and batched greedy generation.

Built against a real (tiny, randomly initialised) Qwen model and the real Qwen
tokenizer rather than mocks, so the transformers generate path is genuinely
exercised. Nothing here downloads weights.
"""

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from mps_es.runner import (
    ModelRunner,
    iter_minibatches,
    resolve_device,
    resolve_dtype,
    weighted_mean,
)

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def runner(tiny_model, tokenizer):
    return ModelRunner(
        model=tiny_model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )


def test_resolve_dtype_maps_precision_names():
    assert resolve_dtype("fp32") is torch.float32
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16


def test_resolve_dtype_rejects_an_unknown_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        resolve_dtype("int4")


def test_resolve_device_honours_an_explicit_choice():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_autodetects_an_available_accelerator():
    device = resolve_device(None)

    expected = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    assert device == expected


def test_iter_minibatches_covers_every_item_in_order():
    assert list(iter_minibatches(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_iter_minibatches_with_size_larger_than_the_batch_yields_one_chunk():
    assert list(iter_minibatches([1, 2], 99)) == [[1, 2]]


def test_iter_minibatches_rejects_a_non_positive_size():
    with pytest.raises(ValueError):
        list(iter_minibatches([1, 2], 0))


def test_weighted_mean_matches_the_mean_over_the_undivided_batch():
    values = [0.0, 1.0, 0.5, 0.25, 1.0, 0.75, 0.5]
    chunks = [values[:3], values[3:6], values[6:]]

    chunk_means = [sum(c) / len(c) for c in chunks]
    chunk_sizes = [len(c) for c in chunks]

    assert weighted_mean(chunk_means, chunk_sizes) == pytest.approx(
        sum(values) / len(values)
    )


def test_generate_returns_one_completion_per_prompt(runner):
    outputs = runner.generate(["hello", "the capital of France is"], mini_batch_size=8)

    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_generate_preserves_prompt_order_across_minibatches(runner):
    prompts = ["alpha", "beta", "gamma", "delta", "epsilon"]

    one_chunk = runner.generate(prompts, mini_batch_size=len(prompts))
    many_chunks = runner.generate(prompts, mini_batch_size=2)

    assert len(one_chunk) == len(many_chunks) == len(prompts)


def test_generate_excludes_the_prompt_from_the_returned_completion(runner):
    prompt = "a distinctive prompt string"

    (completion,) = runner.generate([prompt], mini_batch_size=1)

    assert prompt not in completion


def test_generate_is_deterministic_under_greedy_decoding(runner):
    prompts = ["alpha", "beta"]

    assert runner.generate(prompts, mini_batch_size=2) == runner.generate(
        prompts, mini_batch_size=2
    )


def test_generate_handles_an_empty_prompt_list(runner):
    assert runner.generate([], mini_batch_size=4) == []


def test_trainable_params_exposes_the_models_tensors_for_perturbation(runner, tiny_model):
    params = runner.trainable_params()

    assert len(params) == len(list(tiny_model.parameters()))
    assert all(isinstance(p, torch.Tensor) for p in params)


def test_perturbing_trainable_params_changes_the_live_model(runner, tiny_model):
    """The ES loop mutates these tensors in place, so they must be the model's
    own storage rather than copies."""
    before = next(iter(tiny_model.parameters())).clone()

    params = runner.trainable_params()
    with torch.no_grad():
        params[0].add_(torch.ones_like(params[0]))

    after = next(iter(tiny_model.parameters()))
    assert not torch.equal(before, after)

    with torch.no_grad():  # leave the module-scoped fixture as we found it
        params[0].sub_(torch.ones_like(params[0]))
