"""Model loading and batched greedy generation.

Separated from the ES mechanics so that `es_loop` stays free of HuggingFace and
device concerns, and so this half can be exercised against a tiny local model.
"""

from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "ModelRunner",
    "iter_minibatches",
    "load_runner",
    "resolve_device",
    "resolve_dtype",
    "weighted_mean",
]

_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_dtype(precision: str) -> torch.dtype:
    try:
        return _DTYPES[precision]
    except KeyError:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {sorted(_DTYPES)}"
        ) from None


def resolve_device(preferred: Optional[str] = None) -> str:
    """Pick a device, preferring Apple Silicon then CUDA then CPU."""
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def iter_minibatches(items: Sequence, size: int) -> Iterator[List]:
    """Split a batch into sequential chunks.

    Purely a memory lever, exactly as upstream's `--mini-batch-size`: rewards are
    recombined with `weighted_mean`, so chunking does not change the iteration's
    result.
    """
    if size <= 0:
        raise ValueError(f"mini-batch size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Recombine per-chunk means into the mean over the undivided batch."""
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


class ModelRunner:
    """Owns the model and turns prompts into completions.

    `trainable_params` hands out the model's own tensors, not copies, because the
    ES loop perturbs them in place.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        max_new_tokens: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def trainable_params(self) -> List[torch.Tensor]:
        return [p.data for p in self.model.parameters()]

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], mini_batch_size: int) -> List[str]:
        """Greedily complete each prompt; returns completions in prompt order."""
        if not prompts:
            return []

        completions: List[str] = []
        for chunk in iter_minibatches(prompts, mini_batch_size):
            encoded = self.tokenizer(
                chunk, return_tensors="pt", padding=True
            ).to(self.device)
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # strip the prompt: everything up to the (left-padded) input width
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            completions.extend(
                self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
        return completions


def load_runner(
    model_name: str,
    device: Optional[str],
    precision: str,
    max_new_tokens: int,
) -> ModelRunner:
    """Load a HuggingFace causal LM ready for ES perturbation."""
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # transformers renamed `torch_dtype` to `dtype` in 5.x. Accept both so this
    # module also works in an environment pinned to the upstream 4.57.6.
    dtype = resolve_dtype(precision)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)  # ES never differentiates; saves autograd bookkeeping

    return ModelRunner(
        model=model, tokenizer=tokenizer, device=device, max_new_tokens=max_new_tokens
    )
