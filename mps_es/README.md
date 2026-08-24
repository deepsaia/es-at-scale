# mps_es: ES fine-tuning on Apple Silicon

A single-process port of the Evolution Strategies loop that runs on **Apple
Silicon (MPS)**, CPU, or a single CUDA GPU.

This is **not** a replacement for `es_at_scale/`. The supported trainer needs
vLLM, Ray, NCCL and multiple CUDA GPUs, and is what you should use on a
cluster.

vLLM has never published a macOS wheel: 0 of 95 releases on PyPI, latest 0.27.1.
So `pip install -e .` cannot succeed on Apple Silicon. vLLM does document an
experimental build-from-source path for macOS, but it is CPU only with no Metal
backend and supports FP32/FP16 only, which gives up the throughput that is the
reason to reach for vLLM at all. This module takes the other route: drop the
serving layer and run the ES loop directly on MPS.

Nothing outside this directory is modified. `es_at_scale/`, `archive/`,
`setup.py` and `datasets/` are untouched; the countdown grader and the on-disk
datasets are imported and read, never copied or edited.

## Install

Its own environment, because the repo root's `setup.py` cannot be installed on
macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r mps_es/requirements.txt
```

## Run

From the **repository root**, so that `es_at_scale` resolves without installing
the package:

```bash
python -m mps_es.train_countdown --n-iterations 50
```

Defaults are tuned for an M-series Mac: `Qwen/Qwen2.5-0.5B-Instruct`,
population 30, batch 32, 256 max tokens, σ=0.001, α=σ/2, about **4 minutes per
iteration**. Metrics stream to stdout and to
`experiments/<name>/metrics.jsonl`; checkpoints go alongside them.

Resume an interrupted run:

```bash
python -m mps_es.train_countdown --n-iterations 200 \
  --resume experiments/<name>/checkpoint.pt
```

Resume restores the weights, the iteration counter **and the seed-stream state**,
so a restarted run continues exactly as an uninterrupted one would.

## Watching a run

`mps_es.watch` is read-only, no model and no GPU, so it is safe to run alongside
training:

```bash
python -m mps_es.watch experiments/<name>            # once
python -m mps_es.watch experiments/<name> --follow   # live
```

```
iter 49  mean_reward=0.0836  max=0.2125
reward  ▁▁▁▁▁▁▃▃▂▂▃▂▃▃▃▃▄▄▄▃▄▄▄▄▄▅▅▅▅▄▅▆▅▅▅▅▆▅▇▆▆▆▆▇▆▇▆▇██  0.0298 -> 0.0836
eval    ▁▁▁▁█  accuracy=0.0200 (iter 49, 5 evals)
answer credit present: 17/50 iterations had a member above the 0.1 format ceiling (first at iter 16)
50 iterations, 191 min elapsed
```

That last line is the useful one early on. Reward is `0.1 * format + answer`, so
format credit **saturates at 0.1**, so any population member scoring above that
must have solved at least one puzzle. It is visible well before eval accuracy
clears zero.

Each iteration also logs `mean_format_reward`, `mean_answer_reward` and
`best_member_accuracy`, so you can tell envelope-learning from actual solving
without waiting for an eval.

## Using it for other models and tasks

Only `train_countdown.py` is task-specific, the same split upstream uses, where
`es_at_scale/train.py` is an example and the trainer is task-agnostic. Here
`es_loop.py`, `trainer.py`, `runner.py` and `checkpoint.py` know nothing about
countdown or about any particular model.

- **Another model:** `--model-name <any HF causal LM>`. Memory and throughput are
  the real limits, not the code. At bf16, budget ~2 GB per billion parameters,
  and expect generation to slow roughly in proportion to parameter count.
- **Another task:** copy `train_countdown.py` and replace three things: the
  grader (`score_batch`), the prompt/target extraction
  (`to_prompts_and_targets`), and the dataset paths. The repo already ships a
  math grader at `es_at_scale/reward_function/math_grader.py` and a MATH
  training split under `datasets/train/math_lvl3to5_8k`.

## Measured throughput

Qwen2.5-0.5B-Instruct, bf16, MPS, 256 new tokens, on an M5 Max:

| batch | tok/s | pop-30 iteration |
|------:|------:|-----------------:|
| 16 | 705 | 2.9 min |
| 32 | 1053 | 3.9 min |
| 64 | 1442 | 5.7 min |
| 200 | 1714 | 14.9 min |

Noise generation is not a bottleneck: MPS draws ~36 G samples/s, about 0.4 s per
iteration at 0.5B. On CPU it is ~200 times slower and would dominate.

MPS matters for generation too, though less dramatically. Same model, batch 16,
128 new tokens: MPS bf16 426 tok/s versus CPU fp32 137 tok/s, so about 3 times
faster.

For scale: the paper's reference Countdown run is 500 iterations × population 30
× 200 prompts × 512 tokens ≈ 3 million rollouts on 8 GPUs. At batch 200 and 512
tokens that is roughly **10 days** here. The defaults above are a scaled-down
configuration, not a reproduction.

## Flags

Names mirror `es_at_scale/train.py` where they overlap. Additions:

| Flag | Default | Notes |
|---|---|---|
| `--device` | autodetect | `mps` → `cuda` → `cpu` |
| `--precision` | `bf16` | `fp32` avoids the drift described below |
| `--resume` | none | path to a checkpoint |
| `--eval-samples` | `100` | of the 2000-row eval split; `-1` for all |
| `--decorrelate-layers` | off | see below |
| `--checkpoint-freq` | `10` | `0` disables |

## Two fidelity notes

**bf16 perturb/restore drift.** `theta + sigma*eps - sigma*eps` does not return
exactly to `theta`. In bf16 the residue is a random walk reaching ~20% of σ in
RMS after 480 perturb/restore cycles (fp32: ~1 ulp, six orders of magnitude
below σ). This is inherited from the upstream design, which also perturbs in the
model's own dtype, and is not introduced here. bf16 is the default for
comparability; `--precision fp32` removes it at 2× memory.

**Per-layer noise correlation.** Upstream re-seeds the generator with the same
value for every layer, so identically-shaped layers receive identical noise.
That behaviour is replicated by default. `--decorrelate-layers` mixes the layer
index into the seed instead.

## Tests

```bash
python -m pytest mps_es/tests/ -q
```

63 tests, no model downloads, a few seconds. The ES mechanics are verified
independently of any LLM, including that a perturb/restore round trip returns
to within dtype rounding, that the decomposed update matches a naive reference,
that resume reproduces an identical seed stream, and that the loop climbs a
reward with a known optimum.
