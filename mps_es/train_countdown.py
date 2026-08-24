#!/usr/bin/env python3
"""ES fine-tuning on the Countdown task, on Apple Silicon (or CPU, or CUDA).

Single-process counterpart to `es_at_scale/train.py`, which requires vLLM, Ray
and CUDA. This entry point wires the same countdown grader and the same on-disk
datasets to the device-agnostic ES loop in this package.

Run from the repository root so that `es_at_scale` resolves without installing
the package (installing it would pull vLLM, which has no macOS wheel):

    python -m mps_es.train_countdown --n-iterations 50
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk  # noqa: E402

from es_at_scale.reward_function.countdown_grader import countdown_reward_fn  # noqa: E402
from mps_es.checkpoint import (  # noqa: E402
    load_checkpoint,
    make_rng,
    next_seeds,
    save_checkpoint,
)
from mps_es.runner import load_runner, resolve_device, weighted_mean  # noqa: E402
from mps_es.trainer import ESHyperparams, run_iteration  # noqa: E402

__all__ = ["BatchScore", "score_batch", "to_prompts_and_targets", "main"]


@dataclass
class BatchScore:
    mean_reward: float
    accuracy: float
    mean_format_reward: float = 0.0
    mean_answer_reward: float = 0.0


def score_batch(responses: Sequence[str], targets: Sequence[Dict[str, Any]]) -> BatchScore:
    """Score one batch of completions.

    `mean_reward` (0.1 * format + answer) is what ES optimises; `accuracy` is the
    exact-answer rate, i.e. the pass@1 number comparable to the paper.

    The two components are reported separately because the total alone cannot
    distinguish a model that has learned the `<think>`/`<answer>` envelope from
    one that is actually solving the puzzle -- format credit saturates at 0.1,
    so early progress is all envelope.
    """
    if len(responses) != len(targets):
        raise ValueError(
            f"got {len(responses)} responses for {len(targets)} targets"
        )
    if not responses:
        return BatchScore(mean_reward=0.0, accuracy=0.0)

    rewards, formats, answers, correct = [], [], [], 0
    for response, target in zip(responses, targets):
        detail, reward = countdown_reward_fn(response, target)
        rewards.append(reward)
        formats.append(detail["format_reward"])
        answers.append(detail["answer_reward"])
        correct += int(detail["answer_reward"] == 1.0)

    n = len(responses)
    return BatchScore(
        mean_reward=sum(rewards) / n,
        accuracy=correct / n,
        mean_format_reward=sum(formats) / n,
        mean_answer_reward=sum(answers) / n,
    )


def to_prompts_and_targets(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Dict]]:
    """Split dataset rows into prompts and grader targets (upstream's collate_fn)."""
    prompts = [row["context"] for row in rows]
    targets = [{"numbers": row["numbers"], "target": row["target"]} for row in rows]
    return prompts, targets


def evaluate_split(runner, prompts, targets, mini_batch_size) -> BatchScore:
    """Score a whole split in chunks, recombining exactly as one pass would."""
    means, accuracies, formats, answers, sizes = [], [], [], [], []
    for start in range(0, len(prompts), mini_batch_size):
        chunk_prompts = prompts[start : start + mini_batch_size]
        chunk_targets = targets[start : start + mini_batch_size]
        responses = runner.generate(chunk_prompts, mini_batch_size=mini_batch_size)
        score = score_batch(responses, chunk_targets)
        means.append(score.mean_reward)
        accuracies.append(score.accuracy)
        formats.append(score.mean_format_reward)
        answers.append(score.mean_answer_reward)
        sizes.append(len(chunk_prompts))
    return BatchScore(
        mean_reward=weighted_mean(means, sizes),
        accuracy=weighted_mean(accuracies, sizes),
        mean_format_reward=weighted_mean(formats, sizes),
        mean_answer_reward=weighted_mean(answers, sizes),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ES fine-tuning for Countdown on Apple Silicon / CPU / CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # names mirror es_at_scale/train.py so the upstream README transfers
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--sigma", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=-1, help="defaults to sigma/2")
    p.add_argument("--population-size", type=int, default=30)
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32,
                   help="prompts each population member is scored on")
    p.add_argument("--mini-batch-size", type=int, default=32,
                   help="memory lever only; does not change the result")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--eval-freq", type=int, default=10, help="0 disables evaluation")
    p.add_argument("--eval-samples", type=int, default=100,
                   help="prompts from the 2000-row eval split; -1 for all")
    p.add_argument("--train-dataset", default=str(REPO_ROOT / "datasets/train/countdown"))
    p.add_argument("--eval-dataset",
                   default=str(REPO_ROOT / "datasets/evaluation_suite/countdown"))
    p.add_argument("--output-directory", default=str(REPO_ROOT / "experiments"))
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    # additions over upstream
    p.add_argument("--device", default=None, help="mps / cuda / cpu (autodetected)")
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--resume", default=None, help="path to a checkpoint to continue")
    p.add_argument("--decorrelate-layers", action="store_true",
                   help="give same-shaped layers independent noise (upstream does not)")
    p.add_argument("--checkpoint-freq", type=int, default=10)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    alpha = args.sigma / 2 if args.alpha < 0 else args.alpha
    device = resolve_device(args.device)

    name = args.experiment_name or f"countdown-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_directory) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    checkpoint_path = out_dir / "checkpoint.pt"

    train_rows = list(load_from_disk(args.train_dataset)["train"])
    eval_split = load_from_disk(args.eval_dataset)
    eval_rows = list(eval_split[next(iter(eval_split.keys()))])
    if args.eval_samples >= 0 and args.eval_samples < len(eval_rows):
        print(f"[eval] using {args.eval_samples} of {len(eval_rows)} eval prompts")
        eval_rows = eval_rows[: args.eval_samples]

    train_prompts, train_targets = to_prompts_and_targets(train_rows)
    eval_prompts, eval_targets = to_prompts_and_targets(eval_rows)
    batch_size = min(args.batch_size, len(train_prompts))
    if batch_size < args.batch_size:
        print(f"[train] batch size capped at {batch_size} (train split size)")

    print(f"[setup] device={device} precision={args.precision} model={args.model_name}")
    print(f"[setup] pop={args.population_size} sigma={args.sigma} alpha={alpha} "
          f"batch={batch_size} max_tokens={args.max_tokens}")
    runner = load_runner(args.model_name, device, args.precision, args.max_tokens)
    params = runner.trainable_params()

    hp = ESHyperparams(
        sigma=args.sigma,
        alpha=alpha,
        population_size=args.population_size,
        decorrelate_layers=args.decorrelate_layers,
    )

    start_iteration, rng = 0, make_rng(args.seed)
    if args.resume:
        state = load_checkpoint(args.resume)
        runner.model.load_state_dict(state["model_state_dict"])
        runner.model.to(device)
        params = runner.trainable_params()
        start_iteration, rng = state["iteration"], state["rng"]
        print(f"[resume] continuing from iteration {start_iteration} ({args.resume})")

    config = {k: v for k, v in vars(args).items()}
    config["alpha"] = alpha

    def log(record: Dict[str, Any]) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # a fixed slice per iteration, as upstream does: every member sees the same prompts
    cursor = (start_iteration * batch_size) % max(1, len(train_prompts))

    for iteration in range(start_iteration, args.n_iterations):
        started = time.time()
        window = [
            (cursor + i) % len(train_prompts) for i in range(batch_size)
        ]
        prompts = [train_prompts[i] for i in window]
        targets = [train_targets[i] for i in window]
        cursor = (cursor + batch_size) % len(train_prompts)

        member_scores: List[BatchScore] = []

        def evaluate() -> float:
            responses = runner.generate(prompts, mini_batch_size=args.mini_batch_size)
            score = score_batch(responses, targets)
            member_scores.append(score)
            return score.mean_reward

        seeds = next_seeds(rng, hp.population_size)
        result = run_iteration(params, seeds, evaluate, hp)
        elapsed = time.time() - started

        n_members = max(1, len(member_scores))
        mean_format = sum(s.mean_format_reward for s in member_scores) / n_members
        mean_answer = sum(s.mean_answer_reward for s in member_scores) / n_members
        best_accuracy = max((s.accuracy for s in member_scores), default=0.0)

        record = {
            "iteration": iteration,
            "mean_reward": result.mean_reward,
            "max_reward": result.max_reward,
            "mean_format_reward": mean_format,
            "mean_answer_reward": mean_answer,
            "best_member_accuracy": best_accuracy,
            "seconds": round(elapsed, 1),
        }
        print(f"[iter {iteration:4d}] mean_reward={result.mean_reward:.4f} "
              f"max={result.max_reward:.4f} | fmt={mean_format:.3f} "
              f"ans={mean_answer:.4f} best_acc={best_accuracy:.3f} ({elapsed:.0f}s)")

        if args.eval_freq and (iteration + 1) % args.eval_freq == 0:
            score = evaluate_split(runner, eval_prompts, eval_targets, args.mini_batch_size)
            record["eval_accuracy"] = score.accuracy
            record["eval_mean_reward"] = score.mean_reward
            print(f"[eval {iteration:4d}] accuracy={score.accuracy:.4f} "
                  f"mean_reward={score.mean_reward:.4f}")

        log(record)

        if args.checkpoint_freq and (iteration + 1) % args.checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, iteration=iteration + 1,
                            model=runner.model, rng=rng, config=config)
            print(f"[ckpt] saved at iteration {iteration + 1} -> {checkpoint_path}")

    save_checkpoint(checkpoint_path, iteration=args.n_iterations,
                    model=runner.model, rng=rng, config=config)
    print(f"[done] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
