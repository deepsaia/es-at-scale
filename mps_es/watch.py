#!/usr/bin/env python3
"""Read-only viewer for a run's metrics.jsonl.

Touches no model and no GPU, so it is safe to run alongside training:

    python -m mps_es.watch experiments/<name>            # once
    python -m mps_es.watch experiments/<name> --follow   # live
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

__all__ = ["load_metrics", "sparkline", "summarise", "main"]

BLOCKS = "▁▂▃▄▅▆▇█"

# format_reward maxes at 1.0 and is weighted 0.1, so any member mean above this
# mathematically requires answer credit -- the earliest sign of real solving.
FORMAT_CEILING = 0.1


def sparkline(values: Sequence[float]) -> str:
    """Render a series as one block character per point."""
    values = list(values)
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        return BLOCKS[0] * len(values)
    span = high - low
    return "".join(
        BLOCKS[min(len(BLOCKS) - 1, int((v - low) / span * len(BLOCKS)))] for v in values
    )


def load_metrics(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a metrics.jsonl, tolerating a torn final line.

    The trainer may be mid-write when the watcher reads, so an incomplete last
    line is skipped rather than raising.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial trailing write
    return rows


def _resolve(path: Union[str, Path]) -> Path:
    """Accept either an experiment directory or the metrics file itself."""
    path = Path(path)
    return path / "metrics.jsonl" if path.is_dir() else path


def summarise(rows: Sequence[Dict[str, Any]]) -> str:
    """A compact status block: trend, latest values, and what they imply."""
    if not rows:
        return "no metrics yet"

    latest = rows[-1]
    rewards = [r["mean_reward"] for r in rows]
    lines = [
        f"iter {latest['iteration']}  mean_reward={latest['mean_reward']:.4f}"
        f"  max={latest.get('max_reward', float('nan')):.4f}",
        f"reward  {sparkline(rewards)}  {rewards[0]:.4f} -> {rewards[-1]:.4f}",
    ]

    if "mean_format_reward" in latest:
        lines.append(
            f"format={latest['mean_format_reward']:.3f}/1.0   "
            f"answer={latest['mean_answer_reward']:.4f}   "
            f"best_member_acc={latest.get('best_member_accuracy', 0.0):.3f}"
        )

    evals = [r for r in rows if "eval_accuracy" in r]
    if evals:
        accuracies = [r["eval_accuracy"] for r in evals]
        lines.append(
            f"eval    {sparkline(accuracies)}  accuracy={accuracies[-1]:.4f}"
            f" (iter {evals[-1]['iteration']}, {len(evals)} evals)"
        )

    over = [r for r in rows if r.get("max_reward", 0.0) > FORMAT_CEILING]
    if over:
        lines.append(
            f"answer credit present: {len(over)}/{len(rows)} iterations had a member"
            f" above the {FORMAT_CEILING} format ceiling (first at iter"
            f" {over[0]['iteration']})"
        )
    else:
        lines.append(
            f"no member has yet exceeded the {FORMAT_CEILING} format ceiling"
            " -- still learning the envelope, not the task"
        )

    total_seconds = sum(r.get("seconds", 0.0) for r in rows)
    lines.append(f"{len(rows)} iterations, {total_seconds / 60:.0f} min elapsed")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="experiment directory or metrics.jsonl")
    parser.add_argument("--follow", action="store_true", help="refresh until interrupted")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args(argv)

    target = _resolve(args.path)
    while True:
        print(f"\n=== {target} ===")
        print(summarise(load_metrics(target)))
        if not args.follow:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
