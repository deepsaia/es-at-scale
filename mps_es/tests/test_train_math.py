"""Tests for the MATH task wiring.

The reward is binary, so the interesting logic is the prompt template, the
per-split eval and the metrics layout shared with countdown.
"""

import json

import pytest

from mps_es.train_math import (
    ALL_EVAL_SPLITS,
    build_parser,
    eval_record,
    evaluate_splits,
    score_batch,
    score_responses,
    to_prompts_and_targets,
)


CORRECT = "The sum is $p + q = 118$, so the answer is \\boxed{118}."
WRONG = "Therefore \\boxed{117}."
UNBOXED = "I think it is 118."


def test_score_batch_gives_a_correct_boxed_answer_full_credit():
    result = score_batch([CORRECT], ["118"])

    assert result.accuracy == 1.0
    assert result.mean_reward == 1.0


def test_score_batch_gives_a_boxed_but_wrong_answer_nothing():
    """Unlike countdown there is no format credit: a boxed wrong answer scores
    zero, matching upstream's math grader."""
    result = score_batch([WRONG], ["118"])

    assert result.accuracy == 0.0
    assert result.mean_reward == 0.0
    assert result.mean_format_reward == 1.0  # reported, not rewarded


def test_score_batch_gives_an_unboxed_response_nothing_and_reports_it_as_unformatted():
    result = score_batch([UNBOXED], ["118"])

    assert result.mean_reward == 0.0
    assert result.mean_format_reward == 0.0


def test_score_responses_accepts_numeric_and_list_targets():
    """The eval suite stores answers as strings, floats and lists of strings."""
    records = score_responses(
        ["\\boxed{142}", "\\boxed{9.6}", "\\boxed{2}"],
        [142.0, ["9.6\n"], ["2"]],
    )

    assert [r["correct"] for r in records] == [True, True, True]


def test_score_responses_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score_responses([CORRECT], ["118", "119"])


def test_to_prompts_and_targets_wraps_the_problem_in_the_qwen_math_template():
    prompts, targets = to_prompts_and_targets([{"problem": "What is 1+1?", "answer": "2"}])

    assert "What is 1+1?" in prompts[0]
    assert "\\boxed{}" in prompts[0]
    assert prompts[0].endswith("<|im_start|>assistant\n")
    assert targets == ["2"]


class _ScriptedRunner:
    """Returns fixed responses so eval can be tested without a model."""

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompts, mini_batch_size):
        out, self._responses = self._responses[: len(prompts)], self._responses[len(prompts):]
        return out


def test_evaluate_splits_scores_each_split_separately_and_writes_each_response(tmp_path):
    runner = _ScriptedRunner([CORRECT, WRONG, CORRECT])
    splits = {
        "math500": (["p1", "p2"], ["118", "118"]),
        "amc": (["p3"], ["118"]),
    }

    scores = evaluate_splits(runner, splits, mini_batch_size=1, eval_dir=tmp_path, tag="baseline")

    assert scores["math500"].accuracy == pytest.approx(0.5)
    assert scores["amc"].accuracy == 1.0
    saved = json.loads((tmp_path / "eval_baseline_math500.json").read_text())
    assert [r["prompt"] for r in saved] == ["p1", "p2"]
    assert [r["correct"] for r in saved] == [True, False]
    assert (tmp_path / "eval_baseline_amc.json").exists()


def test_evaluate_splits_result_does_not_depend_on_mini_batch_size():
    responses = [CORRECT, WRONG, UNBOXED, CORRECT, WRONG]
    splits = {"math500": ([f"p{i}" for i in range(5)], ["118"] * 5)}

    whole = evaluate_splits(_ScriptedRunner(responses), splits, mini_batch_size=5)
    chunked = evaluate_splits(_ScriptedRunner(responses), splits, mini_batch_size=2)

    assert whole == chunked


def test_eval_record_reports_every_split_and_the_headline_under_the_shared_key():
    scores = {"math500": score_batch([CORRECT], ["118"]), "amc": score_batch([WRONG], ["118"])}

    record = eval_record(scores, headline="math500")

    assert record["eval_accuracy_math500"] == 1.0
    assert record["eval_accuracy_amc"] == 0.0
    assert record["eval_accuracy"] == 1.0  # what the watcher follows


def test_parser_defaults_to_every_split_of_the_upstream_eval_suite():
    args = build_parser().parse_args([])

    assert args.eval_splits == ALL_EVAL_SPLITS
    assert args.eval_splits.split(",")[0] == "math500"
