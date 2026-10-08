"""Tests for the countdown task wiring.

Scoring is the part with real logic: mean reward drives ES, while accuracy
(exact-answer rate) is the number comparable to the paper's Table 1.
"""

import json

import pytest

from mps_es.train_countdown import (
    evaluate_split,
    score_batch,
    score_responses,
    to_prompts_and_targets,
    write_eval_outputs,
)


CORRECT = "</think>\n<answer> (44 + 19) + 35 </answer>"
WRONG = "</think>\n<answer> 44 + 19 </answer>"
UNFORMATTED = "the answer is 98"


def target(numbers=(44, 19, 35), value=98):
    return {"numbers": list(numbers), "target": value}


def test_score_batch_reports_accuracy_as_the_exact_answer_rate():
    result = score_batch([CORRECT, WRONG], [target(), target()])

    assert result.accuracy == pytest.approx(0.5)


def test_score_batch_gives_a_correct_answer_full_credit():
    result = score_batch([CORRECT], [target()])

    assert result.accuracy == 1.0
    assert result.mean_reward == pytest.approx(1.1)  # 0.1 * format + 1.0 * answer


def test_score_batch_gives_a_wrong_but_well_formatted_answer_partial_credit():
    result = score_batch([WRONG], [target()])

    assert result.accuracy == 0.0
    assert 0.0 < result.mean_reward < 1.0


def test_score_batch_gives_an_unformatted_response_no_credit():
    result = score_batch([UNFORMATTED], [target()])

    assert result.accuracy == 0.0
    assert result.mean_reward == pytest.approx(0.0)


def test_score_batch_on_an_empty_batch_is_zero_rather_than_nan():
    result = score_batch([], [])

    assert result.mean_reward == 0.0
    assert result.accuracy == 0.0


def test_score_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score_batch([CORRECT], [target(), target()])


def test_to_prompts_and_targets_splits_dataset_rows():
    rows = [
        {"context": "prompt one", "numbers": [1, 2], "target": 3},
        {"context": "prompt two", "numbers": [4, 5], "target": 9},
    ]

    prompts, targets = to_prompts_and_targets(rows)

    assert prompts == ["prompt one", "prompt two"]
    assert targets == [{"numbers": [1, 2], "target": 3}, {"numbers": [4, 5], "target": 9}]


def test_score_responses_returns_one_graded_record_per_response():
    records = score_responses([CORRECT, WRONG], [target(), target()])

    assert [r["correct"] for r in records] == [True, False]
    assert records[0]["reward"] == pytest.approx(1.1)
    assert records[0]["response"] == CORRECT
    assert records[0]["target"] == target()


def test_write_eval_outputs_saves_records_as_json(tmp_path):
    path = tmp_path / "eval-output" / "eval_baseline.json"
    records = score_responses([CORRECT], [target()])

    write_eval_outputs(path, records)

    saved = json.loads(path.read_text())
    assert saved[0]["response"] == CORRECT
    assert saved[0]["correct"] is True


class _ScriptedRunner:
    """Returns fixed responses so eval can be tested without a model."""

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompts, mini_batch_size):
        out, self._responses = self._responses[: len(prompts)], self._responses[len(prompts):]
        return out


def test_evaluate_split_grades_every_prompt_and_writes_each_response(tmp_path):
    runner = _ScriptedRunner([CORRECT, WRONG, UNFORMATTED])
    prompts = ["p1", "p2", "p3"]
    targets = [target(), target(), target()]
    out = tmp_path / "eval_iteration1.json"

    score = evaluate_split(runner, prompts, targets, mini_batch_size=2, output_path=out)

    assert score.accuracy == pytest.approx(1 / 3)
    saved = json.loads(out.read_text())
    assert [r["prompt"] for r in saved] == prompts
    assert [r["correct"] for r in saved] == [True, False, False]


def test_evaluate_split_result_does_not_depend_on_mini_batch_size():
    responses = [CORRECT, WRONG, UNFORMATTED, CORRECT, WRONG]
    prompts = [f"p{i}" for i in range(5)]
    targets = [target() for _ in range(5)]

    whole = evaluate_split(_ScriptedRunner(responses), prompts, targets, mini_batch_size=5)
    chunked = evaluate_split(_ScriptedRunner(responses), prompts, targets, mini_batch_size=2)

    assert whole == chunked


def _grade_or_hang(response, target):
    """Stand-in grader: hangs on one specific response, grades the rest."""
    if response == "hang":
        import time
        time.sleep(30)
    from es_at_scale.reward_function.countdown_grader import countdown_reward_fn
    return countdown_reward_fn(response, target)


def test_timed_grader_passes_normal_grades_through():
    from mps_es.train_countdown import TimedGrader

    grader = TimedGrader(_grade_or_hang, timeout=20)
    try:
        detail, reward = grader(CORRECT, target())
    finally:
        grader.close()

    assert reward == pytest.approx(1.1)
    assert detail["answer_reward"] == 1.0


def test_timed_grader_scores_a_hung_call_as_zero_and_keeps_working():
    from mps_es.train_countdown import TimedGrader

    grader = TimedGrader(_grade_or_hang, timeout=1)
    try:
        detail, reward = grader("hang", target())
        _, after = grader(CORRECT, target())
    finally:
        grader.close()

    assert reward == 0.0
    assert detail["timed_out"] is True
    assert after == pytest.approx(1.1)


def test_score_batch_uses_the_given_grader():
    calls = []

    def grader(response, target):
        calls.append(response)
        return {"format_reward": 1.0, "answer_reward": 1.0}, 1.1

    result = score_batch([UNFORMATTED], [target()], grader)

    assert calls == [UNFORMATTED]
    assert result.accuracy == 1.0
