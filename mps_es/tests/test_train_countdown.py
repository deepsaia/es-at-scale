"""Tests for the countdown task wiring.

Scoring is the part with real logic: mean reward drives ES, while accuracy
(exact-answer rate) is the number comparable to the paper's Table 1.
"""

import pytest

from mps_es.train_countdown import score_batch, to_prompts_and_targets


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
