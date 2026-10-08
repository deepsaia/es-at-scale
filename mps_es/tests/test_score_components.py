"""Tests for splitting reward into its format and answer components.

`mean_reward` alone cannot tell you whether the model is solving the task or
merely formatting its output. These components make that visible every
iteration rather than only at eval time.
"""

import pytest

from mps_es.train_countdown import score_batch

CORRECT = "</think>\n<answer> (44 + 19) + 35 </answer>"
WRONG = "</think>\n<answer> 44 + 19 </answer>"
UNFORMATTED = "the answer is 98"
TARGET = {"numbers": [44, 19, 35], "target": 98}


def test_a_correct_answer_scores_full_answer_credit():
    result = score_batch([CORRECT], [TARGET])

    assert result.mean_answer_reward == pytest.approx(1.0)


def test_a_wrong_answer_scores_no_answer_credit_but_keeps_format_credit():
    result = score_batch([WRONG], [TARGET])

    assert result.mean_answer_reward == pytest.approx(0.0)
    assert result.mean_format_reward > 0.0


def test_an_unformatted_response_scores_neither():
    result = score_batch([UNFORMATTED], [TARGET])

    assert result.mean_answer_reward == pytest.approx(0.0)
    assert result.mean_format_reward == pytest.approx(0.0)


def test_components_recombine_into_the_total_reward():
    """total = 0.1 * format + answer, the weighting countdown_reward_fn uses."""
    responses = [CORRECT, WRONG, UNFORMATTED]
    targets = [TARGET] * 3

    result = score_batch(responses, targets)

    assert result.mean_reward == pytest.approx(
        0.1 * result.mean_format_reward + result.mean_answer_reward
    )


def test_components_are_zero_on_an_empty_batch():
    result = score_batch([], [])

    assert result.mean_format_reward == 0.0
    assert result.mean_answer_reward == 0.0
