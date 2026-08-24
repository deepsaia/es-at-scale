"""Tests for reporting how precise an eval sample size actually is.

A 200-iteration run reported 7% accuracy from 100 eval prompts while the true
held-out rate was 11%. Small eval sets read as hard plateaus, so the run should
state its own resolution rather than let the reader assume the number is exact.
"""

import pytest

from mps_es.train_countdown import eval_standard_error


def test_standard_error_shrinks_with_more_samples():
    assert eval_standard_error(500) < eval_standard_error(100)


def test_standard_error_follows_the_binomial_formula():
    # sqrt(p(1-p)/n) at p=0.1, n=100 -> 0.03
    assert eval_standard_error(100, accuracy=0.1) == pytest.approx(0.03, abs=1e-9)


def test_standard_error_quarters_when_samples_multiply_by_sixteen():
    """Error falls as 1/sqrt(n), so 16x the samples halves it twice."""
    assert eval_standard_error(1600) == pytest.approx(
        eval_standard_error(100) / 4, rel=1e-9
    )


def test_standard_error_of_zero_samples_is_zero_rather_than_a_division_error():
    assert eval_standard_error(0) == 0.0


def test_standard_error_at_a_different_assumed_accuracy():
    assert eval_standard_error(100, accuracy=0.5) == pytest.approx(0.05, abs=1e-9)
