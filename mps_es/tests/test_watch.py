"""Tests for the read-only metrics watcher."""

import json

import pytest

from mps_es.watch import load_metrics, sparkline, summarise


def write_metrics(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_sparkline_renders_one_character_per_value():
    assert len(sparkline([1, 2, 3, 4, 5])) == 5


def test_sparkline_puts_the_lowest_value_at_the_bottom_and_highest_at_the_top():
    line = sparkline([0, 1, 2])

    assert line[0] == "▁"
    assert line[-1] == "█"


def test_sparkline_of_a_flat_series_is_flat():
    line = sparkline([3, 3, 3])

    assert len(set(line)) == 1


def test_sparkline_of_an_empty_series_is_empty():
    assert sparkline([]) == ""


def test_load_metrics_reads_every_row(tmp_path):
    path = write_metrics(tmp_path / "m.jsonl", [
        {"iteration": 0, "mean_reward": 0.1},
        {"iteration": 1, "mean_reward": 0.2},
    ])

    assert len(load_metrics(path)) == 2


def test_load_metrics_skips_a_partially_written_trailing_line(tmp_path):
    """The watcher may read while the trainer is mid-write; a torn final line
    must not crash it."""
    path = tmp_path / "m.jsonl"
    path.write_text('{"iteration": 0, "mean_reward": 0.1}\n{"iteration": 1, "mea')

    rows = load_metrics(path)

    assert len(rows) == 1
    assert rows[0]["iteration"] == 0


def test_load_metrics_on_a_missing_file_returns_empty(tmp_path):
    assert load_metrics(tmp_path / "absent.jsonl") == []


def test_summarise_reports_the_latest_iteration_and_reward():
    rows = [
        {"iteration": 0, "mean_reward": 0.03, "max_reward": 0.05},
        {"iteration": 1, "mean_reward": 0.04, "max_reward": 0.11},
    ]

    text = summarise(rows)

    assert "iter 1" in text
    assert "0.0400" in text


def test_summarise_surfaces_the_latest_eval_accuracy():
    rows = [
        {"iteration": 0, "mean_reward": 0.03, "max_reward": 0.05},
        {"iteration": 1, "mean_reward": 0.04, "max_reward": 0.05,
         "eval_accuracy": 0.02, "eval_mean_reward": 0.09},
    ]

    assert "0.0200" in summarise(rows)


def test_summarise_flags_that_answer_credit_has_appeared():
    """Any member above the 0.1 format ceiling proves a correct answer, which is
    the earliest visible sign the model is solving rather than formatting."""
    rows = [{"iteration": 0, "mean_reward": 0.04, "max_reward": 0.1141}]

    assert "answer credit" in summarise(rows).lower()


def test_summarise_does_not_claim_answer_credit_below_the_ceiling():
    rows = [{"iteration": 0, "mean_reward": 0.03, "max_reward": 0.05}]

    assert "answer credit" not in summarise(rows).lower()


def test_summarise_of_no_rows_says_so_rather_than_crashing():
    assert summarise([]) != ""
