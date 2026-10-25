"""Unit tests for the from-scratch retrieval metrics. No infrastructure needed."""

import math

import pytest

from evaluation.eval_harness import (
    bootstrap_delta_ci,
    bootstrap_mean_ci,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_all_relevant_not_just_one():
    # The distinction the harness exists to make: 1 of 2 gold chunks found = 0.5,
    # even though a hit-rate metric would call this a perfect result.
    retrieved = ["a", "x", "y"]
    relevant = frozenset({"a", "b"})
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0


def test_recall_at_k_respects_the_cutoff():
    retrieved = ["x", "y", "a"]
    relevant = frozenset({"a"})
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_hit_rate_is_binary():
    relevant = frozenset({"a", "b"})
    assert hit_rate_at_k(["a", "b"], relevant, 2) == 1.0
    assert hit_rate_at_k(["a", "z"], relevant, 2) == 1.0
    assert hit_rate_at_k(["z", "w"], relevant, 2) == 0.0


def test_reciprocal_rank_is_one_indexed():
    relevant = frozenset({"a"})
    assert reciprocal_rank(["a", "x"], relevant) == 1.0
    assert reciprocal_rank(["x", "a"], relevant) == 0.5
    assert reciprocal_rank(["x", "y", "a"], relevant) == 1 / 3
    assert reciprocal_rank(["x", "y"], relevant) == 0.0


def test_empty_relevant_set_scores_zero_not_crash():
    assert recall_at_k(["a"], frozenset(), 1) == 0.0
    assert hit_rate_at_k(["a"], frozenset(), 1) == 0.0


# --------------------------------------------------------------------------- #
# nDCG@k
# --------------------------------------------------------------------------- #

def test_ndcg_rewards_rank_within_the_cutoff_where_recall_cannot():
    # The reason nDCG is here. Both orderings find the one gold chunk inside k=3,
    # so recall@3 and hit@3 are identical for them and cannot tell them apart.
    first = ["g", "x", "y"]
    third = ["x", "y", "g"]
    gold = frozenset({"g"})

    assert recall_at_k(first, gold, 3) == recall_at_k(third, gold, 3)
    assert ndcg_at_k(first, gold, 3) > ndcg_at_k(third, gold, 3)


def test_ndcg_matches_hand_computed_value():
    # One hit at rank 2, one gold chunk: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(["x", "g", "y"], frozenset({"g"}), 3) == pytest.approx(1 / math.log2(3))

    # Two gold chunks at ranks 1 and 3:
    #   DCG  = 1/log2(2) + 1/log2(4) = 1.5
    #   IDCG = 1/log2(2) + 1/log2(3)          (ideal puts both at the top)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "c"], frozenset({"a", "c"}), 3) == pytest.approx(expected)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_for_a_miss():
    assert ndcg_at_k(["a", "b", "x"], frozenset({"a", "b"}), 3) == 1.0
    assert ndcg_at_k(["x", "y"], frozenset({"a"}), 2) == 0.0


def test_ndcg_ideal_is_capped_at_k_not_at_the_gold_count():
    # 3 gold chunks but k=1: finding one of them at rank 1 is the best achievable,
    # so this must be 1.0. Dividing by an IDCG over all 3 would report 0.55 and
    # make a perfect result look mediocre.
    assert ndcg_at_k(["a", "b", "c"], frozenset({"a", "b", "c"}), 1) == 1.0


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #

def test_ci_of_identical_scores_has_zero_width():
    # Every resample is the same number, so there is nothing to be uncertain about.
    lo, hi = bootstrap_mean_ci([0.4] * 20, iterations=500)
    assert lo == hi                      # every resample is identical
    assert lo == pytest.approx(0.4)      # summing 20 copies of 0.4 is not exactly 8.0


def test_ci_brackets_the_mean_and_is_narrower_with_more_queries():
    noisy = [0.0, 1.0] * 10            # 20 queries, mean 0.5
    calmer = [0.0, 1.0] * 100          # 200 queries, same mean
    lo_n, hi_n = bootstrap_mean_ci(noisy, iterations=2000)
    lo_c, hi_c = bootstrap_mean_ci(calmer, iterations=2000)

    assert lo_n <= 0.5 <= hi_n
    assert lo_c <= 0.5 <= hi_c
    # The point of reporting intervals at all: 42 questions is a wide interval.
    assert (hi_c - lo_c) < (hi_n - lo_n)


def test_ci_is_reproducible():
    # A README may only quote numbers its own commands reproduce, so the interval
    # must not move between runs.
    values = [0.0, 0.25, 0.5, 0.75, 1.0] * 4
    assert bootstrap_mean_ci(values, iterations=1000) == bootstrap_mean_ci(
        values, iterations=1000
    )


def test_ci_handles_degenerate_input():
    assert bootstrap_mean_ci([]) == (0.0, 0.0)
    assert bootstrap_mean_ci([0.7]) == (0.7, 0.7)


def test_paired_delta_detects_a_difference_that_separate_intervals_hide():
    # A system that is better on every question by a small, consistent margin,
    # against questions whose difficulty varies a lot.
    baseline = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] * 5
    candidate = [b + 0.1 for b in baseline]

    base_lo, base_hi = bootstrap_mean_ci(baseline, iterations=2000)
    cand_lo, cand_hi = bootstrap_mean_ci(candidate, iterations=2000)
    # Overlapping intervals — the naive reading is "no difference".
    assert cand_lo < base_hi

    # Paired, the improvement is unambiguous: every question improved by 0.1.
    lo, hi = bootstrap_delta_ci(baseline, candidate, iterations=2000)
    assert lo > 0.0
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_paired_delta_straddles_zero_when_the_change_is_a_wash():
    # Helps as many questions as it harms: the interval must include zero.
    baseline = [0.5] * 20
    candidate = [1.0] * 10 + [0.0] * 10
    lo, hi = bootstrap_delta_ci(baseline, candidate, iterations=2000)
    assert lo < 0.0 < hi


def test_paired_delta_requires_matching_queries():
    with pytest.raises(ValueError, match="same queries"):
        bootstrap_delta_ci([0.1, 0.2], [0.1])
