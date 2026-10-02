"""If the controls do not fail on broken input, they are decoration."""
from __future__ import annotations

import numpy as np

from probeinfra.controls import length_matched, length_only, random_features, shuffled_labels
from probeinfra.probe import auc_ci, cross_val_auc


def test_shuffled_labels_lands_on_chance_for_real_signal():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 150)
    X = rng.standard_normal((300, 20)) + y[:, None] * 1.5   # genuinely separable
    assert cross_val_auc(X, y)[0] is not None
    r = shuffled_labels(X, y, n_boot=400)
    assert r.extra["passes"], f"shuffled control must contain 0.5, got {r.auc:.3f}"


def test_duplicate_prompts_across_folds_manufacture_signal_from_noise():
    """Why `data.py` deduplicates on prompt text.

    Pure noise, no signal. Duplicate each row so every held-out row has its twin
    (same features, same label) sitting in the training fold, and the probe
    "finds" a signal that is not there. This is the leak the dedup prevents, and
    the reason the shuffled-label control is run on the deduplicated set.
    """
    rng = np.random.default_rng(1)
    n, d = 60, 100                      # overparameterised, so it can memorise
    X = rng.standard_normal((n, d))
    y = np.array([0, 1] * (n // 2))
    clean, _ = cross_val_auc(X, y, seed=0)
    dup, ydup = cross_val_auc(np.repeat(X, 2, axis=0), np.repeat(y, 2), seed=0)
    from sklearn.metrics import roc_auc_score
    a_clean = roc_auc_score(y, clean)
    a_dup = roc_auc_score(ydup, dup)
    assert a_clean < 0.65, f"no signal was planted, yet clean scored {a_clean:.3f}"
    assert a_dup > a_clean + 0.15, (
        f"duplicated rows must inflate the score: {a_dup:.3f} vs {a_clean:.3f}")


def test_random_features_is_chance():
    y = np.repeat([0, 1], 100)
    r = random_features(200, 16, y, n_boot=400)
    assert r.extra["passes"], f"noise scored {r.auc:.3f}"


def test_length_only_detects_a_pure_length_confound():
    y = np.repeat([0, 1], 100)
    texts = ["x" * (10 + 40 * int(v)) for v in y]
    assert length_only(texts, y, n_boot=200).auc > 0.99


def test_length_only_is_chance_when_length_is_uninformative():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 100)
    texts = ["x" * int(n) for n in rng.integers(10, 50, 200)]
    r = length_only(texts, y, n_boot=400)
    assert r.lo <= 0.5 <= r.hi


def test_length_matched_says_nothing_is_measurable_when_length_is_the_label():
    """The degenerate case, reported rather than hidden.

    Here length predicts the label perfectly, so every length band is
    single-class and there is no within-band comparison left to make. The right
    output is four n/a rows — a residualisation would instead return a number,
    and that number would be an artifact of the fit.
    """
    y = np.array([0] * 50 + [1] * 50)
    texts = ["x" * (5 + i) for i in range(100)]           # length == index
    rs = length_matched(texts, y, np.arange(100.0), n_bins=4)
    assert len(rs) == 4
    assert all(np.isnan(r.auc) for r in rs)
    assert all(r.extra["note"] == "one class only in this band" for r in rs)
    assert all("chars" in r.extra for r in rs)


def test_length_matched_measures_within_band_when_both_classes_are_present():
    rng = np.random.default_rng(5)
    L = rng.integers(10, 400, 400)
    y = rng.integers(0, 2, 400)
    texts = ["x" * int(n) for n in L]
    scores = y + rng.standard_normal(400) * 0.4           # real, length-free signal
    rs = length_matched(texts, y, scores, n_bins=4)
    assert all(not np.isnan(r.auc) for r in rs)
    assert all(r.auc > 0.7 for r in rs), [round(r.auc, 3) for r in rs]
    assert sum(r.n_test for r in rs) == 400


def test_auc_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 200)
    s = rng.standard_normal(400) + y * 0.8
    p, lo, hi = auc_ci(y, s, n_boot=500)
    assert lo < p < hi and 0.5 < p < 1.0


def test_auc_ci_is_stratified_so_rare_positives_never_vanish():
    """Unstratified resampling of a 3% positive rate can draw zero positives and
    make AUC undefined. ToxicChat is 12.6% positive, so this is not theoretical."""
    rng = np.random.default_rng(4)
    y = np.array([1] * 6 + [0] * 194)
    s = rng.standard_normal(200) + y * 1.0
    p, lo, hi = auc_ci(y, s, n_boot=500)
    assert not np.isnan([p, lo, hi]).any()
