"""Pin the dataset facts the README quotes.

These hit the network once and then the local `hf` cache. They are here because
the README's headline is a property of the *data* — harmful prompts are longer —
and a quiet upstream revision that fixed that would make the headline false
while every other test still passed.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from probeinfra.controls import length_only
from probeinfra.data import toxicchat, wildguard

pytestmark = pytest.mark.slow


def test_wildguard_null_labels_are_dropped_not_cast():
    """26 of 1,725 rows ship a null label. `label == "harmful"` makes them
    negatives, which quietly moves the base rate and every AUC with it."""
    ps = wildguard(None)
    assert len(ps) == 1699, f"expected 1699 usable rows, got {len(ps)}"
    assert int(ps.meta["_dropped_null_labels"].iloc[0]) == 26
    assert np.mean(ps.y) == pytest.approx(0.444, abs=0.002)


def test_the_length_confound_the_readme_leads_with():
    v = wildguard(False)
    r = length_only(v.text, np.asarray(v.y), n_boot=200)
    assert r.auc == pytest.approx(0.737, abs=0.005), (
        f"the README's floor is 0.737; measured {r.auc:.3f}. If the data moved, "
        "the headline moved with it")
    L = np.array([len(t) for t in v.text])
    y = np.asarray(v.y)
    assert L[y == 1].mean() / L[y == 0].mean() == pytest.approx(1.9, abs=0.1)


def test_adversarial_prompts_are_about_six_times_longer():
    """Why `vanilla -> adversarial` is a distribution shift and not just a split."""
    a, v = wildguard(True), wildguard(False)
    ratio = np.mean([len(t) for t in a.text]) / np.mean([len(t) for t in v.text])
    assert 5.0 < ratio < 8.0, ratio


def test_a_length_rule_fit_on_vanilla_is_worse_than_the_majority_class_on_adversarial():
    v, a = wildguard(False), wildguard(True)
    thr = np.median([len(t) for t in v.text])
    pred = (np.array([len(t) for t in a.text]) > thr).astype(int)
    y = np.asarray(a.y)
    majority = max(y.mean(), 1 - y.mean())
    assert (pred == y).mean() < majority, (
        "the vanilla length threshold must fail on adversarial prompts — "
        f"got {(pred == y).mean():.3f} against a {majority:.3f} majority class")


def test_toxicchat_label_columns_share_their_prompts():
    """The fact that makes one extraction serve both. If upstream ever gives the
    two columns different row sets, the shared cache key becomes wrong."""
    t, j = toxicchat("test", "toxicity"), toxicchat("test", "jailbreaking")
    assert t.fingerprint == j.fingerprint
    assert t.pid == j.pid and t.y != j.y


def test_toxicchat_is_rare_positive_which_is_why_the_bootstrap_is_stratified():
    t = toxicchat("test", "toxicity")
    assert 0.10 < np.mean(t.y) < 0.15
    j = toxicchat("test", "jailbreaking")
    assert np.mean(j.y) < 0.05


def test_dedup_removes_repeated_prompts():
    """Duplicate prompts across CV folds manufacture signal from noise
    (tests/test_controls.py proves it), so the loaders drop them."""
    t = toxicchat("test", "toxicity")
    assert len(set(t.text)) == len(t.text)
    assert len(t) == 2803, len(t)
