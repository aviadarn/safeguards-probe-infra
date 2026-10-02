"""The moderation baseline is a join, and a join is where scores detach from labels."""
from __future__ import annotations

import numpy as np
import pytest

from probeinfra.baselines import openai_moderation, openai_moderation_scores
from probeinfra.data import toxicchat

pytestmark = pytest.mark.slow


def test_scores_align_by_text_not_by_row():
    """`data.py` drops 50 duplicate prompts; the raw CSV keeps them. A positional
    join would shift every score after the first duplicate."""
    ps = toxicchat("test", "toxicity")
    s = openai_moderation_scores(ps)
    assert len(s) == len(ps) == 2803
    assert ((s >= 0) & (s <= 1)).all()
    # Shuffling the PromptSet must shuffle the scores with it.
    idx = np.random.default_rng(0).permutation(len(ps))
    shuffled = type(ps)(name=ps.name, pid=[ps.pid[i] for i in idx],
                        text=[ps.text[i] for i in idx],
                        y=[ps.y[i] for i in idx], meta=ps.meta)
    np.testing.assert_allclose(openai_moderation_scores(shuffled), s[idx])


def test_the_two_headline_baseline_numbers():
    tox = openai_moderation(toxicchat("test", "toxicity"), n_boot=300)
    jb = openai_moderation(toxicchat("test", "jailbreaking"), n_boot=300)
    assert tox.auc == pytest.approx(0.913, abs=0.004), tox.auc
    assert jb.auc == pytest.approx(0.856, abs=0.004), jb.auc


def test_length_beats_the_moderation_api_on_jailbreaking():
    """The README's sharpest claim, pinned."""
    from probeinfra.controls import length_only

    ps = toxicchat("test", "jailbreaking")
    api = openai_moderation(ps, n_boot=300)
    length = length_only(ps.text, np.asarray(ps.y), n_boot=300)
    assert length.auc > api.auc + 0.04, (
        f"length {length.auc:.3f} vs moderation API {api.auc:.3f}")
