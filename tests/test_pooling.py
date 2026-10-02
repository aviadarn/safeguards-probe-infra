"""Pooling is where a probe pipeline reads the wrong vector and still reports a score.

These are arithmetic tests on a hand-built tensor: no model, no download. They
pin down exactly which pooling is wrong under which padding side, which is the
claim the README makes.
"""
from __future__ import annotations

import pytest
import torch

from probeinfra.extract import pool

D = 4


def seq(vals):
    """[1, L, D] where row t is filled with vals[t]."""
    return torch.tensor([[[v] * D for v in vals]], dtype=torch.float32)


def test_last_ignores_padding_and_last_raw_does_not():
    # Right padding: two real tokens (1, 2) then two pads (99 = whatever the
    # model computed at a pad position; it is not zero in practice).
    h = seq([1.0, 2.0, 99.0, 99.0])
    mask = torch.tensor([[1, 1, 0, 0]])
    assert pool(h, mask, "last")[0, 0] == 2.0, "last must land on the last REAL token"
    assert pool(h, mask, "last_raw")[0, 0] == 99.0, \
        "last_raw reads a pad position under right padding — this is the bug"


def test_left_padding_makes_last_raw_accidentally_correct():
    """Why padding_side belongs in the cache key.

    The same pooling name is right or wrong depending on a tokenizer setting
    that lives nowhere near the pooling code.
    """
    h = seq([99.0, 99.0, 1.0, 2.0])
    mask = torch.tensor([[0, 0, 1, 1]])
    assert pool(h, mask, "last")[0, 0] == 2.0
    assert pool(h, mask, "last_raw")[0, 0] == 2.0


def test_mean_excludes_pads_and_mean_raw_dilutes_with_them():
    h = seq([1.0, 3.0, 99.0, 99.0])
    mask = torch.tensor([[1, 1, 0, 0]])
    assert pool(h, mask, "mean")[0, 0] == 2.0
    assert pool(h, mask, "mean_raw")[0, 0] == pytest.approx(50.5)


def test_mean_raw_error_scales_with_padding_so_it_encodes_length():
    """The insidious part: the error is a function of how much padding a row got,
    and padding is a function of prompt length, so `mean_raw` leaks length into
    every feature. On this dataset length alone scores 0.737 AUC."""
    pads = []
    for n_pad in (0, 2, 6):
        h = seq([1.0, 1.0] + [99.0] * n_pad)
        mask = torch.tensor([[1, 1] + [0] * n_pad])
        pads.append(pool(h, mask, "mean_raw")[0, 0].item())
    assert pads == sorted(pads) and pads[0] == 1.0
    assert len(set(pads)) == 3, "mean_raw must vary with pad count — that is the leak"


def test_last_handles_a_full_row_and_a_single_token():
    h = seq([1.0, 2.0, 3.0])
    assert pool(h, torch.tensor([[1, 1, 1]]), "last")[0, 0] == 3.0
    assert pool(h, torch.tensor([[1, 0, 0]]), "last")[0, 0] == 1.0


def test_batched_rows_with_different_real_lengths_each_get_their_own_last():
    h = torch.tensor([[[1.0], [2.0], [99.0]],
                      [[5.0], [6.0], [7.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    got = pool(h, mask, "last")[:, 0]
    assert got.tolist() == [2.0, 7.0]


def test_unknown_pooling_raises():
    with pytest.raises(KeyError):
        pool(seq([1.0]), torch.tensor([[1]]), "attention_weighted")


def test_fp16_activation_mass_check_needs_float32():
    """Regression: the all-zeros guard in `extract` summed |fp16| and overflowed.

    29 layers x 4 poolings x 1024 hidden of ordinary activations sums well past
    fp16's 65504 ceiling, so the reduction returned inf. `inf > 0` is True, so
    the assertion passed on every row — including a row that was genuinely zero
    everywhere except one element.
    """
    import numpy as np

    out = np.full((2, 29, 4, 1024), 3.0, dtype=np.float16)
    assert not np.isfinite(np.abs(out).sum(axis=(1, 2, 3))).all(), \
        "if this stops overflowing, the guard below is no longer load-bearing"
    mass = np.abs(out.astype(np.float32)).sum(axis=(1, 2, 3))
    assert np.isfinite(mass).all() and mass.min() > 0
