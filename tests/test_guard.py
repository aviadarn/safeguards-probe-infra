"""The guard baseline has to be fair, or the cost comparison is worthless."""
from __future__ import annotations

import pytest

from probeinfra.guard import PROMPT, answer_token_ids


class FakeTok:
    """Single-token for the spaced forms only — the realistic case."""
    def encode(self, s, add_special_tokens=False):
        return {" harmful": [11], " unharmful": [22], " safe": [33]}.get(s, [1, 2])


def test_both_whitespace_variants_are_searched():
    pos, neg = answer_token_ids(FakeTok())
    assert pos == [11] and set(neg) == {22, 33}


def test_it_refuses_rather_than_scoring_on_a_truncated_token():
    class NoSingles:
        def encode(self, s, add_special_tokens=False):
            return [1, 2]
    with pytest.raises(ValueError, match="not single tokens"):
        answer_token_ids(NoSingles())


def test_the_prompt_asks_for_one_word_and_contains_the_request():
    f = PROMPT.format(prompt="PAYLOAD")
    assert "PAYLOAD" in f
    assert "exactly one word" in f
    assert f.rstrip().endswith("Answer:"), \
        "the scored position must be the token right after 'Answer:'"


def test_guard_scores_are_a_logit_gap_not_a_string_match():
    """Why the guard is scored on logits rather than on generated text.

    String-matching "harmful"/"unharmful" in generated output charges the guard
    for formatting mistakes -- answering "Harmful." or "This is harmful" scores as
    a miss. That is the same error this project's companion audit found a
    benchmark making against models, so the comparison here reads the two answer
    tokens' logits directly and takes their difference.
    """
    import inspect

    from probeinfra import guard

    src = inspect.getsource(guard.run_guard)
    assert "logits" in src and "[:, -1, :]" in src
    assert ".generate(" not in src, "no sampling: the comparison must be deterministic"
