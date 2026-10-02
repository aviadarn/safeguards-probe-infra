"""The cache key is the correctness story, so it gets the tests."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from probeinfra.cache import ActivationCache, CacheKey

BASE = dict(
    model_id="Qwen/Qwen3-0.6B", model_sha="a" * 40, tokenizer_sha="b" * 16,
    dtype="float16", layers=(0, 4, 8), poolings=("last", "mean"),
    padding_side="left", max_length=512, chat_template_sha="none",
    prompt_set="wildguard-vanilla", prompt_fingerprint="c" * 16,
)


def key(**over) -> CacheKey:
    return CacheKey(**{**BASE, **over})


@pytest.mark.parametrize("field,value", [
    ("model_sha", "d" * 40),          # the repo moved
    ("tokenizer_sha", "e" * 16),      # tokenizer bumped
    ("dtype", "bfloat16"),            # different rounding
    ("padding_side", "right"),        # changes `last_raw` completely
    ("max_length", 256),              # truncates different prompts
    ("chat_template_sha", "f" * 16),  # prompts are wrapped differently
    ("prompt_fingerprint", "0" * 16),  # different prompts, same name
    ("schema", 99),
])
def test_every_field_that_moves_activations_moves_the_digest(field, value):
    assert key().digest() != key(**{field: value}).digest(), (
        f"{field} changes the activations but not the cache key — "
        "this is how a run gets answered by a model that no longer exists"
    )


def test_digest_is_order_independent_but_content_dependent():
    assert key().digest() == key().digest()
    assert key(layers=(0, 4, 8)).digest() != key(layers=(8, 4, 0)).digest(), \
        "layer order indexes into the array, so it is part of the identity"


def test_roundtrip(tmp_path):
    c, k = ActivationCache(tmp_path), key()
    acts = np.random.randn(5, 3, 2, 7).astype(np.float16)
    pids = [f"p{i}" for i in range(5)]
    c.write(k, acts, pids)
    got, got_pids = c.read(k)
    assert got_pids == pids
    np.testing.assert_array_equal(np.asarray(got), acts)
    np.testing.assert_allclose(c.slice(k, 4, "mean"), acts[:, 1, 1, :].astype(np.float32))


def test_shape_disagreeing_with_the_key_is_refused(tmp_path):
    c, k = ActivationCache(tmp_path), key()
    with pytest.raises(ValueError, match="does not match key"):
        c.write(k, np.zeros((5, 2, 2, 7), dtype=np.float16), [f"p{i}" for i in range(5)])
    with pytest.raises(ValueError, match="rows for"):
        c.write(k, np.zeros((4, 3, 2, 7), dtype=np.float16), [f"p{i}" for i in range(5)])


def test_tampered_key_file_is_caught_not_served(tmp_path):
    """The failure this whole design exists to prevent."""
    c, k = ActivationCache(tmp_path), key()
    c.write(k, np.zeros((2, 3, 2, 7), dtype=np.float16), ["p0", "p1"])
    d = c.dir_for(k)
    stored = json.loads((d / "key.json").read_text())
    stored["model_sha"] = "9" * 40          # pretend a different model wrote this
    (d / "key.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="cache key mismatch"):
        c.read(k)


def test_unknown_layer_or_pooling_raises_rather_than_returning_a_neighbour(tmp_path):
    c, k = ActivationCache(tmp_path), key()
    c.write(k, np.zeros((2, 3, 2, 7), dtype=np.float16), ["p0", "p1"])
    with pytest.raises(KeyError):
        c.slice(k, 5, "last")
    with pytest.raises(KeyError):
        c.slice(k, 4, "max")


def test_two_label_sets_over_the_same_prompts_share_one_extraction():
    """The waste this fix removes.

    ToxicChat's `toxicity` and `jailbreaking` columns label the same 2,803 user
    turns. Activations do not depend on the label, so both must resolve to one
    digest and one GPU pass.
    """
    a = key(prompt_set="toxicchat-test-toxicity")
    b = key(prompt_set="toxicchat-test-jailbreaking")
    assert a.digest() == b.digest()


def test_different_prompts_under_the_same_name_still_separate():
    """The other half: `prompt_set` leaving the digest must not let a different
    set of prompts answer for this one. The fingerprint is what carries that."""
    a = key(prompt_set="same-name", prompt_fingerprint="1" * 16)
    b = key(prompt_set="same-name", prompt_fingerprint="2" * 16)
    assert a.digest() != b.digest()


def test_verify_tolerates_only_the_excluded_field(tmp_path):
    """Reading a cache written under another set NAME is fine; anything else is not."""
    import json as _json
    c = ActivationCache(tmp_path)
    written = key(prompt_set="toxicchat-test-toxicity")
    c.write(written, np.zeros((2, 3, 2, 7), dtype=np.float16), ["p0", "p1"])
    c.read(key(prompt_set="toxicchat-test-jailbreaking"))     # must not raise
    d = c.dir_for(written)
    stored = _json.loads((d / "key.json").read_text())
    stored["dtype"] = "bfloat16"
    (d / "key.json").write_text(_json.dumps(stored))
    with pytest.raises(ValueError, match="cache key mismatch"):
        c.read(key(prompt_set="toxicchat-test-jailbreaking"))
