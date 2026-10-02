"""Content-addressed activation cache.

The reason this file exists, and the reason it is the first thing in the repo:

    "Build correctness and sanity checking into the stack, so results stay
     trustworthy as models and workloads evolve."

An activation cache is the single highest-leverage object in a probe pipeline —
one GPU pass, then hundreds of CPU-seconds sweeps over layers and poolings — and
it is also the single easiest place to serve a result from a model that no longer
exists. `model="meta-llama/Llama-3.1-8B"` is not an identity. The same string
resolves to different weights across a repo revision, a dtype change, a
tokenizer bump or a chat-template edit, and every one of those changes the
activations while leaving the filename alone.

So the key is the hash of everything that can move the numbers, and the resolved
commit sha is in it. A model that moved under you is a cache MISS, not a silent
hit. `verify()` is what turns that from a convention into a check.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = 3  # bump to invalidate every key on disk

# Fields kept in the manifest for a human reading it, but deliberately OUT of the
# digest, because they do not change a single activation.
#
# `prompt_set` is the one that matters. ToxicChat's toxicity and jailbreaking
# labels annotate the SAME 2,803 user turns; only the label column differs.
# Keyed on the set name, the two asked for different digests and the model ran
# twice — 36 of 155 seconds of the first full run, for byte-identical output.
# The prompts are already identified by `prompt_fingerprint`, which is a hash of
# their text, so the name adds nothing to the identity and costs a GPU pass.
DIGEST_EXCLUDE = ("prompt_set",)


@dataclass(frozen=True)
class CacheKey:
    """Everything that changes an activation. Nothing that does not."""

    model_id: str
    model_sha: str          # resolved commit, never a branch name
    tokenizer_sha: str
    dtype: str
    layers: tuple[int, ...]
    poolings: tuple[str, ...]
    padding_side: str       # part of the key: it changes `last_raw` entirely
    max_length: int
    chat_template_sha: str  # "none" when prompts are fed raw
    prompt_set: str
    prompt_fingerprint: str
    schema: int = SCHEMA
    extras: dict = field(default_factory=dict)

    def digest(self) -> str:
        fields = {k: v for k, v in asdict(self).items() if k not in DIGEST_EXCLUDE}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def path(self, root: str | os.PathLike) -> Path:
        return Path(root) / f"{self.model_id.replace('/', '__')}" / self.digest()


def sha_of(text: str | None) -> str:
    if text is None:
        return "none"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def resolve_revision(model_id: str) -> str:
    """Pin `main` to the commit it points at right now.

    Without this the cache key says "main" and a cache written against last
    month's weights answers today's question.
    """
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


class ActivationCache:
    """A directory per key: one float16 array plus the manifest that explains it.

    Layout
        <root>/<model>/<digest>/acts.npy      [n_prompts, n_layers, n_pool, d]
        <root>/<model>/<digest>/key.json      the CacheKey, in full
        <root>/<model>/<digest>/pids.json     prompt order, for row lookup
    """

    def __init__(self, root: str | os.PathLike = "cache"):
        self.root = Path(root)

    def dir_for(self, key: CacheKey) -> Path:
        return key.path(self.root)

    def exists(self, key: CacheKey) -> bool:
        d = self.dir_for(key)
        return (d / "acts.npy").exists() and (d / "key.json").exists()

    def write(self, key: CacheKey, acts: np.ndarray, pids: list[str]) -> Path:
        if acts.shape[0] != len(pids):
            raise ValueError(f"{acts.shape[0]} rows for {len(pids)} prompts")
        expect = (len(pids), len(key.layers), len(key.poolings))
        if acts.shape[:3] != expect:
            raise ValueError(f"acts {acts.shape[:3]} does not match key {expect}")
        d = self.dir_for(key)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "acts.npy", acts.astype(np.float16))
        (d / "pids.json").write_text(json.dumps(pids))
        (d / "key.json").write_text(json.dumps(asdict(key), indent=2, sort_keys=True))
        return d

    def read(self, key: CacheKey) -> tuple[np.ndarray, list[str]]:
        d = self.dir_for(key)
        if not self.exists(key):
            raise FileNotFoundError(f"no cache at {d}")
        self.verify(key)
        acts = np.load(d / "acts.npy", mmap_mode="r")
        pids = json.loads((d / "pids.json").read_text())
        return acts, pids

    def verify(self, key: CacheKey) -> None:
        """Refuse a hit whose stored key is not byte-identical to the one asked for.

        A digest collision is not the threat here. The threat is someone editing
        a key field, reusing the directory, and getting an answer about the old
        configuration. This check costs a file read and removes that class.
        """
        d = self.dir_for(key)
        stored = json.loads((d / "key.json").read_text())
        want = asdict(key)
        # tuples round-trip through JSON as lists
        want = json.loads(json.dumps(want))
        keys = (set(stored) | set(want)) - set(DIGEST_EXCLUDE)
        diff = {k: (stored.get(k), want.get(k)) for k in keys
                if stored.get(k) != want.get(k)}
        if diff:
            raise ValueError(
                f"cache key mismatch at {d} (stored, requested): {diff}\n"
                "The digest matched but the key did not. Delete the directory."
            )

    def slice(self, key: CacheKey, layer: int, pooling: str,
              acts: np.ndarray | None = None) -> np.ndarray:
        """One [n, d] feature matrix. This is the operation a sweep repeats."""
        if layer not in key.layers:
            raise KeyError(f"layer {layer} not cached; have {key.layers}")
        if pooling not in key.poolings:
            raise KeyError(f"pooling {pooling!r} not cached; have {key.poolings}")
        if acts is None:
            acts, _ = self.read(key)
        return np.asarray(acts[:, key.layers.index(layer), key.poolings.index(pooling), :],
                          dtype=np.float32)
