"""Baselines a probe has to beat, both of which are free.

`length_only` lives in controls.py because it is a control — it rules out an
explanation. The two here are something else: they are *rivals*. A probe on model
internals is worth building only if it beats what a team could deploy instead,
and on this dataset two such things cost nothing to measure:

  openai_moderation   ToxicChat ships the OpenAI moderation endpoint's own
                      category scores alongside every human-annotated row. That
                      is a production detector's output on the exact prompts the
                      probe is scored on, with no API call and no key.
  length              `len(prompt)`. Not a rival anyone would deploy, which is
                      the point: it is the number that makes the others readable.

The result that justifies the file: on real jailbreak attempts the moderation
API scores 0.856 and `len(prompt)` scores 0.916. A deployed commercial detector
loses to character count by six points, and no paper reports that because nobody
runs the trivial baseline.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .data import TOXICCHAT, PromptSet
from .probe import ProbeResult, auc_ci


def openai_moderation_scores(ps: PromptSet, split: str = "test") -> np.ndarray:
    """Max category score from the moderation output ToxicChat already carries.

    Aligned by prompt text, not by row order, because `data.py` deduplicates and
    the raw CSV does not. A positional join here would silently shift the scores
    against the labels and produce a plausible, wrong AUC — the exact failure
    this repo is about.
    """
    raw = pd.read_csv(hf_hub_download(TOXICCHAT[0], TOXICCHAT[1].format(split=split),
                                      repo_type="dataset"))
    raw = raw[raw.openai_moderation.notna()]
    by_text = {}
    for text, mod in zip(raw.user_input, raw.openai_moderation):
        by_text.setdefault(str(text), mod)
    missing = [t for t in ps.text if t not in by_text]
    if missing:
        raise KeyError(f"{len(missing)} prompts have no moderation score; "
                       "the join is not sound")
    return np.array([max(v for _, v in ast.literal_eval(by_text[t])) for t in ps.text])


def openai_moderation(ps: PromptSet, n_boot: int = 2000, seed: int = 0) -> ProbeResult:
    s = openai_moderation_scores(ps)
    auc, lo, hi = auc_ci(np.asarray(ps.y), s, n_boot=n_boot, seed=seed)
    return ProbeResult("openai-moderation", -1, "none", 0, len(ps), auc, lo, hi)
