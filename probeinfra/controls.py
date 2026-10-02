"""The controls. Run these before reading any probe number.

Every artifact in this repo exists because of one observation: a probe pipeline
does not fail by crashing. It fails by returning 0.9 for the wrong reason. These
four controls each pin down one wrong reason, and three of them need no GPU.

  length_only          counting characters. On WildGuard-vanilla this scores
                       0.737 AUC, so a probe reporting 0.80 has cleared the
                       floor by 0.06, not by 0.30.
  shuffled_labels      the same probe on permuted labels. Must contain 0.5. If
                       it does not, the evaluation leaks — usually a scaler fit
                       on the test rows, or a duplicated prompt across folds.
  random_features      Gaussian noise of the same shape. Must contain 0.5. This
                       tests the harness, not the model: it is the control that
                       catches a bug in `evaluate` itself.
  length_matched       AUC inside narrow length bands. Not a residualisation —
                       fitting length out linearly over-corrects and flips the
                       sign of the thing you are measuring. Stratifying is the
                       conservative version of the same question.
"""
from __future__ import annotations

import numpy as np

from .probe import ProbeResult, auc_ci, cross_val_auc


def length_only(texts: list[str], y: np.ndarray, unit: str = "chars",
                n_boot: int = 2000, seed: int = 0) -> ProbeResult:
    f = np.array([len(t) if unit == "chars" else len(t.split()) for t in texts],
                 dtype=float)
    auc, lo, hi = auc_ci(y, f, n_boot=n_boot, seed=seed)
    return ProbeResult(f"length-{unit}", -1, "none", 0, len(y), auc, lo, hi)


def shuffled_labels(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0,
                    n_boot: int = 2000) -> ProbeResult:
    yp = np.random.default_rng(seed).permutation(y)
    s, yy = cross_val_auc(X, yp, k=k, seed=seed)
    auc, lo, hi = auc_ci(yy, s, n_boot=n_boot, seed=seed)
    r = ProbeResult("shuffled", -1, "none", 0, len(y), auc, lo, hi)
    r.extra["passes"] = lo <= 0.5 <= hi
    return r


def random_features(n: int, d: int, y: np.ndarray, k: int = 5, seed: int = 0,
                    n_boot: int = 2000) -> ProbeResult:
    X = np.random.default_rng(seed).standard_normal((n, d))
    s, yy = cross_val_auc(X, y, k=k, seed=seed)
    auc, lo, hi = auc_ci(yy, s, n_boot=n_boot, seed=seed)
    r = ProbeResult("random-X", -1, "none", 0, len(y), auc, lo, hi)
    r.extra["passes"] = lo <= 0.5 <= hi
    return r


def length_matched(texts: list[str], y: np.ndarray, scores: np.ndarray,
                   n_bins: int = 4, seed: int = 0) -> list[ProbeResult]:
    """Probe AUC within length quartiles, where length cannot explain it.

    A bin with one class present is reported as n/a rather than dropped
    silently — the reader needs to see that the comparison was not available
    there, because that is itself a fact about the dataset.
    """
    L = np.array([len(t) for t in texts], dtype=float)
    edges = np.quantile(L, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1
    out = []
    for b in range(n_bins):
        m = (L >= edges[b]) & (L < edges[b + 1])
        yy = np.asarray(y)[m]
        if len(np.unique(yy)) < 2:
            r = ProbeResult(f"len-q{b+1}", -1, "none", 0, int(m.sum()),
                            float("nan"), float("nan"), float("nan"))
            r.extra["note"] = "one class only in this band"
        else:
            auc, lo, hi = auc_ci(yy, np.asarray(scores)[m], n_boot=1000, seed=seed)
            r = ProbeResult(f"len-q{b+1}", -1, "none", 0, int(m.sum()), auc, lo, hi)
        r.extra["chars"] = (float(edges[b]), float(edges[b + 1]))
        r.extra["pos_rate"] = float(yy.mean()) if len(yy) else float("nan")
        out.append(r)
    return out
