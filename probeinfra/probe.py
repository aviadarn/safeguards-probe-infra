"""Fit and score linear probes on cached activations.

The probe itself is deliberately boring — standardise, logistic regression, one
regularisation path. Nothing here is a research contribution and it should not
look like one. What the file does carry is the evaluation protocol, because that
is what decides whether a number means anything:

  iid    fit and score on disjoint halves of the same split
  shift  fit on plain prompts, score on the jailbreak-framed ones
  ood    fit on WildGuard, score on real user turns from ToxicChat

A probe that is only ever reported on `iid` is reported on the easiest of the
three by a wide margin, and `shift` is the setting Safeguards actually operates
in: the phrasing moves, the harm does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000,
           seed: int = 0) -> tuple[float, float, float]:
    """AUC with a stratified bootstrap interval.

    Stratified so every resample keeps the class balance; on ToxicChat the
    positive rate is 12.6% and an unstratified resample can draw zero positives,
    which makes AUC undefined and the interval quietly wrong.
    """
    y, s = np.asarray(y), np.asarray(s)
    point = roc_auc_score(y, s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        draws[b] = roc_auc_score(y[idx], s[idx])
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


@dataclass
class ProbeResult:
    setting: str
    layer: int
    pooling: str
    n_train: int
    n_test: int
    auc: float
    lo: float
    hi: float
    extra: dict = field(default_factory=dict)

    def row(self) -> str:
        return (f"{self.setting:<10} L{self.layer:<3} {self.pooling:<9} "
                f"AUC {self.auc:.3f} [{self.lo:.3f}, {self.hi:.3f}]  "
                f"n={self.n_train}/{self.n_test}")


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 0):
    """Standardise then logistic-regress. Returns (scaler, model)."""
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    lr.fit(sc.transform(X), y)
    return sc, lr


def score_probe(fitted, X: np.ndarray) -> np.ndarray:
    sc, lr = fitted
    return lr.decision_function(sc.transform(X))


def evaluate(Xtr, ytr, Xte, yte, setting: str, layer: int, pooling: str,
             C: float = 1.0, seed: int = 0, n_boot: int = 2000) -> ProbeResult:
    fitted = fit_probe(Xtr, ytr, C=C, seed=seed)
    s = score_probe(fitted, Xte)
    auc, lo, hi = auc_ci(yte, s, n_boot=n_boot, seed=seed)
    return ProbeResult(setting, layer, pooling, len(ytr), len(yte), auc, lo, hi)


def cross_val_auc(X: np.ndarray, y: np.ndarray, k: int = 5, C: float = 1.0,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores for the iid setting.

    Out-of-fold rather than a single split so the iid number does not depend on
    which half happened to be held out, and so every row is scored exactly once
    by a probe that never saw it.
    """
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        oof[te] = score_probe(fit_probe(X[tr], y[tr], C=C, seed=seed), X[te])
    return oof, y


def sweep(cache, key, y: np.ndarray, layers=None, poolings=None, k: int = 5,
          C: float = 1.0, seed: int = 0, n_boot: int = 400) -> list[ProbeResult]:
    """Every (layer, pooling) in the cache, out-of-fold. No GPU touched.

    This is the operation the cache exists for. The whole sweep reads one
    memory-mapped array.
    """
    acts, _ = cache.read(key)
    out = []
    for pooling in (poolings or key.poolings):
        for layer in (layers or key.layers):
            X = cache.slice(key, layer, pooling, acts=acts)
            s, yy = cross_val_auc(X, y, k=k, C=C, seed=seed)
            auc, lo, hi = auc_ci(yy, s, n_boot=n_boot, seed=seed)
            out.append(ProbeResult("iid", layer, pooling, len(y) * (k - 1) // k,
                                   len(y), auc, lo, hi))
    return out
