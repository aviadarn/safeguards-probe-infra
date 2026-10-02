#!/usr/bin/env python3
"""What a probe costs, against what the alternative costs, on the same box.

The argument for detecting misuse from model internals is a cost argument, and
it is usually asserted rather than measured. It has three parts, and this script
measures each one:

  1. Capturing activations during a forward pass the server is ALREADY doing is
     close to free. If it is not, the whole approach is uninteresting, because
     the capture is the only thing a probe adds to the serving path.
  2. Scoring a captured vector is a dot product.
  3. The alternative -- prompting a model to classify -- is a second forward pass
     over a LONGER sequence (the prompt plus the classification instruction).

Only (3) is a real cost, and the point of the measurement is the ratio.

Timings are MPS on a MacBook and are not throughput numbers for a serving fleet.
The ratio between the three is what transfers; the absolute milliseconds do not.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from probeinfra.cache import ActivationCache
from probeinfra.data import toxicchat, wildguard
from probeinfra.extract import resolve_key
from probeinfra.probe import cross_val_auc, fit_probe, score_probe


def time_forward(model, enc, hidden: bool, reps: int) -> float:
    """Median wall time of one forward pass, with and without capturing states."""
    ts = []
    with torch.inference_mode():
        for _ in range(reps + 1):
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            t0 = time.time()
            model(**enc, output_hidden_states=hidden)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            ts.append(time.time() - t0)
    return float(np.median(ts[1:]))          # drop the first: warm-up


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results/cost.json")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device}  model {a.model}")

    ps = wildguard(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, dtype=torch.float16).to(device).eval()

    # --- 1. what capturing activations adds to a pass already happening -------
    enc = tok(ps.text[:a.batch_size], return_tensors="pt", padding=True,
              truncation=True, max_length=1024).to(device)
    plain = time_forward(model, enc, hidden=False, reps=a.reps)
    hooked = time_forward(model, enc, hidden=True, reps=a.reps)
    n = a.batch_size
    print(f"\nforward, no capture   {1000*plain/n:7.2f} ms/prompt")
    print(f"forward, all layers   {1000*hooked/n:7.2f} ms/prompt  "
          f"(+{100*(hooked-plain)/plain:.1f}%)")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    # --- 2. scoring a captured vector ---------------------------------------
    cache = ActivationCache(a.cache)
    key = resolve_key(a.model, ps)
    X = cache.slice(key, a.layer, a.pooling)
    y = np.asarray(ps.y)
    fitted = fit_probe(X, y)
    reps = 50
    t0 = time.time()
    for _ in range(reps):
        score_probe(fitted, X)
    score_s = (time.time() - t0) / reps
    per_prompt_us = 1e6 * score_s / len(y)
    print(f"probe score           {per_prompt_us:7.1f} us/prompt "
          f"({len(y)} prompts in {1000*score_s:.1f} ms)")

    # --- 3. the alternative: prompt the same model to classify ---------------
    from probeinfra.guard import load_guard, run_guard

    loaded = load_guard(a.model)   # once, not once per prompt set

    sets = {"wildguard-vanilla": ps,
            "wildguard-adversarial": wildguard(True),
            "toxicchat-toxicity": toxicchat("test", "toxicity"),
            "toxicchat-jailbreaking": toxicchat("test", "jailbreaking")}
    from probeinfra.probe import auc_ci
    guard_out, cached = {}, {}
    for name, s in sets.items():
        # The two ToxicChat label columns share their prompts, so the guard runs
        # once -- the same reason `prompt_set` is out of the cache key.
        fp = s.fingerprint
        if fp in cached:
            g = cached[fp]
            print(f"  guard reuse for {name} (same prompts as a scored set)")
        else:
            print(f"\nguard on {name} (n={len(s)})")
            g = run_guard(a.model, s.text, batch_size=a.batch_size,
                          loaded=loaded)
            cached[fp] = g
        auc, lo, hi = auc_ci(np.asarray(s.y), g.scores, n_boot=600)
        guard_out[name] = {"auc": auc, "lo": lo, "hi": hi, "n": len(s),
                           "wall_s": g.wall_s, "per_prompt_ms": g.per_prompt_ms,
                           "prompt_tokens": g.n_prompt_tokens}
        print(f"  AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"{g.per_prompt_ms:.1f} ms/prompt")

    guard_ms = float(np.median([v["per_prompt_ms"] for v in guard_out.values()]))
    res = {
        "model": a.model, "device": device, "layer": a.layer, "pooling": a.pooling,
        "batch_size": a.batch_size,
        "forward_no_capture_ms": 1000 * plain / n,
        "forward_with_capture_ms": 1000 * hooked / n,
        "capture_overhead_pct": 100 * (hooked - plain) / plain,
        "probe_score_us": per_prompt_us,
        "guard_ms_median": guard_ms,
        "guard": guard_out,
        "marginal_ratio": guard_ms / (1000 * (hooked - plain) / n
                                      + per_prompt_us / 1000),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    print(f"marginal cost ratio, guard : probe = {res['marginal_ratio']:.0f}x")


if __name__ == "__main__":
    main()
