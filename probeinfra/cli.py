"""probeinfra — the command line a researcher would actually use.

    python -m probeinfra extract  --model Qwen/Qwen3-0.6B --set wildguard-vanilla
    python -m probeinfra sweep    --model Qwen/Qwen3-0.6B --set wildguard-vanilla
    python -m probeinfra controls --model Qwen/Qwen3-0.6B --set wildguard-vanilla
    python -m probeinfra transfer --model Qwen/Qwen3-0.6B --layer 14

`extract` is the only command that needs a GPU. Everything else reads the cache,
which is the point: the cost of asking one more question about an already-
extracted model is seconds of CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .baselines import openai_moderation
from .cache import ActivationCache
from .controls import length_matched, length_only, random_features, shuffled_labels
from .data import toxicchat, wildguard
from .extract import POOLINGS, extract, resolve_key
from .probe import auc_ci, cross_val_auc, evaluate, sweep

SETS = {
    "wildguard-vanilla": lambda: wildguard(False),
    "wildguard-adversarial": lambda: wildguard(True),
    "wildguard-both": lambda: wildguard(None),
    "toxicchat-toxicity": lambda: toxicchat("test", "toxicity"),
    "toxicchat-jailbreaking": lambda: toxicchat("test", "jailbreaking"),
}


def _rehydrate(cache, key):
    """Rebuild the key from the stored JSON so tuples survive the round trip."""
    stored = json.loads((cache.dir_for(key) / "key.json").read_text())
    return type(key)(**{**stored, "layers": tuple(stored["layers"]),
                        "poolings": tuple(stored["poolings"])})


def get_set(name: str):
    if name not in SETS:
        sys.exit(f"unknown --set {name!r}; choose from {', '.join(SETS)}")
    return SETS[name]()


def parse_layers(spec: str | None, n: int | None = None) -> tuple[int, ...] | None:
    """"all" | "0,8,16" | "0:32:4" """
    if spec in (None, "all"):
        return None
    if ":" in spec:
        a, b, st = (int(x) for x in spec.split(":"))
        return tuple(range(a, b + 1, st))
    return tuple(int(x) for x in spec.split(","))


def _key_for(a, ps):
    """One resolver, shared with `extract`, so a lookup cannot miss its own write."""
    return resolve_key(a.model, ps, parse_layers(a.layers), dtype=a.dtype,
                       padding_side=a.padding_side, max_length=a.max_length,
                       length_sorted=not getattr(a, "no_sort", False))


def cmd_extract(a) -> None:
    ps = get_set(a.set)
    print(f"{ps.name}: n={len(ps)} positives={np.mean(ps.y):.1%} fp={ps.fingerprint}")
    key, stats = extract(a.model, ps, layers=parse_layers(a.layers),
                         cache=ActivationCache(a.cache), dtype=a.dtype,
                         padding_side=a.padding_side, max_length=a.max_length,
                         batch_size=a.batch_size, device=a.device,
                         length_sorted=not a.no_sort)
    print(f"key {key.digest()}  layers={len(key.layers)}  poolings={key.poolings}")
    if stats:
        Path(a.cache).mkdir(parents=True, exist_ok=True)
        with open(Path(a.cache) / "extract_log.jsonl", "a") as f:
            f.write(json.dumps({"model": a.model, "set": ps.name,
                                "digest": key.digest(), "n": stats.n_prompts,
                                "tokens": stats.n_tokens, "wall_s": stats.wall_s,
                                "pad_fraction": stats.pad_fraction,
                                "peak_gb": stats.peak_mem_gb,
                                "length_sorted": stats.length_sorted,
                                "padding_side": a.padding_side,
                                "device": a.device or "auto"}) + "\n")


def cmd_sweep(a) -> None:
    ps = get_set(a.set)
    cache = ActivationCache(a.cache)
    key = _key_for(a, ps)
    if not cache.exists(key):
        sys.exit(f"no cache for {key.digest()} — run `extract` first")
    key = _rehydrate(cache, key)
    y = np.asarray(ps.y)
    t0 = time.time()
    rows = sweep(cache, key, y, poolings=a.poolings.split(",") if a.poolings else None,
                 k=a.folds, C=a.C, seed=a.seed, n_boot=a.boot)
    wall = time.time() - t0
    base = length_only(ps.text, y, n_boot=a.boot)
    print(f"\nfloor to beat — {base.row()}\n")
    for pooling in sorted({r.pooling for r in rows}):
        sub = [r for r in rows if r.pooling == pooling]
        best = max(sub, key=lambda r: r.auc)
        print(f"{pooling:<9} best L{best.layer:<3} AUC {best.auc:.3f} "
              f"[{best.lo:.3f}, {best.hi:.3f}]   "
              f"{'ABOVE' if best.lo > base.auc else 'does NOT clear'} the length floor")
    print(f"\n{len(rows)} (layer, pooling) probes in {wall:.1f}s of CPU, zero GPU")
    out = Path(a.out or "results") / f"sweep_{key.digest()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"key": {k: list(v) if isinstance(v, tuple) else v
                                       for k, v in key.__dict__.items()},
                               "set": ps.name, "wall_s": wall,
                               "length_floor": base.__dict__,
                               "rows": [r.__dict__ for r in rows]}, indent=2,
                              default=str))
    print(f"wrote {out}")


def cmd_controls(a) -> None:
    ps = get_set(a.set)
    cache = ActivationCache(a.cache)
    key = _key_for(a, ps)
    y = np.asarray(ps.y)
    print(f"== controls on {ps.name} (n={len(ps)}, {y.mean():.1%} positive) ==\n")
    for unit in ("chars", "words"):
        print("  " + length_only(ps.text, y, unit=unit, n_boot=a.boot).row())
    if not cache.exists(key):
        print("\n  (no cache yet — activation controls need `extract` first)")
        return
    key = _rehydrate(cache, key)
    X = cache.slice(key, a.layer, a.pooling)
    rs = [shuffled_labels(X, y, k=a.folds, seed=a.seed, n_boot=a.boot),
          random_features(len(y), X.shape[1], y, k=a.folds, seed=a.seed, n_boot=a.boot)]
    ok = True
    for r in rs:
        verdict = "PASS" if r.extra.get("passes") else "FAIL — interval excludes 0.5"
        ok &= bool(r.extra.get("passes"))
        print(f"  {r.row()}   {verdict}")
    s, _ = cross_val_auc(X, y, k=a.folds, seed=a.seed)
    auc, lo, hi = auc_ci(y, s, n_boot=a.boot, seed=a.seed)
    print(f"\n  probe L{a.layer} {a.pooling}: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\n  within length quartiles (length cannot explain these):")
    for r in length_matched(ps.text, y, s, n_bins=a.bins, seed=a.seed):
        c = r.extra["chars"]
        note = r.extra.get("note", "")
        val = "    n/a" if np.isnan(r.auc) else f"{r.auc:.3f} [{r.lo:.3f}, {r.hi:.3f}]"
        print(f"    {int(c[0]):>5}-{int(c[1]):<5} chars  n={r.n_test:<5} "
              f"pos={r.extra['pos_rate']:.0%}  AUC {val} {note}")
    sys.exit(0 if ok else 1)


def cmd_transfer(a) -> None:
    """Fit on plain prompts; score the jailbreak-framed ones and real user turns."""
    cache = ActivationCache(a.cache)
    train = get_set("wildguard-vanilla")

    def load(ps):
        key = _key_for(a, ps)
        if not cache.exists(key):
            sys.exit(f"no cache for {ps.name} ({key.digest()}) — extract it first")
        return cache.slice(_rehydrate(cache, key), a.layer, a.pooling), np.asarray(ps.y)

    Xtr, ytr = load(train)
    s, _ = cross_val_auc(Xtr, ytr, k=a.folds, C=a.C, seed=a.seed)
    auc, lo, hi = auc_ci(ytr, s, n_boot=a.boot, seed=a.seed)
    floor_tr = length_only(train.text, ytr, n_boot=a.boot)
    out = {"model": a.model, "layer": a.layer, "pooling": a.pooling,
           "padding_side": a.padding_side, "settings": [
               {"name": "wildguard-vanilla",
                "pretty": "plain prompts\n(out-of-fold, same distribution)",
                "n": len(ytr), "auc": auc, "lo": lo, "hi": hi,
                "length_floor": floor_tr.auc}]}
    print(f"iid        L{a.layer} {a.pooling} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"n={len(ytr)}  length floor {floor_tr.auc:.3f}")
    pretty = {"wildguard-adversarial": "jailbreak-framed prompts\n(same harms, new phrasing)",
              "toxicchat-toxicity": "real user turns\n(ToxicChat, human-labelled)",
              "toxicchat-jailbreaking": "real jailbreak attempts\n(ToxicChat)"}
    for name in a.to.split(","):
        ps = get_set(name)
        Xte, yte = load(ps)
        r = evaluate(Xtr, ytr, Xte, yte, name, a.layer, a.pooling, C=a.C,
                     seed=a.seed, n_boot=a.boot)
        floor = length_only(ps.text, yte, n_boot=a.boot)
        verdict = "clears" if r.lo > floor.auc else "DOES NOT CLEAR"
        rival = None
        if name.startswith("toxicchat"):
            rival = openai_moderation(ps, n_boot=a.boot).auc
        rtxt = f"   moderation API {rival:.3f}" if rival is not None else ""
        print(f"{r.row()}   length floor {floor.auc:.3f}   {verdict} it{rtxt}")
        out["settings"].append({"name": name, "pretty": pretty.get(name, name),
                                "n": len(yte), "auc": r.auc, "lo": r.lo, "hi": r.hi,
                                "length_floor": floor.auc, "moderation_api": rival})
    # The model belongs in the filename: two models whose best layer differs
    # would otherwise be told apart only by that coincidence, and two models
    # sharing a best layer would overwrite each other.
    tag = a.model.replace("/", "__")
    p = Path("results") / f"transfer_{tag}_L{a.layer}_{a.pooling}_{a.padding_side}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser("probeinfra", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, need_layer=False):
        q.add_argument("--model", default="Qwen/Qwen3-0.6B")
        q.add_argument("--cache", default="cache")
        q.add_argument("--dtype", default="float16")
        q.add_argument("--padding-side", default="left", choices=["left", "right"])
        q.add_argument("--max-length", type=int, default=1024)
        q.add_argument("--no-sort", action="store_true",
                       help="batch in dataset order; measures what length-sorting buys")
        q.add_argument("--layers", default="all")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--folds", type=int, default=5)
        q.add_argument("--C", type=float, default=1.0)
        q.add_argument("--boot", type=int, default=400)
        if need_layer:
            q.add_argument("--layer", type=int, required=True)
            q.add_argument("--pooling", default="last", choices=list(POOLINGS))

    q = sub.add_parser("extract", help="one GPU pass; fills the cache")
    common(q); q.add_argument("--set", default="wildguard-vanilla")
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--device", default=None)
    q.set_defaults(fn=cmd_extract)

    q = sub.add_parser("sweep", help="every layer x pooling, CPU only")
    common(q); q.add_argument("--set", default="wildguard-vanilla")
    q.add_argument("--poolings", default=None)
    q.add_argument("--out", default="results")
    q.set_defaults(fn=cmd_sweep)

    q = sub.add_parser("controls", help="run the control suite; exits non-zero on failure")
    common(q, need_layer=True); q.add_argument("--set", default="wildguard-vanilla")
    q.add_argument("--bins", type=int, default=4)
    q.set_defaults(fn=cmd_controls)

    q = sub.add_parser("transfer", help="fit on vanilla, score shift and OOD sets")
    common(q, need_layer=True)
    q.add_argument("--to", default="wildguard-adversarial,toxicchat-toxicity")
    q.set_defaults(fn=cmd_transfer)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
