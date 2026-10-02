#!/usr/bin/env python3
"""Generate every number the README quotes, from the JSON the runs wrote.

Typed numbers drift. This prints the tables and the README pastes them, so a
re-run that changes a result changes the table rather than silently disagreeing
with the prose.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def arm(key: dict) -> str:
    sort = "length-sorted" if key.get("extras", {}).get("length_sorted", True) else "dataset order"
    return f"{key['padding_side']} padding, {sort}"


ARM_ORDER = ["left padding, length-sorted", "right padding, length-sorted",
             "right padding, dataset order"]


def best(rows: list[dict], pooling: str) -> dict:
    sub = [r for r in rows if r["pooling"] == pooling]
    return max(sub, key=lambda r: r["auc"]) if sub else {}


def short(model: str) -> str:
    return model.split("/")[-1]


def main() -> None:
    sweeps = defaultdict(dict)
    for p in sorted(Path("results").glob("sweep_*.json")):
        d = json.load(open(p))
        sweeps[d["key"]["model_id"]][arm(d["key"])] = d
    if not sweeps:
        raise SystemExit("no results/sweep_*.json — run ./run_all.sh first")
    models = sorted(sweeps)   # "Qwen3-0.6B" sorts before "Qwen3-8B"
    log = ([json.loads(l) for l in open("cache/extract_log.jsonl")]
           if Path("cache/extract_log.jsonl").exists() else [])

    print("## Extraction — the only step that needs a GPU\n")
    print("| model | prompt set | n | padding | batching | pad tokens | prompts/s | wall |")
    print("|---|---|---|---|---|---|---|---|")
    for e in log:
        print(f"| {short(e['model'])} | {e['set']} | {e['n']} | {e['padding_side']} | "
              f"{'length-sorted' if e['length_sorted'] else 'dataset order'} | "
              f"{e['pad_fraction']:.1%} | {e['n']/e['wall_s']:.1f} | {e['wall_s']:.1f}s |")
    for m in models:
        sub = [e for e in log if e["model"] == m]
        if sub:
            tox = [e["wall_s"] for e in sub if "toxicchat" in e["set"]]
            saved = tox[0] if tox else 0.0
            tot = sum(e["wall_s"] for e in sub)
            print(f"\n{short(m)}: **{tot:.0f}s** of forward passes over {len(sub)} passes. "
                  f"Without the prompt-keyed cache it would have been {tot+saved:.0f}s "
                  f"({saved/(tot+saved):.0%} wasted on a second pass over identical prompts).")

    print("\n## What the `h[:, -1]` pooling bug costs\n")
    # Padding fraction is per model, not per configuration: batch size differs
    # (8 vs 4), so the same "dataset order" arm pads 29.0% on one and 22.3% on
    # the other. Collapsing them into one column would hide the variable the
    # table is about.
    header = " | ".join(f"{short(m)} cost (pad)" for m in models)
    print(f"| configuration | {header} |")
    print("|---|" + "---|" * len(models))
    for a in ARM_ORDER:
        cells = []
        for m in models:
            d = sweeps[m].get(a)
            if not d:
                cells.append("—")
                continue
            g, b = best(d["rows"], "last"), best(d["rows"], "last_raw")
            moved = "" if g["layer"] == b["layer"] else f", L{g['layer']}→L{b['layer']}"
            e = [x for x in log if x["model"] == m
                 and x["padding_side"] == a.split()[0]
                 and x["length_sorted"] == ("sorted" in a)
                 and "vanilla" in x["set"]]
            pad = f" ({e[0]['pad_fraction']:.1%})" if e else ""
            cells.append(f"**{b['auc']-g['auc']:+.3f}**{moved}{pad}")
        print(f"| {a} | " + " | ".join(cells) + " |")
    print("\n(cost = `h[:, -1]` minus `last`, each at its own best layer; "
          "pad fraction in brackets)")

    print("\n## Layer sweep, primary arm (left padding, length-sorted)\n")
    for m in models:
        d = sweeps[m][ARM_ORDER[0]]
        f = d["length_floor"]["auc"]
        print(f"\n**{short(m)}** — length floor {f:.3f}, "
              f"{len(d['rows'])} probes in {d['wall_s']:.0f}s of CPU\n")
        print("| pooling | best layer | AUC [95% CI] | vs floor |")
        print("|---|---|---|---|")
        for p in ("last", "mean", "last_raw", "mean_raw"):
            b = best(d["rows"], p)
            if b:
                print(f"| `{p}` | L{b['layer']} | {b['auc']:.3f} "
                      f"[{b['lo']:.3f}, {b['hi']:.3f}] | {b['auc']-f:+.3f} |")
        e0 = [r for r in d["rows"] if r["layer"] == 0]
        print(f"\nLayer 0 (embedding output, no block has run): `mean` "
              f"{best(e0,'mean')['auc']:.3f} against the {f:.3f} floor.")

    trs = {}
    for p in sorted(Path("results").glob("transfer_*.json")):
        t = json.load(open(p))
        trs[t["model"]] = t
    if trs:
        print("\n## Generalisation — one probe, fit once on plain prompts\n")
        cols = " | ".join(f"{short(m)} L{trs[m]['layer']}" for m in models if m in trs)
        print(f"| scored on | n | {cols} | `len(prompt)` | moderation API |")
        print("|---|---|" + "---|" * (len(trs) + 2))
        ref = trs[models[0]]
        for i, s in enumerate(ref["settings"]):
            cells = [f"{trs[m]['settings'][i]['auc']:.3f}" for m in models if m in trs]
            api = s.get("moderation_api")
            print(f"| {s['pretty'].replace(chr(10), ' ')} | {s['n']} | "
                  + " | ".join(cells)
                  + f" | **{s['length_floor']:.3f}** | "
                  + (f"{api:.3f}" if api else "—") + " |")


if __name__ == "__main__":
    main()
