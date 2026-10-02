#!/usr/bin/env python3
"""Generate every number the README quotes, from the JSON the runs wrote.

Typed numbers drift. This prints the tables, and the README pastes them, so a
re-run that changes a result changes the table rather than silently disagreeing
with the prose.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sweeps() -> list[dict]:
    out = []
    for p in sorted(Path("results").glob("sweep_*.json")):
        d = json.load(open(p))
        d["_path"] = str(p)
        out.append(d)
    return out


def arm_label(key: dict) -> str:
    sort = "length-sorted" if key.get("extras", {}).get("length_sorted", True) else "dataset order"
    return f"{key['padding_side']} padding, {sort}"


def best(rows: list[dict], pooling: str) -> dict:
    sub = [r for r in rows if r["pooling"] == pooling]
    return max(sub, key=lambda r: r["auc"]) if sub else {}


def main() -> None:
    sweeps = load_sweeps()
    if not sweeps:
        raise SystemExit("no results/sweep_*.json — run ./run_all.sh first")

    log = [json.loads(l) for l in open("cache/extract_log.jsonl")] \
        if Path("cache/extract_log.jsonl").exists() else []

    print("## Extraction (the only part that needs a GPU)\n")
    print("| prompt set | n | padding | batching | pad tokens | prompts/s | wall |")
    print("|---|---|---|---|---|---|---|")
    for e in log:
        print(f"| {e['set']} | {e['n']} | {e['padding_side']} | "
              f"{'length-sorted' if e['length_sorted'] else 'dataset order'} | "
              f"{e['pad_fraction']:.1%} | {e['n']/e['wall_s']:.0f} | {e['wall_s']:.1f}s |")
    if log:
        tot = sum(e["wall_s"] for e in log)
        print(f"\nTotal GPU wall: **{tot:.0f}s** for {len(log)} passes.")

    print("\n## What the pooling bug costs, by configuration\n")
    print("| configuration | pad tokens | `last` (correct) | `last_raw` (the bug) | cost | "
          "best layer moves |")
    print("|---|---|---|---|---|---|")
    for d in sweeps:
        k = d["key"]
        pad = next((e["pad_fraction"] for e in log
                    if e["digest"] == "".join(c for c in Path(d["_path"]).stem.split("_")[-1])),
                   None)
        good, bad = best(d["rows"], "last"), best(d["rows"], "last_raw")
        if not good or not bad:
            continue
        delta = bad["auc"] - good["auc"]
        moved = "—" if good["layer"] == bad["layer"] else f"L{good['layer']} -> L{bad['layer']}"
        padtxt = f"{pad:.1%}" if pad is not None else "?"
        print(f"| {arm_label(k)} | {padtxt} | {good['auc']:.3f} | {bad['auc']:.3f} | "
              f"**{delta:+.3f}** | {moved} |")

    print("\n## Layer sweep, primary arm\n")
    prim = next((d for d in sweeps if d["key"]["padding_side"] == "left"
                 and d["key"].get("extras", {}).get("length_sorted", True)), sweeps[0])
    floor = prim["length_floor"]["auc"]
    print(f"Length-only floor: **{floor:.3f}** "
          f"[{prim['length_floor']['lo']:.3f}, {prim['length_floor']['hi']:.3f}]  ·  "
          f"{len(prim['rows'])} probes in **{prim['wall_s']:.0f}s of CPU**, zero GPU\n")
    print("| pooling | best layer | AUC [95% CI] | vs length floor |")
    print("|---|---|---|---|")
    for p in ("last", "mean", "last_raw", "mean_raw"):
        b = best(prim["rows"], p)
        if b:
            print(f"| `{p}` | L{b['layer']} | {b['auc']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] | "
                  f"{b['auc']-floor:+.3f} |")
    emb = [r for r in prim["rows"] if r["layer"] == 0]
    if emb:
        print(f"\nLayer 0 is the embedding output — no transformer block has run. "
              f"`last` there scores {best(emb,'last')['auc']:.3f} and `mean` "
              f"{best(emb,'mean')['auc']:.3f}, against the {floor:.3f} length floor.")

    for p in sorted(Path("results").glob("transfer_*.json")):
        t = json.load(open(p))
        print(f"\n## Generalisation — one probe, fit on plain prompts "
              f"(L{t['layer']} `{t['pooling']}`)\n")
        print("| scored on | n | AUC [95% CI] | length floor | clears it? |")
        print("|---|---|---|---|---|")
        for s in t["settings"]:
            clears = "yes" if s["lo"] > s["length_floor"] else "**no**"
            print(f"| {s['pretty'].replace(chr(10),' ')} | {s['n']} | "
                  f"{s['auc']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] | "
                  f"{s['length_floor']:.3f} | {clears} |")


if __name__ == "__main__":
    main()
