#!/usr/bin/env python3
"""Two panels, in the same house style as the other repos' charts.

Left panel is the pooling bug's dose-response. The correct pooling is one navy
line; the same wrong pooling is drawn three times, once per configuration, and
the three curves are the whole point — identical code, identical weights, and a
cost that runs from nothing to eight points depending on two settings that live
nowhere near the pooling.

Right panel is generalisation, and it is drawn so the floor cannot be skipped:
each bar carries the length-only baseline for its own set as an orange tick. On
the last row that tick sits to the RIGHT of the bar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, STEEL, ORANGE, GREY = "#17365d", "#5b7fae", "#c2703a", "#98a2b3"
TEAL = "#3f7d78"
INK, MUTED = "#1c1a19", "#5d666e"


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9ced6")
    ax.set_axisbelow(True)
    ax.grid(color="#e8eaee", lw=0.9)
    ax.tick_params(colors=MUTED, labelsize=11)


def arm_label(key: dict) -> str:
    sort = "length-sorted" if key.get("extras", {}).get("length_sorted", True) else "dataset order"
    return f"{key['padding_side']} padding, {sort}"


def curve(rows: list[dict], pooling: str) -> tuple[list[int], list[float]]:
    r = sorted([x for x in rows if x["pooling"] == pooling], key=lambda x: x["layer"])
    return [x["layer"] for x in r], [x["auc"] for x in r]


def main(out: str = "results/probe-floor.png") -> None:
    sweeps = []
    for p in sorted(Path("results").glob("sweep_*.json")):
        d = json.load(open(p))
        d["_arm"] = arm_label(d["key"])
        sweeps.append(d)
    prim = next(d for d in sweeps if d["_arm"] == "left padding, length-sorted")
    tr = json.load(open(next(iter(sorted(Path("results").glob("transfer_*.json"))))))
    floor = prim["length_floor"]["auc"]
    model = prim["key"]["model_id"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.6))
    fig.suptitle("The same pooling bug costs 0.000, 0.014 or 0.083 AUC — "
                 "and counting characters scores 0.737",
                 fontsize=14.5, color=INK, y=0.985)

    # ---- left: one correct curve, the wrong one three times ----------------
    x, y = curve(prim["rows"], "last")
    ax1.plot(x, y, "-", color=NAVY, lw=2.4, zorder=5,
             label="`last` — last real token (correct, all 3 configs)")

    for d, colour in [(next(s for s in sweeps if s["_arm"] == "right padding, length-sorted"), STEEL),
                      (next(s for s in sweeps if s["_arm"] == "right padding, dataset order"), ORANGE)]:
        xr, yr = curve(d["rows"], "last_raw")
        pad = "2.3% padding" if "sorted" in d["_arm"] else "29.0% padding"
        ax1.plot(xr, yr, "--", color=colour, lw=2.0, zorder=4,
                 label=f"`h[:, -1]` — {d['_arm'].replace('padding, ', 'pad, ')} ({pad})")

    ax1.axhline(floor, color=ORANGE, ls=":", lw=1.8, zorder=2)
    ax1.text(5.0, floor + 0.008, f"length-only floor  {floor:.3f}", fontsize=10.2,
             color=ORANGE, va="bottom", ha="left")
    ax1.set_xlabel("residual-stream layer   (0 = embedding output)", fontsize=11.5, color=MUTED)
    ax1.set_ylabel("AUC, out-of-fold on plain prompts", fontsize=11.5, color=MUTED)
    ax1.set_ylim(0.58, 1.0)
    ax1.set_xlim(-0.8, 28.8)
    ax1.legend(fontsize=9.6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax1.set_title("What `h[:, -1]` costs, by configuration", fontsize=12.5, color=INK, pad=10)
    style(ax1)
    # The fourth curve — `h[:, -1]` under LEFT padding — is not drawn: it is
    # bit-identical to the navy line and would only hide it. The legend says so
    # ("all 3 configs"), and tests/test_pooling.py asserts it.

    # ---- right: generalisation, labels in a reserved gutter -----------------
    S = tr["settings"]
    yy = np.arange(len(S))[::-1]
    colours = [NAVY, STEEL, TEAL, GREY][:len(S)]
    aucs = [s["auc"] for s in S]
    ax2.barh(yy, aucs, 0.54, color=colours, zorder=3)
    ax2.errorbar(aucs, yy, xerr=[[s["auc"] - s["lo"] for s in S],
                                 [s["hi"] - s["auc"] for s in S]],
                 fmt="none", ecolor="#2b3540", capsize=4, lw=1.3, zorder=6)

    LBL_A, LBL_B = 1.085, 1.175          # reserved gutter, no bar reaches it
    for i, (s, row) in enumerate(zip(S, yy)):
        f = s["length_floor"]
        ax2.plot([f, f], [row - 0.32, row + 0.32], color=ORANGE, lw=3.0, zorder=7)
        gap = s["auc"] - f
        ax2.text(LBL_A, row, f"{s['auc']:.3f}", va="center", ha="right",
                 fontsize=12, fontweight="bold", color=INK)
        ax2.text(LBL_B, row, f"{gap:+.3f}", va="center", ha="left", fontsize=11,
                 fontweight="bold" if gap < 0 else "normal",
                 color=ORANGE if gap < 0 else MUTED)
    ax2.text(LBL_A, len(S) - 0.52, "AUC", ha="right", fontsize=10, color=MUTED)
    ax2.text(LBL_B, len(S) - 0.52, "vs floor", ha="left", fontsize=10, color=MUTED)

    ax2.set_yticks(yy)
    ax2.set_yticklabels([s["pretty"] for s in S], fontsize=10.2)
    ax2.set_xlim(0.40, 1.30)
    ax2.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_ylim(-1.15, len(S) - 0.3)
    ax2.set_xlabel(f"AUC  ·  {model.split('/')[-1]}  ·  layer {tr['layer']}, "
                   f"{tr['pooling']} pooling", fontsize=10.5, color=MUTED)
    ax2.set_title("Orange tick = counting characters on that same set",
                  fontsize=12.5, color=INK, pad=10)
    style(ax2)
    ax2.grid(axis="y", visible=False)
    last = S[-1]
    ax2.annotate("the floor is to the RIGHT of the bar:\n"
                 "length alone beats the probe by 18.5 points",
                 xy=(last["length_floor"], -0.36), xytext=(0.73, -0.95),
                 fontsize=9.8, color=ORANGE, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2,
                                 shrinkA=2, shrinkB=5,
                                 connectionstyle="arc3,rad=-0.25"))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/probe-floor.png")
