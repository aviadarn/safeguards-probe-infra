#!/usr/bin/env python3
"""Two panels, in the same house style as the other repos' charts.

Left is the pooling bug's cost, and it is a bar chart rather than a curve because
the finding is a comparison across configurations, not a shape across layers. The
third variable is model size: the bug that costs 8 points on a 0.6B costs 0.3 on
an 8B, so a team that only ever tested on its production model would conclude
`h[:, -1]` was fine.

Right is generalisation with both rivals drawn on every row: the orange tick is
`len(prompt)` and the grey diamond is the OpenAI moderation API's own scores,
which ToxicChat ships. On the bottom row both rivals sit to the right of both
probes.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, STEEL, ORANGE, GREY = "#17365d", "#5b7fae", "#c2703a", "#98a2b3"
INK, MUTED = "#1c1a19", "#5d666e"

ARMS = [("left padding\nlength-sorted", "left padding, length-sorted"),
        ("right padding\nlength-sorted", "right padding, length-sorted"),
        ("right padding\ndataset order", "right padding, dataset order")]


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9ced6")
    ax.set_axisbelow(True)
    ax.grid(color="#e8eaee", lw=0.9)
    ax.tick_params(colors=MUTED, labelsize=11)


def arm_of(key: dict) -> str:
    sort = "length-sorted" if key.get("extras", {}).get("length_sorted", True) else "dataset order"
    return f"{key['padding_side']} padding, {sort}"


def best(rows, pooling):
    sub = [r for r in rows if r["pooling"] == pooling]
    return max(sub, key=lambda r: r["auc"])


def main(out: str = "results/probe-floor.png") -> None:
    sweeps = defaultdict(dict)
    for p in sorted(Path("results").glob("sweep_*.json")):
        d = json.load(open(p))
        sweeps[d["key"]["model_id"]][arm_of(d["key"])] = d
    trs = {}
    for p in sorted(Path("results").glob("transfer_*.json")):
        t = json.load(open(p))
        trs[t["model"]] = t
    models = sorted(sweeps)
    names = [m.split("/")[-1] for m in models]
    cols = [STEEL, NAVY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.6, 5.8),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    fig.suptitle("Counting characters beats every detector here on real jailbreak "
                 "attempts — and the pooling bug hides at scale",
                 fontsize=14.2, color=INK, y=0.985)

    # ---- left: what the bug costs, per configuration, per model -------------
    x = np.arange(len(ARMS))
    w = 0.35
    for i, (m, nm) in enumerate(zip(models, names)):
        vals = []
        for _, a in ARMS:
            d = sweeps[m].get(a)
            vals.append(best(d["rows"], "last_raw")["auc"] - best(d["rows"], "last")["auc"]
                        if d else 0.0)
        pos = x + (i - 0.5) * w
        ax1.bar(pos, vals, w, color=cols[i], label=nm, zorder=3)
        for xx, v in zip(pos, vals):
            # A zero-height bar puts both models' labels on the same baseline,
            # where "+0.000+0.000" reads as one number. Stagger by model.
            off = 0.0018 + (0.0052 if (abs(v) < 5e-4 and i == 0) else 0.0)
            ax1.text(xx, v - 0.0035 if v < 0 else v + off, f"{v:+.3f}",
                     ha="center", va="top" if v < 0 else "bottom",
                     fontsize=10.2, fontweight="bold", color=INK)
    ax1.axhline(0, color="#6b7480", lw=1.1, zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for lbl, _ in ARMS], fontsize=10.2)
    ax1.set_ylabel("AUC cost of using `h[:, -1]`", fontsize=11.5, color=MUTED)
    ax1.set_ylim(-0.098, 0.022)
    ax1.legend(fontsize=10.5, frameon=False, loc="lower left")
    ax1.set_title("Same bug, same code, three configurations", fontsize=12.5,
                  color=INK, pad=10)
    style(ax1)
    ax1.grid(axis="x", visible=False)
    ax1.annotate("best layer also moves\nL16 → L28",
                 xy=(1.83, -0.083), xytext=(1.30, -0.052), fontsize=9.6, color=ORANGE,
                 ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1,
                                 shrinkA=2, shrinkB=4,
                                 connectionstyle="arc3,rad=-0.25"))

    # ---- right: generalisation, both models, both rivals --------------------
    ref = trs[models[0]]
    S = ref["settings"]
    rows = np.arange(len(S))[::-1]
    bw = 0.30
    for i, (m, nm) in enumerate(zip(models, names)):
        t = trs[m]
        vals = [s["auc"] for s in t["settings"]]
        pos = rows + (0.5 - i) * bw
        ax2.barh(pos, vals, bw, color=cols[i], zorder=3,
                 label=f"{nm} L{t['layer']}")
        for yy, s in zip(pos, t["settings"]):
            ax2.errorbar(s["auc"], yy, xerr=[[s["auc"] - s["lo"]], [s["hi"] - s["auc"]]],
                         fmt="none", ecolor="#2b3540", capsize=3, lw=1.1, zorder=6)
            # Inside the bar: outside, every label collided with its own
            # error bar or with a rival marker.
            ax2.text(s["auc"] - 0.006, yy, f"{s['auc']:.3f}", va="center",
                     ha="right", fontsize=10, color="white", fontweight="bold",
                     zorder=7)

    for yy, s in zip(rows, S):
        f = s["length_floor"]
        ax2.plot([f, f], [yy - 0.44, yy + 0.44], color=ORANGE, lw=3.0, zorder=8)
        api = s.get("moderation_api")
        if api:
            ax2.scatter([api], [yy], marker="D", s=58, color="#4a5560",
                        edgecolor="white", lw=1.2, zorder=9)

    ax2.plot([], [], color=ORANGE, lw=3.0, label="`len(prompt)`")
    ax2.scatter([], [], marker="D", s=58, color="#4a5560", edgecolor="white",
                lw=1.2, label="OpenAI moderation API")
    ax2.set_yticks(rows)
    ax2.set_yticklabels([s["pretty"] for s in S], fontsize=10.2)
    ax2.set_xlim(0.60, 1.03)
    ax2.set_ylim(-0.95, len(S) - 0.42)
    ax2.set_xlabel("AUC — probes fit once on plain prompts, then frozen",
                   fontsize=11, color=MUTED)
    ax2.legend(fontsize=9.8, frameon=False, ncol=4, loc="lower left",
               bbox_to_anchor=(-0.02, 1.012), handlelength=1.5,
               columnspacing=1.5, handletextpad=0.5)
    ax2.set_title("One probe per model, scored on four distributions",
                  fontsize=12.5, color=INK, pad=34)
    style(ax2)
    ax2.grid(axis="y", visible=False)
    ax2.annotate("on this row both rivals beat both probes",
                 xy=(S[-1]["length_floor"], -0.46), xytext=(0.80, -0.80),
                 fontsize=10, color=ORANGE, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2,
                                 shrinkA=2, shrinkB=5,
                                 connectionstyle="arc3,rad=-0.3"))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/probe-floor.png")
