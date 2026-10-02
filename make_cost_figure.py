#!/usr/bin/env python3
"""Quality against marginal cost, on two slices, because the ranking flips.

Left is the distribution the probe was fit on. Right is real user traffic. The
probe wins the left panel and loses the right one, which is the whole engineering
argument: a probe is 138x cheaper at the margin and better where it was trained,
and a general model prompted as a classifier generalises better to traffic nobody
fit anything on.

"Marginal" is load-bearing in the x-axis label. A probe's cost is the capture
overhead plus a dot product, because the forward pass it reads from is one the
server already ran to answer the user. A prompted guard has no such luxury: it is
a second pass over a longer sequence. Charging the probe for a full forward would
be the wrong comparison; saying so explicitly is part of the panel.

Log x, because the costs span five orders of magnitude.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, STEEL, ORANGE, TEAL = "#17365d", "#5b7fae", "#c2703a", "#3f7d78"
INK, MUTED = "#1c1a19", "#5d666e"

PANELS = [("wildguard-vanilla", "Plain prompts — the distribution the probe was fit on"),
          ("toxicchat-jailbreaking", "Real jailbreak attempts — traffic nobody fit anything on")]


def style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9ced6")
    ax.grid(color="#e8eaee", lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10.5)


def main(out: str = "results/cost-quality.png") -> None:
    c = json.load(open("results/cost.json"))
    tr = json.load(open("results/transfer_Qwen__Qwen3-8B_L18_mean_left.json"))
    probe = {s["name"]: s for s in tr["settings"]}
    probe_ms = (c["forward_with_capture_ms"] - c["forward_no_capture_ms"]
                + c["probe_score_us"] / 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.0), sharey=True)
    fig.suptitle(f"The probe is {c['marginal_ratio']:.0f}x cheaper at the margin — "
                 "and the ranking flips between these two panels",
                 fontsize=14.5, color=INK, y=0.975)

    for ax, (slice_name, title) in zip(axes, PANELS):
        p, g = probe[slice_name], c["guard"][slice_name]
        # Short labels, alternating above and below. Earlier versions carried
        # the explanation inline ("a second forward pass") and no placement rule
        # survived it -- on the right-hand panel the guard, the API and the probe
        # all sit within 0.04 AUC. The explanations moved to the caption.
        pts = [("len(prompt)", 0.0018, p["length_floor"], ORANGE, (0, 15), "bottom"),
               ("probe", probe_ms, p["auc"], NAVY, (0, -16), "top"),
               ("prompted guard", g["per_prompt_ms"], g["auc"], STEEL, (0, 15), "bottom")]
        if p.get("moderation_api"):
            pts.insert(2, ("moderation API", 40.0, p["moderation_api"], TEAL,
                           (0, -16), "top"))
        for name, x, y, col, off, va in pts:
            ax.scatter([x], [y], s=190, color=col, edgecolor="white", lw=1.8, zorder=5)
            ax.annotate(f"{name}\n{y:.3f}", xy=(x, y), xytext=off,
                        textcoords="offset points", ha="center", va=va,
                        fontsize=10, color=col, fontweight="bold")
        best = max(v[2] for v in pts)
        ax.axhline(best, color="#c9ced6", ls=":", lw=1.4, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(2e-4, 2e4)
        ax.set_title(title, fontsize=11.8, color=INK, pad=10)
        ax.set_xlabel("marginal cost per prompt, ms (log)", fontsize=11, color=MUTED)
        style(ax)

    axes[0].set_ylim(0.62, 1.06)
    axes[0].set_ylabel("AUC", fontsize=11.5, color=MUTED)
    fig.text(0.5, 0.055,
             "probe = capturing the residual stream of a forward pass the server already ran, "
             "plus a dot product   ·   prompted guard = a second forward pass over a longer "
             "sequence   ·   moderation API = a network round-trip",
             ha="center", fontsize=9.4, color=MUTED)
    fig.text(0.5, 0.013,
             "The API's 40 ms is a stand-in for a round-trip, not a measurement — its scores "
             "ship with ToxicChat and were not collected here.   Timings are MPS on a MacBook: "
             "the ratio transfers, the milliseconds do not.",
             ha="center", fontsize=8.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.085, 1, 0.935))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/cost-quality.png")
