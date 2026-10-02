"""Dataset loaders for misuse-detection probes.

Three splits with deliberately different jobs:

  vanilla      WildGuard prompts written plainly. The easy, in-distribution case.
  adversarial  The same harm categories wrapped in jailbreak framing. Prompts are
               ~6x longer, which is why `controls.length_only` exists.
  toxicchat    Real user turns from an LMSYS deployment, human-annotated. Out of
               distribution in every way that matters: language, length, topic.

Every loader asserts on null labels. WildGuardTest ships 26 rows with a null
label; casting `label == "harmful"` silently files them as negatives.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download

WILDGUARD = ("walledai/WildGuardTest", "data/train-00000-of-00001.parquet")
TOXICCHAT = ("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_{split}.csv")


@dataclass(frozen=True)
class PromptSet:
    """Prompts in a fixed, content-addressed order.

    `pid` is a hash of the prompt text, so the activation cache is keyed on
    content rather than row position. Re-downloading the dataset in a different
    order cannot produce a stale cache hit.
    """

    name: str
    pid: list[str]
    text: list[str]
    y: list[int]
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.pid)

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in self.pid:
            h.update(p.encode())
        return h.hexdigest()[:16]


def _pid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build(name: str, df: pd.DataFrame, text_col: str, y_col: str) -> PromptSet:
    if df[y_col].isna().any() or df[text_col].isna().any():
        raise ValueError(
            f"{name}: {df[y_col].isna().sum()} null labels and "
            f"{df[text_col].isna().sum()} null prompts reached _build; "
            "drop them explicitly rather than letting a cast decide"
        )
    # Deduplicate on prompt text: the cache is keyed on content, so two rows with
    # identical text would otherwise write the same key twice and let a label
    # disagreement pass unnoticed.
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    text = df[text_col].astype(str).tolist()
    return PromptSet(
        name=name,
        pid=[_pid(t) for t in text],
        text=text,
        y=df[y_col].astype(int).tolist(),
        meta=df,
    )


def wildguard(adversarial: bool | None = None) -> PromptSet:
    """WildGuardTest prompt-harm labels.

    adversarial=False -> plain phrasing; True -> jailbreak framing; None -> both.
    """
    df = pd.read_parquet(hf_hub_download(*WILDGUARD, repo_type="dataset"))
    n_raw = len(df)
    df = df[df.label.notna() & df.prompt.notna()].copy()
    dropped = n_raw - len(df)
    df["y"] = (df.label == "harmful").astype(int)
    tag = "both"
    if adversarial is not None:
        df = df[df.adversarial == adversarial].copy()
        tag = "adversarial" if adversarial else "vanilla"
    ps = _build(f"wildguard-{tag}", df.reset_index(drop=True), "prompt", "y")
    object.__setattr__(ps, "meta", ps.meta.assign(_dropped_null_labels=dropped))
    return ps


def toxicchat(split: str = "test", label: str = "toxicity",
              human_only: bool = True) -> PromptSet:
    """Real user turns, human-annotated. `label` is "toxicity" or "jailbreaking".

    human_only keeps the 2,853 rows a person actually read. The rest carry
    propagated labels and do not belong in an evaluation set.
    """
    path = TOXICCHAT[1].format(split=split)
    df = pd.read_csv(hf_hub_download(TOXICCHAT[0], path, repo_type="dataset"))
    if human_only:
        df = df[df.human_annotation].copy()
    df = df[df[label].notna() & df.user_input.notna()].copy()
    df["y"] = df[label].astype(int)
    return _build(f"toxicchat-{split}-{label}", df.reset_index(drop=True),
                  "user_input", "y")
