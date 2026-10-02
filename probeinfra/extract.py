"""One GPU pass over the prompts; every layer and every pooling, cached.

Design notes that are not obvious:

*Length-sorted batching.* Prompts here run from 30 to 3,000 characters, and the
adversarial split averages 6x the vanilla one. Batching in dataset order pads
every short prompt out to the longest in its batch, and padding is most of the
compute. Sorting by token count first, then restoring the original order on the
way out, is the single biggest throughput lever in this file. The restore is
asserted, because a silent permutation here would misalign activations against
labels and still produce a plausible AUC.

*Why `output_hidden_states` rather than forward hooks.* Hooks are the usual
answer and they are also where off-by-one layer indexing comes from: a hook on
`model.layers[k]` fires on that block's output, which is the residual stream
*entering* block k+1. `hidden_states` is unambiguous — index 0 is the embedding
output and index k is the output of block k — so the layer number in a published
figure means one thing only.

*The two wrong poolings are implemented on purpose.* `last_raw` and `mean_raw`
are what you get from `h[:, -1]` and `h.mean(1)`, which is what most code does.
They are cached beside the correct ones so the cost of the mistake is a number
in a table rather than a claim in a README.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .cache import ActivationCache, CacheKey, resolve_revision, sha_of
from .data import PromptSet

POOLINGS = ("last", "mean", "last_raw", "mean_raw")


def pool(h: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """h [B, L, D] float, mask [B, L] of 1 for real tokens. Returns [B, D]."""
    if how == "last_raw":
        # The bug. Under right padding this is a pad position; under left padding
        # it happens to be correct, which is why padding_side is in the cache key.
        return h[:, -1, :]
    if how == "mean_raw":
        return h.mean(dim=1)
    m = mask.to(h.dtype).unsqueeze(-1)
    if how == "mean":
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    if how == "last":
        idx = mask.long().cumsum(dim=1).argmax(dim=1)  # last index where mask==1
        return h[torch.arange(h.shape[0], device=h.device), idx, :]
    raise KeyError(f"unknown pooling {how!r}")


@dataclass
class ExtractStats:
    n_prompts: int
    n_tokens: int
    pad_tokens: int
    wall_s: float
    peak_mem_gb: float
    length_sorted: bool = True

    @property
    def prompts_per_s(self) -> float:
        return self.n_prompts / self.wall_s

    @property
    def pad_fraction(self) -> float:
        return self.pad_tokens / max(1, self.n_tokens + self.pad_tokens)

    def report(self) -> str:
        sort = "length-sorted" if self.length_sorted else "dataset order"
        return (f"{self.n_prompts} prompts in {self.wall_s:.1f}s "
                f"({self.prompts_per_s:.1f}/s, {self.n_tokens} real tokens, "
                f"{self.pad_fraction:.1%} padding, {sort}, "
                f"peak {self.peak_mem_gb:.1f} GB)")


def resolve_key(model_id: str, ps: PromptSet, layers: tuple[int, ...] | None = None,
                dtype: str = "float16", padding_side: str = "left",
                max_length: int = 1024, chat_template: str | None = None,
                poolings: tuple[str, ...] = POOLINGS,
                length_sorted: bool = True) -> CacheKey:
    """The cache key, computable WITHOUT loading the weights.

    This matters more than it looks. If the key can only be built by the code
    that does the forward pass, then nothing can look up the cache without a
    GPU — and the entire value of the cache is that the next ninety questions
    are answered without one. So the key is derived from the tokenizer and the
    config (both a few KB over the network, both cached locally by `hf`), and
    `layers=None` is resolved against `config.num_hidden_layers` here rather
    than inside the extraction loop.

    An earlier version built the key in two places with two different tokenizer
    hashes. `extract` wrote one digest and `sweep` looked for another, so every
    sweep reported a cache miss. A cache whose key is not reproducible is not a
    cache.
    """
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = cfg.num_hidden_layers
    if layers is None:
        layers = tuple(range(n_layers + 1))  # 0 = embeddings, k = output of block k
    if max(layers) > n_layers:
        raise ValueError(f"layer {max(layers)} > {n_layers} blocks in {model_id}")

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok_sha = sha_of(f"{tok.__class__.__name__}|{tok.vocab_size}|{tok.pad_token_id}|"
                     f"{len(tok)}")

    return CacheKey(
        model_id=model_id,
        model_sha=resolve_revision(model_id),
        tokenizer_sha=tok_sha,
        dtype=dtype, layers=tuple(layers), poolings=tuple(poolings),
        padding_side=padding_side, max_length=max_length,
        chat_template_sha=sha_of(chat_template),
        prompt_set=ps.name, prompt_fingerprint=ps.fingerprint,
        extras={"hidden_size": int(cfg.hidden_size), "n_blocks": int(n_layers),
                # Batching changes how much padding each row sees, and the two
                # `*_raw` poolings read the padding. Same prompts, same weights,
                # different numbers — so it belongs in the key.
                "length_sorted": bool(length_sorted)},
    )


def extract(model_id: str, ps: PromptSet, layers: tuple[int, ...] | None = None,
            cache: ActivationCache | None = None, device: str | None = None,
            dtype: str = "float16", padding_side: str = "left",
            max_length: int = 1024, batch_size: int = 8,
            poolings: tuple[str, ...] = POOLINGS,
            chat_template: str | None = None,
            length_sorted: bool = True,
            progress_every: int = 20) -> tuple[CacheKey, ExtractStats | None]:
    """Fill the cache for (model, prompts, layers, poolings). Idempotent."""
    from transformers import AutoModel, AutoTokenizer

    cache = cache or ActivationCache()
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype)

    key = resolve_key(model_id, ps, layers, dtype, padding_side, max_length,
                      chat_template, poolings, length_sorted)
    layers = key.layers
    if cache.exists(key):
        cache.verify(key)
        print(f"cache hit {cache.dir_for(key)}")
        return key, None

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_id, dtype=torch_dtype).to(device).eval()

    texts = list(ps.text)
    if chat_template is not None:
        texts = [tok.apply_chat_template([{"role": "user", "content": t}],
                                         tokenize=False, add_generation_prompt=True)
                 for t in texts]

    # Length-sorted batching. `order` maps sorted position -> original row.
    # `length_sorted=False` is kept because it is the only way to measure what
    # the sort buys, and because it is also the configuration under which the
    # `*_raw` pooling bug is large enough to notice. The optimisation and the
    # bug are controlled by the same knob, which is why neither is safe to
    # judge from a single number.
    lens = [len(tok(t, truncation=True, max_length=max_length).input_ids) for t in texts]
    order = (np.argsort(lens, kind="stable") if length_sorted
             else np.arange(len(texts)))
    d = model.config.hidden_size
    out = np.zeros((len(texts), len(layers), len(poolings), d), dtype=np.float16)

    n_tok = n_pad = 0
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for bi, start in enumerate(range(0, len(order), batch_size)):
            rows = order[start:start + batch_size]
            enc = tok([texts[r] for r in rows], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"]
            n_tok += int(mask.sum())
            n_pad += int((1 - mask).sum())
            for li, layer in enumerate(layers):
                h = hs[layer].float()
                for pi, how in enumerate(poolings):
                    out[rows, li, pi, :] = pool(h, mask, how).cpu().numpy().astype(np.float16)
            del hs
            if progress_every and bi % progress_every == 0:
                done = start + len(rows)
                print(f"  {done}/{len(order)} ({done/len(order):.0%}) "
                      f"{time.time()-t0:.0f}s", flush=True)

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    stats = ExtractStats(len(texts), n_tok, n_pad, time.time() - t0, peak,
                         length_sorted=length_sorted)

    # The restore is asserted: a permutation bug here misaligns every label.
    assert sorted(order.tolist()) == list(range(len(texts))), "batching lost a row"
    assert not np.isnan(out).any(), "NaN in activations"
    # float32 for the reduction: |fp16| summed over layers x poolings x hidden
    # overflows fp16's 65504 ceiling and reports inf, which compares fine against
    # zero and hides the check. The cheapest sanity assertion in the file was the
    # one that needed a dtype.
    row_mass = np.abs(out.astype(np.float32)).sum(axis=(1, 2, 3))
    assert np.isfinite(row_mass).all(), "non-finite activation mass"
    assert row_mass.min() > 0, "a prompt produced all zeros"

    cache.write(key, out, list(ps.pid))
    print(f"wrote {cache.dir_for(key)} :: {stats.report()}")
    return key, stats
