# Counting characters scores 0.737

Infrastructure for training and evaluating probes on a language model's internal
activations — the lightweight detectors a safety team runs instead of a second
large model. An activation cache keyed on everything that can move a number, a
layer × pooling sweep that never touches a GPU twice, and a control suite that
runs before any result is readable.

**The first number here is a baseline, not a result.** On WildGuard's plainly
worded prompts, a classifier whose only feature is `len(prompt)` scores **0.737
AUC**, because harmful prompts are 1.9× longer than unharmful ones. A probe at
0.963 has not beaten chance by 0.46. It has beaten the free baseline by 0.23 —
and on one of the four evaluation sets below, it loses to that baseline by 18.5
points.

![probe results](results/probe-floor.png)

Three findings, all of them infrastructure problems rather than research ones:

1. **A pooling bug that costs 0.000, 0.014 or 0.083 AUC** depending on two
   settings that live nowhere near the pooling code — and costs *nothing* under
   this project's defaults, which is precisely why it needs a unit test.
2. **The probe loses to character-counting on real jailbreak attempts**: 0.731
   against a 0.916 length floor. Without that control it reads as a working
   detector.
3. **Two label columns over the same prompts ran the model twice**, because the
   cache key included the label set. Activations do not depend on labels.

Everything ran on a laptop. Total compute: **90 seconds of forward passes.**
Every number after that is CPU.

## Setup

| | |
|---|---|
| Model | Qwen3-0.6B, fp16, 28 blocks (`MODEL=` is the only thing that changes it) |
| Features | residual stream at every layer, 4 poolings, cached once |
| Probe | logistic regression on standardised features, 5-fold out-of-fold |
| Train | WildGuard plain prompts, n=903, 45.7% harmful |
| Shift | WildGuard jailbreak-framed prompts, n=796 — same harms, new phrasing |
| OOD | ToxicChat, n=2,803 real user turns, **human**-annotated |
| Interval | stratified bootstrap, 600 draws |

## Extraction — the only step that needs a GPU

| prompt set | n | padding | batching | pad tokens | prompts/s | wall |
|---|---|---|---|---|---|---|
| wildguard-vanilla | 903 | left | length-sorted | 2.3% | 118 | 7.7s |
| wildguard-adversarial | 796 | left | length-sorted | 1.5% | 29 | 27.2s |
| toxicchat (both label columns) | 2803 | left | length-sorted | 2.1% | 75 | 37.2s |
| wildguard-vanilla | 903 | right | length-sorted | 2.3% | 106 | 8.5s |
| wildguard-vanilla | 903 | right | dataset order | 29.0% | 99 | 9.1s |

**90 seconds total, five passes.** Then **116 (layer, pooling) probes in 35
seconds of CPU with the GPU idle** — that ratio is the only reason to build a
cache at all.

Prompts here run 30 to 3,000 characters and the adversarial split averages six
times the vanilla one, so batching in dataset order pads most of the compute.
Sorting by token length first cuts padding from **29.0% to 2.3%**. On the matched pair
(right padding, sorted vs unsorted) that is worth **6.5% of wall clock** — a 0.6B
forward pass on MPS is dominated by per-kernel overhead rather than by tokens, so
the padding fraction is the figure that transfers to a larger model, not the
speedup.

## The pooling bug, priced three ways

`h[:, -1]` and `h.mean(1)` are what most code does. Both are wrong on a padded
sequence. Both are cached here beside the correct versions, so the cost is a
measured number rather than an assertion.

| configuration | pad tokens | `last` (correct) | `h[:, -1]` (the bug) | cost | best layer |
|---|---|---|---|---|---|
| left padding, length-sorted | 2.3% | 0.955 | 0.955 | **0.000** | L16 → L16 |
| right padding, length-sorted | 2.3% | 0.955 | 0.941 | **−0.014** | L16 → L16 |
| right padding, dataset order | 29.0% | 0.955 | 0.872 | **−0.083** | L16 → **L28** |

Three things worth saying out loud:

- **Under left padding the bug is free.** `h[:, -1]` lands on the last real token
  by accident, and the two arrays are bit-identical. A reviewer checking the
  number would find nothing wrong.
- **The cost is set by the batching strategy**, which is chosen for throughput by
  someone who is not thinking about pooling. Sorting by length is a good idea
  that makes a correctness bug 6× smaller and correspondingly harder to find.
- **It moves the answer, not just the score.** In the third row the best layer
  moves from 16 to 28, so a figure captioned "harm is represented in the last
  layers" would be reporting an artifact of padding.

That is why `tests/test_pooling.py` pins the arithmetic on a hand-built tensor
instead of trusting the metric: the metric agrees with the bug under the
configuration this repo actually runs.

## Layer sweep

Length-only floor **0.737** [0.701, 0.768] · 116 probes in **35s of CPU**

| pooling | best layer | AUC [95% CI] | vs length floor |
|---|---|---|---|
| `last` (last real token) | L16 | 0.955 [0.942, 0.968] | +0.219 |
| `mean` (masked mean) | L15 | 0.963 [0.950, 0.975] | **+0.227** |
| `h[:, -1]` | L16 | 0.955 [0.942, 0.968] | +0.219 |
| `h.mean(1)` | L16 | 0.963 [0.950, 0.974] | +0.226 |

Layer 0 is the embedding output — no transformer block has run. `mean` there
already scores **0.877**, well above the 0.737 length floor. Most of what this
probe detects is available from a bag of token embeddings; the 28 blocks add
0.086.

## Generalisation — one probe, fit once on plain prompts

| scored on | n | AUC [95% CI] | length floor | clears it? |
|---|---|---|---|---|
| plain prompts (out-of-fold) | 903 | 0.963 [0.951, 0.974] | 0.737 | yes, +0.227 |
| jailbreak-framed prompts | 796 | 0.795 [0.763, 0.825] | 0.684 | yes, +0.110 |
| real user turns, toxicity | 2803 | 0.732 [0.707, 0.758] | 0.664 | yes, +0.068 |
| **real jailbreak attempts** | 2803 | 0.731 [0.696, 0.762] | **0.916** | **no, −0.185** |

The last row is the one that matters, and it is the reason the control suite
exists rather than the figure.

A probe scoring 0.731 AUC at detecting jailbreak attempts in real traffic reads
as a usable detector. It is beaten by **18.5 points** by counting the characters
in the prompt. Real jailbreak attempts are long — DAN-style preambles, nested
roleplay, pasted system prompts — so length is an extremely strong signal in that
distribution, and the model-internals probe never learned it because its training
split did not have it. Reported without the floor, this number would have been a
result. Reported with it, it is a reason not to ship.

Note what the controls did *not* do: on the plain-prompt split they **cleared**
the probe. Within each length quartile the probe still scores 0.936–0.958, so
there the 0.963 is not a length detector. A control suite that found a problem
everywhere would be useless.

## The controls

| control | rules out | result |
|---|---|---|
| `length_only` | the probe is a length detector | **0.737** — quoted beside every number |
| `shuffled_labels` | the evaluation leaks | 0.503 [0.466, 0.542] — PASS |
| `random_features` | the harness itself is broken | 0.511 [0.471, 0.551] — PASS |
| `length_matched` | length explains the within-split result | 0.936–0.958 across quartiles |

`python -m probeinfra controls` **exits non-zero when a control fails**, so it
belongs in CI rather than in a notebook. That is the difference between a check
and a habit.

`length_matched` stratifies rather than residualises, on purpose. Fitting length
out linearly over-corrects — on an earlier project that flipped the sign of the
effect being measured — so the conservative form of the question gets asked
instead, and a length band containing only one class is reported as `n/a` rather
than quietly dropped.

## The cache key is the whole argument

```python
CacheKey(model_id, model_sha, tokenizer_sha, dtype, layers, poolings,
         padding_side, max_length, chat_template_sha,
         prompt_set, prompt_fingerprint, schema, extras)
```

`model_id` is not an identity: `"Qwen/Qwen3-8B"` resolves to different weights
across a repo revision. `model_sha` is the **resolved commit**, so a model that
moved under you is a cache *miss*, not a silent hit. Everything else in that list
changes activations too — dtype changes rounding, `padding_side` changes which
vector `h[:, -1]` reads, `max_length` changes which prompts lose their tail, and
`extras["length_sorted"]` changes how much padding each row sees.

Two things are deliberately **excluded**:

- **`prompt_set` is in the manifest and out of the digest.** ToxicChat's
  `toxicity` and `jailbreaking` columns annotate the same 2,803 user turns. Keyed
  on the set name they asked for two digests and the model ran twice for
  byte-identical output — **37.2 of 127.0 seconds**, 29% of this run's GPU time. Prompts are
  already identified by a hash of their text.
- **Nothing about the probe.** Fitting is cheap and extraction is not; a new
  regularisation strength must never trigger a forward pass.

Three design facts, each of which came out of a bug:

1. **The key is computable without the weights.** The first version built it
   inside the extraction function from the loaded tokenizer, so nothing could
   look up the cache without a GPU — and the lookup path built a *second* key
   with a different tokenizer hash, so every `sweep` reported a miss on what
   `extract` had just written. A cache whose key is not reproducible is not a
   cache.
2. **`verify()` re-reads the stored key and compares it field by field** on every
   read. The threat is not a digest collision; it is someone editing a field,
   reusing the directory, and getting an answer about the old configuration.
3. **Array shape is checked against the key on write.** An array whose layer
   count disagrees with `key.layers` would index cleanly and mean something else.

### Why `hidden_states` and not forward hooks

Hooks are the usual answer and they are where off-by-one layer indexing comes
from: a hook on `model.layers[k]` fires on that block's *output*, which is the
residual stream entering block k+1. `hidden_states[k]` is unambiguous — 0 is the
embedding output, k is the output of block k. A layer number in a figure should
mean one thing.

## What is in here

| | |
|---|---|
| `probeinfra/cache.py` | content-addressed activation cache; the key is the design |
| `probeinfra/extract.py` | one forward pass, every layer, four poolings, length-sorted batches |
| `probeinfra/probe.py` | logistic probe, stratified bootstrap AUC, three eval settings |
| `probeinfra/controls.py` | the four controls, and why one stratifies instead of residualising |
| `probeinfra/data.py` | loaders that refuse nulls and deduplicate prompts |
| `probeinfra/cli.py` | `extract` / `sweep` / `controls` / `transfer` |
| `tests/` | 40 tests, most of them about the ways a probe pipeline lies |

```bash
pip install -e .
python -m probeinfra extract  --model Qwen/Qwen3-0.6B --set wildguard-vanilla   # the only GPU step
python -m probeinfra sweep    --model Qwen/Qwen3-0.6B --set wildguard-vanilla   # 116 probes, CPU
python -m probeinfra controls --model Qwen/Qwen3-0.6B --layer 16 --pooling mean # exits 1 on failure
python -m probeinfra transfer --model Qwen/Qwen3-0.6B --layer 16 --pooling mean
./run_all.sh                                                                     # all of the above
python tables.py && python make_figures.py                                       # every number above
```

`tables.py` prints every table in this README from the JSON the runs wrote, so a
re-run that changes a result changes the table rather than silently disagreeing
with the prose.

## Two bugs this repo found in itself

**An fp16 sanity check that could not fail.** `extract` asserts no prompt
produced an all-zero activation, via `np.abs(out).sum(...)`. Summed over 29
layers × 4 poolings × 1024 hidden, ordinary activations pass fp16's 65504
ceiling, so the reduction returned `inf` — and `inf > 0` is `True`. The guard
passed on every input including a genuinely empty one. One `astype(np.float32)`.

**A cache that could not be looked up.** Described above. It cost an afternoon's
confusion and produced the design rule the rest of the repo is built on: the key
is derived from config and tokenizer metadata, never from the loaded model, and
exactly one function computes it.

Both are regression-tested.

## Honest limitations

- **The labels are model-produced.** WildGuard's prompt-harm labels come from
  GPT-4, audited at 92% agreement with human annotators on a 500-item sample.
  That agreement rate is a ceiling on any AUC measured against them, and it is
  not estimated here. ToxicChat's labels *are* human, which is why it is the
  out-of-distribution set rather than the training one.
- **One model family, one task.** The stack is model-agnostic — `MODEL=` is the
  only thing `run_all.sh` needs — but every number above is Qwen3-0.6B, which is
  small enough that "model internals" means less than it would at 70B.
- **A linear probe is the floor of this method, not its ceiling.** Attention
  probes and small MLPs do better in the literature. Nothing here argues
  otherwise; the harness fits whatever `fit_probe` returns.
- **The shift result is not diagnosed.** The probe drops from 0.963 to 0.795 on
  jailbreak-framed prompts. Whether that is the probe, the 6× length difference
  between the splits, or genuinely different internal representations is not
  resolved — the length-matched control narrows it and does not settle it.
- **`max_length=1024` truncates.** The adversarial split reaches 3,013 characters
  and a handful of prompts lose their tail. Truncation length is in the cache
  key, so raising it is a new key rather than a silent change.

## Data

- [`walledai/WildGuardTest`](https://huggingface.co/datasets/walledai/WildGuardTest)
  — 1,699 usable prompts (26 ship a null label and are dropped, not cast)
- [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat)
  — 2,803 human-annotated user turns, 12.6% toxic, 3.2% jailbreak attempts

Both are public safety-research datasets. This repo trains detectors on them; it
does not generate, improve, or evaluate attacks.
