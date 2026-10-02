# Probe infrastructure for misuse detection — design

**Date** 2026-10-02 · **Status** implemented (move 1)

## The question this answers

Safeguards-style misuse detection trains lightweight classifiers on a model's
internal activations rather than on its text output. The research loop is: pick a
layer, pick a pooling, fit, read an AUC, repeat. That loop is cheap in compute and
expensive in trust, because every step has a way of returning a plausible number
for the wrong reason.

So the deliverable is not a probe. It is the stack the probe runs on, and the
checks that make its output mean something.

## Design decisions, and what each one is defending against

### 1. The cache key is a hash of everything that can move an activation

`model="X"` is not an identity. The same string resolves to different weights
across a repo revision; the same weights give different activations across dtype,
padding side, truncation length, chat template and batching. Every one of those
is in the key, and the model's **resolved commit sha** is in it rather than a
branch name.

Consequence: a model that moved under you is a cache *miss*. That is the whole
mechanism behind "results stay trustworthy as models evolve" — not a convention,
a key.

### 2. The key is computable without the weights

An earlier version built the key inside the extraction function, from the loaded
tokenizer. Nothing could then look up the cache without a GPU, which destroys the
point of having one. Worse, the lookup path built a *second* key with a different
tokenizer hash, so `sweep` reported a miss on what `extract` had just written.

`resolve_key()` now derives everything from the config and tokenizer — a few KB
over the network — and both paths call it. A cache whose key is not reproducible
is not a cache.

### 3. The key excludes what cannot move an activation

ToxicChat's `toxicity` and `jailbreaking` columns label the **same 2,803 user
turns**. Keyed on the set name, those asked for two digests and the model ran
twice for byte-identical output: 37.2 of 127.0 seconds of GPU time, 29% of the
run. Prompts
are identified by a hash of their text, so the label-set name is metadata, not
identity. It stays in the manifest and leaves the digest.

### 4. Both wrong poolings are implemented and cached on purpose

`h[:, -1]` and `h.mean(1)` are what most code does. They are wrong whenever the
sequence is padded, and `pool()` ships them beside the correct versions so the
cost of the mistake is a measured number in a table rather than an assertion in
prose.

The finding that justifies this: the same bug costs **0.000, 0.014 or 0.083 AUC**
depending on padding side and batch ordering — two settings that live nowhere
near the pooling code. Under the project's default configuration it costs
*nothing*, which is exactly why it needs a unit test and not a glance at the
metric.

### 5. `hidden_states`, not forward hooks

Hooks are the usual answer and they are where off-by-one layer indexing comes
from: a hook on `model.layers[k]` fires on that block's output, i.e. the residual
stream entering block k+1. `hidden_states[k]` is unambiguous. A layer number in a
published figure should mean one thing.

### 6. Length-sorted batching, with the sort reversible

Prompts run 30 to 3,000 characters and the adversarial split averages six times
the vanilla one, so dataset-order batching pads most of the compute. Sorting cuts
padding from 29.0% to 2.3%.

`--no-sort` is kept, for two reasons: it is the only way to price the sort, and it
is the configuration under which the pooling bug is large enough to notice. The
optimisation and the bug share a knob, so neither is safe to judge alone — and the
batching flag is therefore in the cache key.

### 7. Controls before results, and the length floor first

On WildGuard-vanilla, **counting characters scores 0.737 AUC**: harmful prompts
are 1.9x longer than unharmful ones. Any probe number read without that floor
beside it is unreadable. The suite is:

| control | what it rules out | passes when |
|---|---|---|
| `length_only` | the probe is a length detector | reported as a floor, always |
| `shuffled_labels` | the evaluation leaks | CI contains 0.5 |
| `random_features` | the harness itself is broken | CI contains 0.5 |
| `length_matched` | length explains the within-split result | per-quartile AUC holds up |

`length_matched` stratifies rather than residualises on purpose. Fitting length
out linearly over-corrects — on an earlier project that flipped the sign of the
effect being measured — so the conservative version of the question is asked
instead, and single-class bands are reported as `n/a` rather than dropped.

### 8. Three evaluation settings, not one

`iid` (out-of-fold on plain prompts), `shift` (fit on plain, score
jailbreak-framed), `ood` (score real ToxicChat user turns). A probe reported only
on `iid` is reported on the easiest of the three, and `shift` is the setting a
deployed detector actually lives in: the phrasing moves, the harm does not.

## What is deliberately not here

- No probe architecture research. Logistic regression on standardised features,
  one regularisation path. The contribution is the stack, not the classifier.
- No adjudicated ground truth. WildGuard's prompt-harm labels are GPT-4-generated
  with 92% human agreement, which puts a ceiling on any measured AUC. Named in the
  README's limitations, not estimated here.
- No serving path. Pricing a probe against a guard model is move 3.
